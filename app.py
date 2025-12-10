import base64
import io
import os
import shutil
import datetime
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import redis
import torch
from torchvision import transforms
from torchvision.utils import save_image

from src.dataset import get_transform
from src.network import TiPGANGenerator

# 设置推理设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(ckpt_path: str) -> TiPGANGenerator:
    """从 checkpoint 加载生成器"""
    model = TiPGANGenerator(
        input_nc=3, output_nc=3, ngf=64,
        norm_layer=torch.nn.InstanceNorm2d,
        use_dropout=False, padding_type='reflect'
    )
    state = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state, dict) and 'generator' in state:
        state = state['generator']
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


app = FastAPI(title="TiPGAN API", version="1.0.0")
MODEL_CACHE = {}

# 统一输出根目录（对外暴露）：api_managed_outputs/{YYYYMMDD}/{job_id}
OUTPUT_ROOT = Path("api_managed_outputs")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/static/outputs", StaticFiles(directory=OUTPUT_ROOT), name="outputs")


def _to_public_url(local_path: str | None) -> str | None:
    """将本地路径转换为可访问的静态 URL（相对路径）"""
    if not local_path:
        return None
    try:
        rel = Path(local_path).resolve().relative_to(OUTPUT_ROOT.resolve())
        return f"/static/outputs/{rel.as_posix()}"
    except Exception:
        return local_path


def _get_redis() -> redis.Redis:
    """创建 Redis 连接"""
    return redis.Redis(
        host=os.getenv("TIPGAN_REDIS_HOST", "redis"),
        port=int(os.getenv("TIPGAN_REDIS_PORT", "6379")),
        db=int(os.getenv("TIPGAN_REDIS_DB", "0")),
        decode_responses=True
    )


# 推理预处理
_base_transform = get_transform()
_preprocess = transforms.Compose([
    transforms.CenterCrop(128),
    *_base_transform.transforms
])


def _tensor_to_base64_img(tensor: torch.Tensor) -> str:
    """张量转 Base64 JPEG"""
    buf = io.BytesIO()
    save_image(tensor, buf, format="JPEG", normalize=True, value_range=(-1, 1))
    return base64.b64encode(buf.getvalue()).decode('utf-8')


_QUEUE_KEY = "tipgan:queue"
_TASK_KEY_FMT = "tipgan:task:{}"


@app.post("/train")
async def train(file: UploadFile = File(...), total_iter: int = Form(300)) -> JSONResponse:
    """上传图片，提交训练任务"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少文件名")

    media_dir = Path("media")
    media_dir.mkdir(exist_ok=True)
    img_path = media_dir / file.filename
    
    with img_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    task_id = uuid.uuid4().hex
    task_key = _TASK_KEY_FMT.format(task_id)
    
    payload = {
        "id": task_id,
        "img_path": str(img_path),
        "total_iter": total_iter,
        "status": "pending",
        "created_at": time.time(),
    }
    
    r = _get_redis()
    r.hset(task_key, mapping=payload)
    r.lpush(_QUEUE_KEY, task_id)
    
    return JSONResponse({
        "task_id": task_id,
        "message": f"训练任务已提交: {file.filename}"
    })


@app.post("/infer")
async def infer(task_id: str = Form(...)) -> JSONResponse:
    """使用 task_id 推理"""
    r = _get_redis()
    task_key = _TASK_KEY_FMT.format(task_id)
    task = r.hgetall(task_key)
    
    if not task:
        return JSONResponse({
            "image_base64": None,
            "message": "任务不存在"
        }, status_code=404)

    status = task.get("status")
    if status != "done":
        return JSONResponse({
            "image_base64": None,
            "message": f"训练未完成: {status}"
        }, status_code=400)

    ckpt_path = task.get("ckpt_path")
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return JSONResponse({
            "image_base64": None,
            "message": "权重丢失"
        }, status_code=400)

    img_path = task.get("img_path")
    if not os.path.isfile(img_path):
        return JSONResponse({
            "image_base64": None,
            "message": "图片丢失"
        }, status_code=400)

    try:
        image = Image.open(img_path).convert("RGB")
        input_tensor = _preprocess(image).unsqueeze(0).to(DEVICE)

        if ckpt_path not in MODEL_CACHE:
            MODEL_CACHE[ckpt_path] = _build_model(ckpt_path)
        model = MODEL_CACHE[ckpt_path]

        with torch.no_grad():
            seamless = model(input_tensor)

        # 将推理结果保存到与 checkpoint 同级目录
        job_dir = Path(ckpt_path).parent
        job_dir.mkdir(parents=True, exist_ok=True)
        out_path = job_dir / "infer_seamless.jpg"
        save_image(seamless, str(out_path), format="JPEG", normalize=True, value_range=(-1, 1))

        return JSONResponse({
            "image_output_path": _to_public_url(str(out_path)),
            "ckpt_path": _to_public_url(ckpt_path),
            "message": "推理成功"
        })
    except Exception as e:
        return JSONResponse({
            "image_output_path": None,
            "ckpt_path": _to_public_url(ckpt_path),
            "message": f"推理失败: {str(e)}"
        }, status_code=500)


@app.get("/task_status")
async def task_status(task_id: str) -> JSONResponse:
    """查询任务状态，返回完整信息"""
    r = _get_redis()
    task_key = _TASK_KEY_FMT.format(task_id)
    data = r.hgetall(task_key)
    
    if not data:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 输出路径：训练阶段仅返回权重路径；预览恒为 null
    ckpt_path = data.get("ckpt_path") or None
    preview_path = None
    # 状态文案（对外）
    status_map = {
        "pending": "训练未开始",
        "running": "训练中",
        "done": "训练结束",
        "failed": "训练失败",
    }
    status_label = status_map.get(data.get("status"), data.get("status"))

    return JSONResponse({
        "status": status_label,
        "output_path": _to_public_url(ckpt_path),   # ckpt 对外可访问 URL
        "preview_path": _to_public_url(preview_path),  # 恒为 null
        "total_iter": data.get("total_iter"),
        "duration_sec": data.get("duration_sec"),
        "error": data.get("error"),
    })
