import os
import shutil
import subprocess
import time
import uuid
import datetime
from pathlib import Path

import redis

_QUEUE_KEY = "tipgan:queue"
_TASK_KEY_FMT = "tipgan:task:{}"
# 统一输出根目录（无版本前缀）
OUTPUT_ROOT = Path("api_managed_outputs") / "tipgan"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def get_redis() -> redis.Redis:
    """创建 Redis 连接"""
    return redis.Redis(
        host=os.getenv("TIPGAN_REDIS_HOST", "redis"),
        port=int(os.getenv("TIPGAN_REDIS_PORT", "6379")),
        db=int(os.getenv("TIPGAN_REDIS_DB", "0")),
        decode_responses=True
    )


def _train(task: dict) -> tuple[bool, str]:
    """执行训练任务，返回 (成功, 权重路径或错误)"""
    img_path = task["img_path"]
    total_iter = int(task.get("total_iter", 300))
    texture_name = Path(img_path).stem
    job_id = task["id"]  # 使用 task_id 作为 job_id
    meta = f"task-{job_id}"
    
    cmd = [
        "python", "train.py",
        "--img_path", img_path,
        "--total_iter", str(total_iter),
        "--save_step", str(total_iter),
        "--log_step", "100",
        "--vis_step", str(total_iter),
        "--save_base", "experiments",
        "--save_meta", meta,
    ]
    
    try:
        subprocess.check_call(cmd)
        
        root = Path("experiments") / texture_name
        candidates = sorted(root.glob(f"{meta}-*"), reverse=True)
        if not candidates:
            return False, "权重目录未找到"
        
        ckpt_path = candidates[0] / f"checkpoint-iter_{total_iter}.pth"
        if not ckpt_path.is_file():
            return False, f"权重不存在: {ckpt_path}"
        
        return True, str(ckpt_path)
    except subprocess.CalledProcessError as exc:
        return False, f"训练失败: {exc}"


def worker():
    """Worker 循环：消费 Redis 队列任务，执行训练"""
    r = get_redis()
    while True:
        popped = r.brpop(_QUEUE_KEY, timeout=5)
        if popped is None:
            continue
        
        _, task_id = popped
        task_key = _TASK_KEY_FMT.format(task_id)
        task = r.hgetall(task_key)
        if not task:
            continue
        
        start_ts = time.time()
        r.hset(task_key, mapping={"status": "running", "started_at": start_ts})
        
        ok, info = _train(task)
        if ok:
            try:
                job_id = task_id  # task_id 作为 job_id
                date_str = datetime.date.today().strftime("%Y%m%d")
                job_dir = OUTPUT_ROOT / date_str / job_id
                job_dir.mkdir(parents=True, exist_ok=True)
                # 复制 checkpoint 为统一命名
                src_ckpt = Path(info)
                dst_ckpt = job_dir / "checkpoint.pth"
                shutil.copy2(src_ckpt, dst_ckpt)
                # 如果有训练可视化，复制为 train_preview.jpg
                preview = ""
                try:
                    total_iter = task.get("total_iter", "300")
                    candidate = src_ckpt.parent / "imgs" / f"{total_iter}.jpg"
                    if candidate.is_file():
                        dst_preview = job_dir / "train_preview.jpg"
                        shutil.copy2(candidate, dst_preview)
                        preview = str(dst_preview)
                except Exception:
                    preview = ""
                finished_ts = time.time()
                r.hset(
                    task_key,
                    mapping={
                        "status": "done",
                        "ckpt_path": str(dst_ckpt),
                        "preview_path": preview or "",
                        "finished_at": finished_ts,
                        "duration_sec": round(finished_ts - start_ts, 3),
                    }
                )
            except Exception as exc:
                r.hset(
                    task_key,
                    mapping={
                        "status": "failed",
                        "error": f"产物整理失败: {exc}",
                        "finished_at": time.time(),
                    }
                )
        else:
            r.hset(
                task_key,
                mapping={
                    "status": "failed",
                    "error": info,
                    "finished_at": time.time(),
                }
            )


if __name__ == "__main__":
    worker()
