"""挂上训练好的边界模型启动服务。

自动取 `training/runs/` 下最新的线性权重，设好 `GESTUREFLOW_MODEL` 与
`GESTUREFLOW_SEGMENTER=model` 再起 uvicorn。想回到纯规则分段，直接用
`启动视觉系统.cmd`（不设这两个变量）即可，两者互不影响。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def latest_checkpoint() -> Path:
    runs = sorted((ROOT / "training" / "runs").glob("*-boundary-linear/best.pt"))
    if not runs:
        raise SystemExit("没有找到训练好的线性模型，先跑 python scripts/train_boundary.py")
    return runs[-1]


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import torch
    import uvicorn

    checkpoint = latest_checkpoint()
    meta = torch.load(checkpoint, map_location="cpu", weights_only=True)["metadata"]
    os.environ["GESTUREFLOW_MODEL"] = str(checkpoint)
    os.environ["GESTUREFLOW_SEGMENTER"] = "model"
    print(f"模型：{checkpoint.relative_to(ROOT)}")
    print(f"参数：阈值 {meta['threshold']}、最小间隔 {meta['min_gap_ms']}ms、"
          f"合并 {meta.get('valley_ratio', 1.0):.2f}、留一 F1 {meta['loo_f1']:.3f}")
    print("分段：模型接管（GESTUREFLOW_SEGMENTER=model）；回退用 启动视觉系统.cmd")
    uvicorn.run("apps.api.main:app", host="127.0.0.1", port=8000, app_dir=str(ROOT))


if __name__ == "__main__":
    main()
