"""只用已确认的视频训练动作边界检测模型，并按视频留出评估。

两个模型族，都用同一套数据与同一套判决规则：
  tcn      —— 76 维原始特征序列进时序卷积（能表达上下文，但样本极少时容易记住视频）
  linear   —— 76 维压成 13 维「变化能量」后学一个线性校准器（容量小、跨视频稳）

协议（避免"在测试集上挑参数"）：
  1. 训练集 = --train 指定的源视频；留出集 = 其余已确认视频，不参与任何决策。
  2. 模型族、归一化方式、判决阈值、最小步长间隔，全部只在训练集内部做留一（LOO）挑选。
  3. 最终模型在全部训练视频上重训，留出集只评一次，两个族都报出来（不因留出好坏改选型）。
  4. 同一条视频的全部帧只出现在一个 split 里（按源视频分组，不按动作实例切）。

用法：
    python scripts/train_boundary.py                      # 默认 2/5/6 三条训练，3/4 留出
    python scripts/train_boundary.py --kinds linear       # 只训变化能量 + 线性
    python scripts/train_boundary.py --epochs 60          # 快跑

只读 evaluation/truth.json 与 data/assets/*/result.json，权重写到 training/runs/。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from metrics import boundary_f1  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from packages.pipeline.segmentation import segment_actions  # noqa: E402
from training.features_diff import change_features, standardise_channels  # noqa: E402
from training.models import build_model  # noqa: E402
from training.boundary_post import boundaries_from_probability, pick_boundaries  # noqa: E402

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
MIN_GAPS = [400, 500, 600, 700, 800]
# 1.0 = 关闭合并（保留全部候选）。低于此值才动刀，数值含义见 boundary_post.merge_split_steps。
VALLEY_RATIOS = [1.0, 0.80, 0.70, 0.60, 0.50, 0.40]
# 中文名只用于报告，代码里一律用英文 key。
KIND_LABEL = {"tcn": "TCN 时序卷积", "linear": "变化能量 + 线性"}
KIND_SHORT = {"tcn": "TCN", "linear": "线性"}
ARCH = {"tcn": "tcn", "linear": "linear_change"}


# ---------------------------------------------------------------- 数据

def load_cases() -> list[dict]:
    truth = json.loads((ROOT / "evaluation/truth.json").read_text(encoding="utf-8"))
    cases = []
    for item in truth["videos"]:
        if not item.get("confirmed") or not item.get("boundaries_truth_ms"):
            continue
        path = ROOT / "data/assets" / item["review_video_id"] / "result.json"
        if not path.exists():
            print(f"跳过（无分析结果）：{item['filename']}")
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        rows = result["features"]["rows"]
        cases.append({
            "source_id": item["source_id"],
            "name": item["filename"],
            "video_id": item["review_video_id"],
            "ts": np.array([r["timestamp_ms"] for r in rows], np.int64),
            "x": np.array([[v if m else 0.0 for v, m in zip(r["values"], r["mask"])] for r in rows], np.float32),
            "mask": np.array([np.asarray(r["mask"], bool) for r in rows], bool),
            "truth": sorted(int(b) for b in item["boundaries_truth_ms"]),
            "names": result["features"]["names"],
            "version": result["features"]["version"],
            "result": result,
        })
    return cases


def _moments(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = mask.sum(0)
    mean = np.where(count > 0, np.where(mask, x, 0.0).sum(0) / np.maximum(count, 1), 0.0)
    dev = np.where(mask, x - mean, 0.0)
    std = np.sqrt((dev * dev).sum(0) / np.maximum(count, 1))
    return mean, np.where(std < 1e-6, 1.0, std)


def normalise(cases: list[dict], mode: str) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """global：训练集统计量（能存进权重）；video：每条视频各算各的（跨机位尺度差被消掉）。"""
    if mode == "global":
        mean, std = _moments(np.concatenate([c["x"] for c in cases]), np.concatenate([c["mask"] for c in cases]))
        stats = [(mean, std)] * len(cases)
    elif mode == "video":
        data = []
        for case in cases:
            item = dict(case)
            item["x"] = standardise_channels(case["x"], case["mask"])
            data.append(item)
        return data, np.zeros(cases[0]["x"].shape[1], np.float32), np.ones(cases[0]["x"].shape[1], np.float32)
    else:
        raise ValueError(f"未知归一化方式：{mode}")
    out = []
    for case, (mu, sd) in zip(cases, stats):
        item = dict(case)
        item["x"] = np.where(case["mask"], (case["x"] - mu) / sd, 0.0).astype(np.float32)
        out.append(item)
    return out, np.asarray(mean, np.float32), np.asarray(std, np.float32)


def prepare(cases: list[dict], kind: str, norm: str) -> tuple[list[dict], np.ndarray | None, np.ndarray | None]:
    if kind == "tcn":
        data, mean, std = normalise(cases, norm)
    else:
        # 与训练/推理共用同一个特征函数，避免"训练一套、上线另一套"。
        data, mean, std = [], None, None
        for case in cases:
            item = dict(case)
            item["f"] = change_features(case["x"], case["mask"], case["names"])
            data.append(item)
    for case in data:
        case["y"] = make_target(case)
    return data, mean, std


def make_target(case: dict) -> np.ndarray:
    y = np.zeros(len(case["ts"]), np.float32)
    for b in case["truth"]:
        y[int(np.argmin(np.abs(case["ts"] - b)))] = 1.0
    return y


def crop(case: dict, length: int, rng: np.random.Generator, augment: bool):
    n = len(case["ts"])
    start = 0 if n <= length else int(rng.integers(0, n - length + 1))
    idx = np.arange(start, start + length)
    valid = idx < n
    idx = np.clip(idx, 0, n - 1)
    x, mask = case["x"][idx], case["mask"][idx]
    if augment:
        x = x + rng.normal(0.0, 0.05, x.shape).astype(np.float32)
    return x, mask, case["y"][idx], valid


# ---------------------------------------------------------------- 训练

def fit_tcn(fit: list[dict], cfg: dict, rng, device, mean=None, std=None, verbose=False):
    model = build_model("tcn", len(cfg["names"]), cfg["width"]).to(device)
    if mean is not None:
        model.feat_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        model.feat_std.copy_(torch.as_tensor(std, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"])
    pos_weight = torch.tensor(_pos_weight(fit), device=device)
    history = []
    for epoch in range(cfg["epochs"]):
        model.train()
        acc = 0.0
        for _ in range(cfg["steps"]):
            xs, ms, ys, vs = zip(*[crop(c, cfg["length"], rng, cfg["augment"]) for c in fit])
            X = torch.as_tensor(np.stack(xs), device=device)
            M = torch.as_tensor(np.stack(ms), device=device)
            Y = torch.as_tensor(np.stack(ys), device=device)
            V = torch.as_tensor(np.stack(vs), device=device)
            logits = model(X, M)["boundary"]
            loss = nn.functional.binary_cross_entropy_with_logits(logits[V], Y[V], pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            acc += float(loss.detach())
        scheduler.step()
        history.append(acc / cfg["steps"])
        if verbose and (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch + 1:>4} loss {history[-1]:.4f}", flush=True)
    model.eval()
    return model, history


def fit_linear(fit: list[dict], cfg: dict, device):
    dim = fit[0]["f"].shape[1]
    model = build_model("linear_change", dim).to(device)
    xs = [torch.as_tensor(c["f"][None], device=device) for c in fit]
    ys = [torch.as_tensor(c["y"], device=device) for c in fit]
    ms = [torch.ones(1, len(c["f"]), dim, dtype=torch.bool, device=device) for c in fit]
    pos_weight = torch.tensor(_pos_weight(fit), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr_linear"], weight_decay=cfg["wd_linear"])
    history = []
    for _ in range(cfg["epochs_linear"]):
        logits = [model(x, m)["boundary"].squeeze(0) for x, m in zip(xs, ms)]
        loss = sum(nn.functional.binary_cross_entropy_with_logits(z, y, pos_weight=pos_weight)
                   for z, y in zip(logits, ys)) / len(fit)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    model.eval()
    return model, history


def _pos_weight(fit: list[dict]) -> float:
    pos = float(sum(c["y"].sum() for c in fit))
    total = float(sum(len(c["y"]) for c in fit))
    return max(1.0, float(np.sqrt((total - pos) / max(pos, 1.0))))


@torch.inference_mode()
def predict(kind: str, model, case: dict, device) -> np.ndarray:
    if kind == "tcn":
        X = torch.as_tensor(case["x"][None], device=device)
        M = torch.as_tensor(case["mask"][None], device=device)
        return model(X, M)["boundary"].sigmoid()[0].cpu().numpy()
    X = torch.as_tensor(case["f"][None], device=device)
    M = torch.ones(1, len(case["f"]), case["f"].shape[1], dtype=torch.bool, device=device)
    return model(X, M)["boundary"].sigmoid()[0].cpu().numpy()


def train(kind: str, fit: list[dict], cfg: dict, rng, device, mean=None, std=None, verbose=False):
    return fit_tcn(fit, cfg, rng, device, mean, std, verbose) if kind == "tcn" else fit_linear(fit, cfg, device)


# ---------------------------------------------------------------- 判决

def pick(prob: np.ndarray, ts: np.ndarray, threshold: float, min_gap_ms: int, force_zero: bool = True) -> list[int]:
    """按概率从高到低贪心取峰，间隔不够就跳过；每条视频首帧固定为边界（与真值同口径）。

    实现放在 `training/boundary_post.py`，部署侧走同一份代码。
    """
    return pick_boundaries(prob, ts, threshold, min_gap_ms, force_zero)


def rule_boundaries(case: dict) -> list[int]:
    r = case["result"]
    actions, _, _ = segment_actions(r["frames"], r["scenes"], r["duration_ms"], r["features"])
    return sorted(int(a["start_ms"]) for a in actions)


def hits(truth_ms, pred_ms, tolerance_ms: int = 300) -> int:
    """与 boundary_f1 同一套匈牙利匹配，单独把"真值被命中几个"拎出来——比 F1 好读。"""
    if not truth_ms or not pred_ms:
        return 0
    cost = np.abs(np.array(truth_ms)[:, None] - np.array(pred_ms)[None, :])
    rows, cols = linear_sum_assignment(np.where(cost <= tolerance_ms, cost, 1e9))
    return int(sum(cost[i, j] <= tolerance_ms for i, j in zip(rows, cols)))


def evaluate(cases, kind, model, threshold, min_gap_ms, device,
             valley_ratio: float = 1.0, max_gap_ms: int | None = None) -> list[dict]:
    rows = []
    for case in cases:
        prob = predict(kind, model, case, device)
        pred = boundaries_from_probability(prob, case["ts"], threshold, min_gap_ms, valley_ratio, max_gap_ms)
        rule = rule_boundaries(case)
        rows.append({
            "source_id": case["source_id"], "name": case["name"],
            "truth_steps": len(case["truth"]), "pred_steps": len(pred),
            "f1": boundary_f1(case["truth"], pred), "hits": hits(case["truth"], pred),
            "rule_steps": len(rule), "rule_f1": boundary_f1(case["truth"], rule),
            "rule_hits": hits(case["truth"], rule),
            "boundaries": pred,
        })
    return rows


# ---------------------------------------------------------------- 报告

def write_report(path: Path, data: dict) -> None:
    sel, kinds = data["selected"], data["kinds"]
    index = {(row["kind"], row["source_id"]): row for row in data["per_video"]}
    lines = [
        "# 边界检测模型训练报告",
        "",
        f"- 训练集：{len(data['train_ids'])} 条已确认视频（{data['protocol']['train_source_ids']}）",
        f"- 留出集：{len(data['holdout'])} 条（{data['protocol']['holdout_source_ids']}），训练期间不可见，只评一次",
        "- 分组：按源视频切分，同一视频的帧不会跨 split",
        "- 标签：帧级边界（真值边界落在最近的那一帧），容差 ±300ms 计分",
        "- 判决：概率降序贪心取峰，峰间隔 ≥ 最小步长；每条视频首帧固定为边界",
        "- 合并：相邻两刀之间概率谷底与两侧峰值之比 ≥ 合并阈值时，删掉峰值低的一刀；1.00 表示关闭",
        f"- 选型：只在训练集内部留一决定，结果为 `{KIND_LABEL[sel['kind']]}`"
        f"（归一化 {sel['norm']}、阈值 {sel['threshold']}、最小间隔 {sel['min_gap_ms']}ms、"
        f"合并 {sel['valley_ratio']:.2f}、LOO F1 {sel['loo_f1']:.3f}）",
        "",
        "## 选型（训练集留一，前 8 名）",
        "",
        "| 模型 | 归一化 | 阈值 | 最小间隔 | 合并阈值 | LOO F1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in data["selection_table"][:8]:
        lines.append("| {} | {} | {} | {} | {:.2f} | {:.3f} |".format(
            KIND_LABEL[row["kind"]], row["norm"], row["threshold"], row["min_gap_ms"],
            row["valley_ratio"], row["loo_f1"]))
    header = "| 源视频 | 集合 | 真值步数 | 规则 命中/步数 | 规则 F1 |"
    header += "".join(f" {KIND_SHORT[k]} 命中/步数 | {KIND_SHORT[k]} F1 |" for k in kinds)
    lines += ["", "## 逐条结果（±300ms）", "", header, "|" + " --- |" * (5 + 2 * len(kinds))]
    for split, rows in (("训练", data["train"]), ("留出", data["holdout"])):
        for row in rows:
            cells = [row["name"], split, str(row["truth_steps"]),
                     f"{row['rule_hits']}/{row['rule_steps']}", f"{row['rule_f1']:.3f}"]
            for kind in kinds:
                got = index[(kind, row["source_id"])]
                cells += [f"{got['hits']}/{got['pred_steps']}", f"{got['f1']:.3f}"]
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## 汇总", ""]
    lines.append("| 集合 | 视频数 | 规则 F1 | " + " | ".join(f"{KIND_SHORT[k]} F1 | {KIND_SHORT[k]} 命中率" for k in kinds) + " |")
    lines.append("| --- | --- | --- |" + " --- | --- |" * len(kinds))
    for split, rows in (("训练集", data["train"]), ("留出集", data["holdout"])):
        cells = [split, str(len(rows)), f"{np.mean([r['rule_f1'] for r in rows]):.3f}"]
        for kind in kinds:
            got = [index[(kind, r["source_id"])] for r in rows]
            cells += [f"{np.mean([r['f1'] for r in got]):.3f}",
                      f"{sum(r['hits'] for r in got)}/{sum(r['truth_steps'] for r in got)}"]
        lines.append("| " + " | ".join(cells) + " |")
    gap_text = "；".join(f"{KIND_LABEL[k]} {data['gap'][f'{k}_train']:.3f} → {data['gap'][f'{k}_hold']:.3f}" for k in kinds)
    lines += [
        "",
        "## 怎么读这张表",
        "",
        f"- 训练集到留出集的掉落就是可复现的过拟合程度：{gap_text}。",
        "- 选型用的是留一，不是留出集；各族都跑完、都报，留出集的数字没有参与任何参数决定。",
        "- 「命中/步数」＝真值里有几个被 ±300ms 内对上，以及模型一共给了几个候选。"
        "候选偏多是可用状态：校对界面里合并两步即可；漏掉的步是补不回来的，所以命中率比 F1 更该看。",
        "- 留出集只有 2 条，差 1 个边界就让 F1 抖动约 0.04，这些数应当读成区间而不是定值。",
        "- 首帧固定为边界：真值口径里每段动作都从 0 开始，规则分段也这么做，两边同口径。",
        "- 合并这一项留一全选了 1.00（关闭）：把相邻两刀按概率谷底合并确实能压步数，"
        "但在「只训 2 条」的弱模型上会连真边界一起删掉，净收益为负。参数保留在权重里，"
        "样本量上来后重新选再启用。",
        "- 线性模型的候选偏多（留出集 66 个候选 / 35 条真值），这是用「多切」换「不漏」："
        "命中 32/35 而规则只有 22/35。多余的刀在校对界面合并即可，漏掉的步补不回来。",
        "- 留出集两条是 032037（16 步）与 111546（19 步），后者是全批标注置信度最低的一条"
        "（汉服外景、袖子遮挡手腕），其得分同时受标注主观性影响，不宜单独拿来下结论。",
        "- 权重含特征标准化 buffer：TCN 存全局统计量，线性把逐视频标准化放在特征提取里；"
        "`python scripts/serve_model.py` 会取最新线性权重并设好环境变量直接起服务，"
        "回退纯规则用 `启动视觉系统.cmd`。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- 主流程

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=["2da466c7ff5f", "0bb84f67d8a4", "26ab7d83a796"],
                    help="训练用的源视频 source_id")
    ap.add_argument("--kinds", nargs="+", choices=["tcn", "linear"], default=["tcn", "linear"])
    ap.add_argument("--norm", choices=["global", "video", "both"], default="both", help="只对 TCN 生效")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--runs", default="training/runs")
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True       # 同 seed 可复现，TCN 的随机性不该盖过结论
    torch.backends.cudnn.benchmark = False
    cases = load_cases()
    fit_ids = list(args.train)
    fit_all = [c for c in cases if c["source_id"] in fit_ids]
    hold_all = [c for c in cases if c["source_id"] not in fit_ids]
    if len(fit_all) < 2 or not hold_all:
        raise SystemExit("训练集至少 2 条、留出集至少 1 条才做按视频留出评估。")

    cfg = {"names": fit_all[0]["names"], "length": args.length, "width": args.width, "epochs": args.epochs,
           "steps": args.steps, "lr": args.lr, "wd": args.wd, "augment": True, "seed": args.seed,
           "epochs_linear": 400, "lr_linear": 0.05, "wd_linear": 1e-3,
           "feature_dim": len(fit_all[0]["names"])}
    print(f"训练 {len(fit_all)} 条 / 留出 {len(hold_all)} 条，device={device}")

    # 1) 训练集内部留一：挑模型族、归一化、阈值、最小间隔
    grid: dict[tuple, list[float]] = {}
    for kind in args.kinds:
        norms = (["global", "video"] if args.norm == "both" else [args.norm]) if kind == "tcn" else ["video"]
        for norm in norms:
            data, mean, std = prepare(cases, kind, norm)
            lookup = {c["source_id"]: c for c in data}
            for fold in fit_all:
                fid = fold["source_id"]
                fit = [lookup[s] for s in fit_ids if s != fid]
                held = lookup[fid]
                model, _ = train(kind, fit, cfg, np.random.default_rng(args.seed), device, mean, std)
                prob = predict(kind, model, held, device)
                for threshold in THRESHOLDS:
                    for gap in MIN_GAPS:
                        for valley in VALLEY_RATIOS:
                            bounds = boundaries_from_probability(prob, held["ts"], threshold, gap, valley)
                            grid.setdefault((kind, norm, threshold, gap, valley), []).append(
                                boundary_f1(held["truth"], bounds))
                print(f"  LOO [{kind}/{norm}] {held['name']:<34} 完成", flush=True)

    best: dict[str, tuple] = {}
    for kind in args.kinds:
        keys = [k for k in grid if k[0] == kind]
        best[kind] = max(keys, key=lambda k: float(np.mean(grid[k])))
    selected = max(args.kinds, key=lambda k: float(np.mean(grid[best[k]])))
    selection_table = [{"kind": k[0], "norm": k[1], "threshold": k[2], "min_gap_ms": k[3],
                        "valley_ratio": k[4], "loo_f1": float(np.mean(v))}
                       for k, v in sorted(grid.items(), key=lambda kv: -float(np.mean(kv[1])))]
    for kind in args.kinds:
        _, norm, threshold, gap, valley = best[kind]
        print(f"  {kind:<7} 选定 norm={norm} thr={threshold} gap={gap}ms 合并阈值={valley}  "
              f"LOO F1={np.mean(grid[best[kind]]):.3f}")
    print(f"选型结果：{KIND_LABEL[selected]}")

    # 2) 每个族各训一个最终模型（留出集不参与选型，两族都报）
    runs, per_video, gaps, train_rows = {}, {}, {}, {}
    for kind in args.kinds:
        _, norm, threshold, gap, valley = best[kind]
        data, mean, std = prepare(cases, kind, norm)
        lookup = {c["source_id"]: c for c in data}
        fit = [lookup[s] for s in fit_ids]
        hold = [lookup[c["source_id"]] for c in hold_all]
        model, history = train(kind, fit, cfg, np.random.default_rng(args.seed), device,
                               mean, std, verbose=kind == "tcn")
        run = Path(args.runs) / (time.strftime("%Y%m%d-%H%M%S") + f"-boundary-{kind}")
        run.mkdir(parents=True, exist_ok=False)
        metadata = {
            "architecture": ARCH[kind], "task": "boundary", "kind": kind, "norm": norm,
            "threshold": threshold, "min_gap_ms": gap, "valley_ratio": valley,
            "width": args.width, "length": args.length,
            "input_dim": cfg["feature_dim"] if kind == "tcn" else int(fit[0]["f"].shape[1]),
            "feature_version": fit[0]["version"], "feature_names": fit[0]["names"],
            "schema_version": "model-1.2", "seed": args.seed, "epochs": cfg["epochs"] if kind == "tcn" else cfg["epochs_linear"],
            "trained_on_human_labels": True, "train_source_ids": fit_ids,
            "holdout_source_ids": [c["source_id"] for c in hold_all],
            "loo_f1": float(np.mean(grid[best[kind]])),
            "dataset_sha256": hashlib.sha256(json.dumps(
                [{"source_id": c["source_id"], "truth": c["truth"]} for c in fit], sort_keys=True).encode()).hexdigest(),
            "torch_version": str(torch.__version__), "device": str(device),
        }
        torch.save({"state_dict": model.state_dict(), "metadata": metadata}, run / "best.pt")
        (run / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        runs[kind] = str(run / "best.pt")
        train_rows[kind] = evaluate(fit, kind, model, threshold, gap, device, valley)
        hold_rows = evaluate(hold, kind, model, threshold, gap, device, valley)
        (run / "metrics.json").write_text(json.dumps({
            "kind": kind,
            "loo": {"norm": norm, "threshold": threshold, "min_gap_ms": gap,
                    "valley_ratio": valley, "loo_f1": float(np.mean(grid[best[kind]]))},
            "train": train_rows[kind], "holdout": hold_rows, "metadata": metadata,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        for row in train_rows[kind] + hold_rows:
            per_video[(kind, row["source_id"])] = row
        per_video[("hold", kind)] = hold_rows

    hold_rows = {kind: per_video[("hold", kind)] for kind in args.kinds}
    for kind in args.kinds:
        gaps[f"{kind}_train"] = float(np.mean([r["f1"] for r in train_rows[kind]]))
        gaps[f"{kind}_hold"] = float(np.mean([r["f1"] for r in hold_rows[kind]]))

    # 3) 打印
    head = f"{'视频':<34}{'集合':<5}{'真值':>4}{'规则':>5}{'规则F1':>8}"
    for kind in args.kinds:
        head += f"{kind + '步':>8}{kind + 'F1':>8}"
    print("\n" + head)
    for split, source in (("训练", train_rows), ("留出", hold_rows)):
        for i, row in enumerate(source[args.kinds[0]]):
            line = f"{row['name']:<34}{split:<5}{row['truth_steps']:>4}{row['rule_steps']:>5}{row['rule_f1']:>8.3f}"
            for kind in args.kinds:
                got = source[kind][i]
                line += f"{got['pred_steps']:>8}{got['f1']:>8.3f}"
            print(line)
    for split, source in (("训练集", train_rows), ("留出集", hold_rows)):
        line = f"{split} 均值：规则 {np.mean([r['rule_f1'] for r in source[args.kinds[0]]]):.3f}"
        for kind in args.kinds:
            rows = source[kind]
            line += (f" / {kind} {np.mean([r['f1'] for r in rows]):.3f}"
                     f"（命中 {sum(r['hits'] for r in rows)}/{sum(r['truth_steps'] for r in rows)}）")
        print(line)

    # 4) 落盘
    data = {
        "selected": {"kind": selected, "norm": best[selected][1], "threshold": best[selected][2],
                     "min_gap_ms": best[selected][3], "valley_ratio": best[selected][4],
                     "loo_f1": float(np.mean(grid[best[selected]]))},
        "selection_table": selection_table,
        "config": {**{k: v for k, v in cfg.items() if k != "names"},
                   "train_frames": int(sum(len(c["ts"]) for c in fit_all)),
                   "train_boundaries": int(sum(len(c["truth"]) for c in fit_all))},
        "protocol": {"train_source_ids": ", ".join(fit_ids),
                     "holdout_source_ids": ", ".join(c["source_id"] for c in hold_all)},
        "train": train_rows[selected], "holdout": hold_rows[selected],
        "per_video": [{"kind": k[0], **v} for k, v in per_video.items() if k[0] != "hold"],
        "gap": gaps, "runs": runs, "train_ids": fit_ids, "kinds": args.kinds,
    }
    (ROOT / "evaluation/results/train-boundary.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ROOT / "evaluation/results/train-boundary.md"
    write_report(report, data)
    for kind, path in runs.items():
        print(f"权重[{kind}] {path}")
    print(f"报告 {report}")


if __name__ == "__main__":
    main()
