"""Reproducible CE-CSL skeleton / CTC baseline, with resumable feature extraction."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
FEATURE_VERSION = "cecsl-shoulder-hand-local-xy-mask-v1"
PUNCTUATION = set("。，！？、；：,.!?;")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_manifest(root):
    records = []
    for split in ("train", "dev", "test"):
        with (root / "label" / f"{split}.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                tokens = [word.strip() for word in row["Gloss"].split("/")]
                tokens = [word for word in tokens if word and word not in PUNCTUATION]
                path = root / "video" / split / row["Translator"] / (row["Number"] + ".mp4")
                if not path.is_file() or not tokens:
                    raise ValueError(f"Missing video or gloss: {path}")
                records.append({"id": row["Number"], "signer": row["Translator"],
                                "split": split, "path": str(path), "tokens": tokens,
                                "sentence": row["Chinese Sentences"]})
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Video IDs are not unique across splits")
    return records


def init_extractor(models, fps, side):
    global model_directory, extraction_fps, extraction_side
    import cv2
    cv2.setNumThreads(1)
    model_directory, extraction_fps, extraction_side = Path(models), fps, side


def create_detectors():
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    options = {"running_mode": vision.RunningMode.VIDEO}
    detector_hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_buffer=(model_directory / "hand_landmarker.task").read_bytes()),
        num_hands=2, min_hand_detection_confidence=.4, **options))
    detector_pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_buffer=(model_directory / "pose_landmarker.task").read_bytes()),
        num_poses=1, **options))
    return detector_hand, detector_pose


def skeleton_vector(hand_result, pose_result, aspect):
    # Pixel-aspect-aware coordinates. Depth from different detectors is never mixed.
    pose = np.zeros((33, 3), dtype=np.float32)
    if pose_result.pose_landmarks:
        for index, point in enumerate(pose_result.pose_landmarks[0]):
            pose[index] = (point.x * aspect, point.y, float(point.visibility >= .5))
    shoulders = pose[[11, 12]]
    if shoulders[:, 2].all():
        origin = shoulders[:, :2].mean(0)
        scale = max(float(np.linalg.norm(shoulders[0, :2] - shoulders[1, :2])), .05)
    else:
        origin, scale = np.array([aspect / 2, .5]), .3
    pose[:, :2] = (pose[:, :2] - origin) / scale * pose[:, 2:3]
    hands = np.zeros((2, 21, 5), dtype=np.float32)
    occupied = set()
    for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
        slot = 0 if handedness[0].category_name.lower() == "left" else 1
        if slot in occupied:
            continue
        occupied.add(slot)
        xy = np.array([[point.x * aspect, point.y] for point in landmarks], dtype=np.float32)
        palm = max(float(np.linalg.norm(xy[9] - xy[0])), .015)
        hands[slot, :, :2] = (xy - origin) / scale
        hands[slot, :, 2:4] = (xy - xy[0]) / palm
        hands[slot, :, 4] = 1
    vector = np.concatenate((pose.ravel(), hands.ravel()))
    return np.clip(vector, -10, 10)


def extract_one(task):
    import cv2
    import mediapipe as mp
    record, cache = task
    output = Path(cache) / (record["id"] + ".npz")
    if output.exists():
        with np.load(output) as data:
            return record["id"], len(data["features"]), float(data["hand_coverage"])
    capture = cv2.VideoCapture(record["path"])
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        raise ValueError(f"Invalid FPS: {record['path']}")
    frames, timestamps = [], []
    frame_index, next_sample = 0, 0.0
    detector_hand, detector_pose = create_detectors()
    try:
        while capture.grab():
            seconds = frame_index / source_fps
            frame_index += 1
            if seconds + 1e-6 < next_sample:
                continue
            ok, bgr = capture.retrieve()
            if not ok:
                raise ValueError(f"Frame decode failed: {record['path']}")
            next_sample += 1 / extraction_fps
            height, width = bgr.shape[:2]
            factor = min(1, extraction_side / max(height, width))
            bgr = cv2.resize(bgr, None, fx=factor, fy=factor)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            stream_time = round(seconds * 1000)
            hand = detector_hand.detect_for_video(image, stream_time)
            pose = detector_pose.detect_for_video(image, stream_time)
            frames.append(skeleton_vector(hand, pose, width / height))
            timestamps.append(round(seconds * 1000))
    finally:
        capture.release()
        detector_hand.close()
        detector_pose.close()
    if not frames:
        raise ValueError(f"Empty video: {record['path']}")
    features = np.stack(frames).astype(np.float16)
    coverage = float(features[:, 99:].reshape(-1, 2, 21, 5)[:, :, :, 4].mean())
    temporary = output.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, features=features, timestamps_ms=timestamps, hand_coverage=coverage)
    temporary.replace(output)
    return record["id"], len(frames), coverage


def extract(records, cache, models, fps, side, workers):
    cache.mkdir(parents=True, exist_ok=True)
    configuration = {"version": FEATURE_VERSION, "fps": fps, "side": side,
                     "models": {name: hashlib.sha256((models / name).read_bytes()).hexdigest()
                                for name in ("hand_landmarker.task", "pose_landmarker.task")},
                     "manifest_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()}
    metadata = cache / "configuration.json"
    if metadata.exists() and json.loads(metadata.read_text(encoding="utf-8")) != configuration:
        raise ValueError("Feature cache configuration changed. Use a new cache directory.")
    write_json(metadata, configuration)
    started = time.monotonic()
    quality = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init_extractor,
                             initargs=(str(models), fps, side)) as pool:
        tasks = ((record, str(cache)) for record in records)
        for index, (ident, length, coverage) in enumerate(pool.map(extract_one, tasks), 1):
            quality.append({"id": ident, "frames": length, "hand_coverage": coverage})
            if index % 25 == 0 or index == len(records):
                write_json(cache / "progress.json", {"done": index, "total": len(records),
                           "elapsed_seconds": round(time.monotonic() - started), "last": ident})
                print(f"Features {index}/{len(records)} | {time.monotonic() - started:.0f}s", flush=True)
    write_json(cache / "quality.json", quality)


class Sequences(Dataset):
    def __init__(self, records, cache, vocabulary, mean, scale, augment=False):
        self.items = []
        self.augment = augment
        for record in records:
            with np.load(cache / (record["id"] + ".npz")) as data:
                features = data["features"].astype(np.float32)
            target = [vocabulary.get(token, 1) for token in record["tokens"]]
            minimum = len(target) + sum(a == b for a, b in zip(target, target[1:]))
            if len(features) < minimum:
                raise ValueError(f"CTC alignment impossible: {record['id']}. Increase extraction FPS.")
            self.items.append((torch.from_numpy((features - mean) / scale), torch.tensor(target), record))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        features, target, record = self.items[index]
        if self.augment:
            features = features + torch.randn_like(features) * .02
        return features, target, record


def collate(items):
    features, targets, records = zip(*items)
    return (pad_sequence(features, batch_first=True), torch.tensor([len(x) for x in features]),
            torch.cat(targets), torch.tensor([len(x) for x in targets]), records)


class ContinuousRecognizer(nn.Module):
    def __init__(self, channels, classes, hidden=192):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(channels * 2, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(.25))
        self.temporal = nn.GRU(hidden, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=.3)
        self.head = nn.Linear(hidden * 2, classes)

    def forward(self, features, lengths):
        velocity = torch.zeros_like(features)
        velocity[:, 1:] = features[:, 1:] - features[:, :-1]
        projected = self.project(torch.cat((features, velocity), dim=-1))
        packed = pack_padded_sequence(projected, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.temporal(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=features.shape[1])
        return self.head(output).log_softmax(-1).transpose(0, 1)


def edit_distance(reference, predicted):
    previous = list(range(len(predicted) + 1))
    for row, word in enumerate(reference, 1):
        current = [row]
        for column, other in enumerate(predicted, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (word != other)))
        previous = current
    return previous[-1]


def decode(indices, tokens):
    words, previous = [], 0
    for value in indices:
        if value != previous and value != 0:
            words.append(tokens[value])
        previous = value
    return words


def evaluate(model, loader, device, tokens):
    model.eval()
    errors = total = exact = 0
    details = []
    with torch.inference_mode():
        for features, lengths, _, _, records in loader:
            predictions = model(features.to(device), lengths).argmax(-1).T.cpu().tolist()
            for indices, length, record in zip(predictions, lengths.tolist(), records):
                predicted = decode(indices[:length], tokens)
                reference = record["tokens"]
                distance = edit_distance(reference, predicted)
                errors += distance
                total += len(reference)
                exact += int(reference == predicted)
                details.append({"id": record["id"], "reference": reference, "predicted": predicted,
                                "edit_distance": distance})
    return {"wer": errors / max(total, 1), "edits": errors, "reference_tokens": total,
            "sentence_accuracy": exact / max(len(details), 1), "videos": len(details), "details": details}


def train(records, cache, output, epochs, batch_size):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = [record for record in records if record["split"] == "train"]
    tokens = ["<blank>", "<unk>"] + sorted({token for record in training for token in record["tokens"]})
    vocabulary = {token: index for index, token in enumerate(tokens)}
    sums, squares, count = np.zeros(309), np.zeros(309), 0
    for record in training:
        with np.load(cache / (record["id"] + ".npz")) as data:
            values = data["features"].astype(np.float64)
        sums += values.sum(0)
        squares += (values ** 2).sum(0)
        count += len(values)
    mean = (sums / count).astype(np.float32)
    scale = np.sqrt(np.maximum(squares / count - mean.astype(np.float64) ** 2, .01)).astype(np.float32)
    loaders = {}
    for split in ("train", "dev", "test"):
        dataset = Sequences([record for record in records if record["split"] == split], cache,
                            vocabulary, mean, scale, augment=split == "train")
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=split == "train", collate_fn=collate)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"feature_version": FEATURE_VERSION, "tokens": tokens, "channels": 309, "hidden": 192,
                "architecture": "shoulder_local_hand_xy_velocity_bigru192x2_ctc",
                "cache_configuration": json.loads((cache / "configuration.json").read_text(encoding="utf-8")),
                "device": str(device), "seed": 42, "deployed": False,
                "splits": {split: len(loader.dataset) for split, loader in loaders.items()},
                "oov_tokens": {split: sorted({token for record in records if record["split"] == split
                                              for token in record["tokens"] if token not in vocabulary})
                               for split in ("dev", "test")}}
    write_json(output / "configuration.json", metadata)
    model = ContinuousRecognizer(309, len(tokens)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    criterion = nn.CTCLoss(blank=0, zero_infinity=False)
    history, best = [], float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        started = time.monotonic()
        for features, lengths, targets, target_lengths, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            probabilities = model(features.to(device), lengths)
            loss = criterion(probabilities, targets.to(device), lengths, target_lengths)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite CTC loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate(model, loaders["dev"], device, tokens)
        summary = {"epoch": epoch, "train_ctc_loss": float(np.mean(losses)),
                   "dev_wer": validation["wer"], "dev_sentence_accuracy": validation["sentence_accuracy"],
                   "seconds": round(time.monotonic() - started, 1)}
        history.append(summary)
        write_json(output / "history.json", history)
        print(json.dumps(summary), flush=True)
        checkpoint = {"state_dict": model.state_dict(), "mean": torch.from_numpy(mean),
                      "scale": torch.from_numpy(scale), "metadata": metadata, "epoch": epoch,
                      "optimizer": optimizer.state_dict()}
        torch.save(checkpoint, output / "last.pt")
        if validation["wer"] < best:
            best = validation["wer"]
            torch.save(checkpoint, output / "best.pt")
            write_json(output / "dev_predictions.json", validation)
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test = evaluate(model, loaders["test"], device, tokens)
    write_json(output / "test_predictions.json", test)
    report = {"best_epoch": checkpoint["epoch"], "dev_wer": best,
              "test": {key: value for key, value in test.items() if key != "details"},
              "history": history, "configuration": metadata,
              "limitations": ["Skeleton-only continuous gloss baseline; no mouth or RGB features.",
                              "Original provided splits are preserved, not asserted signer-independent.",
                              "Greedy CTC decoding; no language model. WER can exceed 100%.",
                              "Experimental checkpoint, not installed in live demonstration matching."]}
    write_json(output / "report.json", report)
    print(json.dumps({"output": str(output), "best_epoch": checkpoint["epoch"],
                      "dev_wer": best, "test_wer": test["wer"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "cecsl-skeleton-8fps")
    parser.add_argument("--output", type=Path, default=ROOT / "training" / "runs" / ("cecsl-ctc-" + time.strftime("%Y%m%d-%H%M%S")))
    parser.add_argument("--fps", type=float, default=8)
    parser.add_argument("--side", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stage", choices=("all", "extract", "train"), default="all")
    parser.add_argument("--limit-per-split", type=int, help="Smoke test only. Keep its cache separate.")
    args = parser.parse_args()
    if args.fps <= 0 or args.workers < 1 or args.epochs < 1 or args.batch_size < 1:
        parser.error("FPS, workers, epochs and batch size must be positive")
    records = read_manifest(args.dataset.resolve())
    if args.limit_per_split:
        records = [record for split in ("train", "dev", "test")
                   for record in [item for item in records if item["split"] == split][:args.limit_per_split]]
    print(json.dumps({"videos": len(records), "cuda": torch.cuda.is_available(),
                      "cache": str(args.cache), "output": str(args.output)}), flush=True)
    if args.stage in ("all", "extract"):
        extract(records, args.cache, args.models, args.fps, args.side, args.workers)
    if args.stage in ("all", "train"):
        train(records, args.cache, args.output, args.epochs, args.batch_size)


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
