"""Run a trained CE-CSL CTC checkpoint on one complete sign-language video."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train_cecsl import ContinuousRecognizer, FEATURE_VERSION, decode, extract_one, init_extractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/cecsl-predictions"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if metadata["feature_version"] != FEATURE_VERSION:
        raise ValueError("Checkpoint uses a different feature representation")
    configuration = metadata["cache_configuration"]
    for filename, digest in configuration["models"].items():
        if hashlib.sha256((args.models / filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Landmark model differs from training: {filename}")
    with args.video.open("rb") as stream:
        video_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    config_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:16]
    cache = args.cache / config_hash
    cache.mkdir(parents=True, exist_ok=True)
    init_extractor(args.models, configuration["fps"], configuration["side"])
    ident, frames, coverage = extract_one(({"id": video_hash, "path": str(args.video)}, str(cache)))
    with np.load(cache / (ident + ".npz")) as data:
        features = torch.from_numpy(data["features"].astype(np.float32))
    features = (features - checkpoint["mean"]) / checkpoint["scale"]
    model = ContinuousRecognizer(metadata["channels"], len(metadata["tokens"]), metadata["hidden"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.inference_mode():
        indices = model(features[None].to(device), torch.tensor([len(features)])).argmax(-1)[:, 0].tolist()
    print(json.dumps({"gloss": decode(indices, metadata["tokens"]), "frames": frames,
                      "hand_coverage": coverage, "checkpoint_epoch": checkpoint["epoch"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
