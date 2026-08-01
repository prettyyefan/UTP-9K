from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from utp9k.adapters import SAM2AutomaticMaskProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute tactile-ROI-filtered SAM2 masks")
    parser.add_argument("--manifest", required=True, help="JSONL with image/roi/output fields")
    parser.add_argument("--root", default=".")
    parser.add_argument("--sam2-config", required=True)
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-candidates", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    records = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    provider = SAM2AutomaticMaskProvider(
        model_cfg=args.sam2_config,
        checkpoint=args.sam2_checkpoint,
        device=args.device,
        max_candidates=args.max_candidates,
    )

    for record in tqdm(records, desc="SAM2 candidates"):
        bgr = cv2.imread(str(root / record["image"]), cv2.IMREAD_COLOR)
        roi = cv2.imread(str(root / record["roi"]), cv2.IMREAD_GRAYSCALE)
        if bgr is None or roi is None:
            raise FileNotFoundError(f"Missing image or ROI in record: {record}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
        candidates = provider.generate(rgb, roi)
        output = root / record["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, masks=candidates.masks, scores=candidates.scores)


if __name__ == "__main__":
    main()
