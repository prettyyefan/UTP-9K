from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from utp9k.metrics import role_f1_scores, region_confusion_false_positive_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved region-role predictions")
    parser.add_argument("--predictions", required=True, help="Torch file with target and prediction tensors")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = torch.load(args.predictions, map_location="cpu", weights_only=True)
    target = data["target_role"].long()
    prediction = data["predicted_role"].long()

    f1 = role_f1_scores(target, prediction)
    metrics = {
        "macro_f1": float(f1["macro_f1"] * 100.0),
        "f1_stationary": float(f1["class_f1"][0] * 100.0),
        "f1_moving": float(f1["class_f1"][1] * 100.0),
        "f1_artifact_shadow": float(f1["class_f1"][2] * 100.0),
        "moving_as_obstacle_fp": float(
            region_confusion_false_positive_rate(target, prediction, source_role=1) * 100.0
        ),
        "artifact_as_obstacle_fp": float(
            region_confusion_false_positive_rate(target, prediction, source_role=2) * 100.0
        ),
    }
    print(json.dumps(metrics, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
