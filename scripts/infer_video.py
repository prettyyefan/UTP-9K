from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from utp9k.adapters import SAM2AutomaticMaskProvider, TorchvisionRAFT
from utp9k.models import RoleAwareTactileHead
from utp9k.pipeline import RoleAwareTactilePipeline
from utp9k.visualization import overlay_role_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Role-aware inference on a UAV video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--roi-mask", required=True, help="Static tactile-path ROI mask")
    parser.add_argument("--checkpoint", required=True, help="Motion-semantic head checkpoint")
    parser.add_argument("--sam2-config", required=True)
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--output", default="outputs/demonstration.mp4")
    parser.add_argument("--altitude", type=float, default=30.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-candidates", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = RoleAwareTactileHead()
    model.load_state_dict(checkpoint["model"], strict=True)

    raft = TorchvisionRAFT(device=device)
    sam2 = SAM2AutomaticMaskProvider(
        model_cfg=args.sam2_config,
        checkpoint=args.sam2_checkpoint,
        device=str(device),
        max_candidates=args.max_candidates,
    )
    pipeline = RoleAwareTactilePipeline(raft, sam2, model, device=device)

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    roi = cv2.imread(args.roi_mask, cv2.IMREAD_GRAYSCALE)
    if roi is None:
        raise FileNotFoundError(args.roi_mask)
    roi = cv2.resize(roi, (width, height), interpolation=cv2.INTER_NEAREST) > 127

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    ok, previous_bgr = capture.read()
    if not ok:
        raise RuntimeError("Video contains no frames")
    writer.write(previous_bgr)

    while True:
        ok, current_bgr = capture.read()
        if not ok:
            break
        previous_rgb = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2RGB)
        current_rgb = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2RGB)
        prediction = pipeline.predict(previous_rgb, current_rgb, roi, args.altitude)
        rendered = overlay_role_masks(
            current_bgr, prediction.masks, prediction.role_probabilities
        )
        writer.write(rendered)
        previous_bgr = current_bgr

    capture.release()
    writer.release()
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
