from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from utp9k.adapters import TorchvisionRAFT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute RAFT flow for frame pairs")
    parser.add_argument("--manifest", required=True, help="JSONL with previous/current/output fields")
    parser.add_argument("--root", default=".")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--small", action="store_true", help="Use torchvision RAFT-small")
    return parser.parse_args()


def image_tensor(path: Path) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    records = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    raft = TorchvisionRAFT(device=args.device, use_large=not args.small)

    for record in tqdm(records, desc="RAFT flow"):
        previous = image_tensor(root / record["previous"])[None]
        current = image_tensor(root / record["current"])[None]
        flow = raft(previous, current)[0].cpu().numpy().astype(np.float32)
        output = root / record["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, flow)


if __name__ == "__main__":
    main()
