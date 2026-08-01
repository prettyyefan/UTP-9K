from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utp9k.data import UTP9KRegionDataset, collate_variable_regions
from utp9k.models import RoleAwareTactileHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the UTP-9K motion-semantic role head")
    parser.add_argument("--manifest", required=True, help="Training JSONL manifest")
    parser.add_argument("--root", default=None, help="Optional data root")
    parser.add_argument("--output", default="outputs/role_head.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--appearance-dim", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = UTP9KRegionDataset(args.manifest, args.root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        collate_fn=collate_variable_regions,
    )

    model = RoleAwareTactileHead(appearance_dim=args.appearance_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            flow = batch["flow"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)
            roi = batch["tactile_roi"].to(device, non_blocking=True)
            altitude = batch["altitude_m"].to(device, non_blocking=True)
            roles = batch["roles"].to(device, non_blocking=True)
            appearance = batch.get("appearance_features")
            if appearance is not None:
                appearance = appearance.to(device, non_blocking=True)

            output = model(flow, masks, roi, altitude, appearance)
            loss = criterion(output.role_logits.reshape(-1, 3), roles.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.detach())
            progress.set_postfix(loss=f"{float(loss):.4f}")
        scheduler.step()
        print(f"epoch={epoch + 1} mean_loss={running_loss / max(len(loader), 1):.6f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "appearance_dim": args.appearance_dim,
                "num_direction_bins": 8,
                "prompt_dim": 128,
            },
        },
        output_path,
    )
    print(f"Saved checkpoint to {output_path}")


if __name__ == "__main__":
    main()
