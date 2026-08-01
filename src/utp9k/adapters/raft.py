from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TorchvisionRAFT(nn.Module):
    """Frozen RAFT optical-flow adapter backed by torchvision.

    The adapter pads inputs to multiples of eight, applies the preprocessing
    bundled with the selected pretrained weights, and returns the final flow
    estimate at the original spatial resolution.
    """

    def __init__(self, device: str | torch.device = "cuda", use_large: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models.optical_flow import (
                Raft_Large_Weights,
                Raft_Small_Weights,
                raft_large,
                raft_small,
            )
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Torchvision optical-flow models are unavailable. Install a "
                "torchvision build matching your PyTorch version."
            ) from exc

        self.device = torch.device(device)
        if use_large:
            self.weights = Raft_Large_Weights.DEFAULT
            self.model = raft_large(weights=self.weights, progress=True)
        else:
            self.weights = Raft_Small_Weights.DEFAULT
            self.model = raft_small(weights=self.weights, progress=True)
        self.transforms = self.weights.transforms()
        self.model.eval().requires_grad_(False).to(self.device)

    @staticmethod
    def _pad_to_multiple_of_eight(image: Tensor) -> tuple[Tensor, tuple[int, int]]:
        height, width = image.shape[-2:]
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        return F.pad(image, (0, pad_w, 0, pad_h), mode="replicate"), (pad_h, pad_w)

    def forward(self, image1: Tensor, image2: Tensor) -> Tensor:
        if image1.shape != image2.shape or image1.ndim != 4 or image1.shape[1] != 3:
            raise ValueError("image1 and image2 must both have shape [B,3,H,W]")

        original_h, original_w = image1.shape[-2:]
        image1, _ = self._pad_to_multiple_of_eight(image1)
        image2, _ = self._pad_to_multiple_of_eight(image2)
        image1, image2 = self.transforms(image1, image2)
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)

        autocast_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            predictions = self.model(image1, image2)
        return predictions[-1][..., :original_h, :original_w].float()
