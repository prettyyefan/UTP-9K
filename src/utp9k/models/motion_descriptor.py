from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RegionMotionDescriptor(nn.Module):
    """Pool dense optical flow into region-level motion evidence.

    For every candidate mask, the module computes the mean and variance of
    optical-flow magnitude and the entropy of quantized flow directions:

        m_i = [mean_magnitude, magnitude_variance, directional_entropy].

    Args:
        num_direction_bins: Number of angular bins used for direction entropy.
        eps: Numerical stabilizer for empty/very small masks.
        normalize_magnitude: Divide flow magnitude by the image diagonal. This
            makes descriptors less sensitive to input resolution.
        normalize_entropy: Divide entropy by log(num_direction_bins), yielding
            values in approximately [0, 1]. Set False to match the raw paper
            expression exactly.
    """

    def __init__(
        self,
        num_direction_bins: int = 8,
        eps: float = 1e-6,
        normalize_magnitude: bool = True,
        normalize_entropy: bool = False,
    ) -> None:
        super().__init__()
        if num_direction_bins < 2:
            raise ValueError("num_direction_bins must be at least 2")
        self.num_direction_bins = int(num_direction_bins)
        self.eps = float(eps)
        self.normalize_magnitude = bool(normalize_magnitude)
        self.normalize_entropy = bool(normalize_entropy)

    def forward(self, flow: Tensor, masks: Tensor) -> Tensor:
        """Compute descriptors.

        Args:
            flow: Optical flow of shape ``[B, 2, H, W]``.
            masks: Candidate masks of shape ``[B, N, H, W]``. Masks may be
                binary or soft; values are clamped to [0, 1].

        Returns:
            Tensor of shape ``[B, N, 3]``.
        """
        if flow.ndim != 4 or flow.shape[1] != 2:
            raise ValueError(f"flow must have shape [B,2,H,W], got {tuple(flow.shape)}")
        if masks.ndim != 4:
            raise ValueError(f"masks must have shape [B,N,H,W], got {tuple(masks.shape)}")
        if flow.shape[0] != masks.shape[0] or flow.shape[-2:] != masks.shape[-2:]:
            raise ValueError("flow and masks must share batch and spatial dimensions")

        _, _, height, width = flow.shape
        weights = masks.to(dtype=flow.dtype).clamp_(0.0, 1.0)
        area = weights.sum(dim=(-2, -1)).clamp_min(self.eps)  # [B, N]

        magnitude = torch.linalg.vector_norm(flow, ord=2, dim=1)  # [B,H,W]
        if self.normalize_magnitude:
            magnitude = magnitude / math.sqrt(float(height * height + width * width))

        mean = torch.einsum("bnhw,bhw->bn", weights, magnitude) / area
        centered = magnitude[:, None] - mean[:, :, None, None]
        variance = (weights * centered.square()).sum(dim=(-2, -1)) / area

        angle = torch.atan2(flow[:, 1], flow[:, 0])  # [-pi, pi]
        scaled = (angle + math.pi) / (2.0 * math.pi)
        bin_index = torch.floor(scaled * self.num_direction_bins).long()
        bin_index = bin_index.clamp_(0, self.num_direction_bins - 1)
        direction_one_hot = F.one_hot(
            bin_index, num_classes=self.num_direction_bins
        ).to(dtype=flow.dtype)
        direction_one_hot = direction_one_hot.permute(0, 3, 1, 2)  # [B,K,H,W]

        histogram = torch.einsum("bnhw,bkhw->bnk", weights, direction_one_hot)
        probability = histogram / area.unsqueeze(-1)
        entropy = -(probability * (probability + self.eps).log()).sum(dim=-1)
        if self.normalize_entropy:
            entropy = entropy / math.log(self.num_direction_bins)

        descriptor = torch.stack((mean, variance, entropy), dim=-1)

        # Explicitly zero descriptors for empty masks instead of returning an
        # arbitrary value produced by the stabilizing denominator.
        nonempty = masks.sum(dim=(-2, -1)) > 0
        return descriptor * nonempty.unsqueeze(-1).to(descriptor.dtype)
