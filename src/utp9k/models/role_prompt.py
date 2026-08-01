from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .motion_descriptor import RegionMotionDescriptor


@dataclass
class PromptOutput:
    descriptor: Tensor
    role_logits: Tensor
    role_probabilities: Tensor
    motion_token: Tensor
    role_token: Tensor
    spatial_token: Tensor
    altitude_token: Tensor
    fused_prompt: Tensor


class SemanticRolePromptBank(nn.Module):
    """Learnable embeddings for stationary, moving, and artifact roles."""

    def __init__(self, prompt_dim: int, num_roles: int = 3) -> None:
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty(num_roles, prompt_dim))
        nn.init.normal_(self.embeddings, mean=0.0, std=0.02)

    def forward(self, role_probabilities: Tensor) -> Tensor:
        return role_probabilities @ self.embeddings


class RegionSpatialEncoder(nn.Module):
    """Encode candidate geometry and overlap with the tactile-path ROI."""

    def __init__(self, prompt_dim: int, hidden_dim: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prompt_dim),
        )

    def forward(self, masks: Tensor, tactile_roi: Tensor) -> Tensor:
        if masks.ndim != 4 or tactile_roi.ndim != 4 or tactile_roi.shape[1] != 1:
            raise ValueError("masks must be [B,N,H,W] and tactile_roi must be [B,1,H,W]")
        if masks.shape[0] != tactile_roi.shape[0] or masks.shape[-2:] != tactile_roi.shape[-2:]:
            raise ValueError("masks and tactile_roi must share batch and spatial dimensions")

        bsz, _, height, width = masks.shape
        masks_f = masks.float().clamp(0.0, 1.0)
        roi = tactile_roi.float().clamp(0.0, 1.0)
        area = masks_f.sum(dim=(-2, -1)).clamp_min(self.eps)

        ys = torch.linspace(0.0, 1.0, height, device=masks.device, dtype=masks_f.dtype)
        xs = torch.linspace(0.0, 1.0, width, device=masks.device, dtype=masks_f.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        center_x = torch.einsum("bnhw,hw->bn", masks_f, grid_x) / area
        center_y = torch.einsum("bnhw,hw->bn", masks_f, grid_y) / area
        area_fraction = area / float(height * width)
        roi_overlap = (masks_f * roi).sum(dim=(-2, -1)) / area

        features = torch.stack((area_fraction, center_x, center_y, roi_overlap), dim=-1)
        return self.network(features)


class AltitudeEncoder(nn.Module):
    def __init__(self, prompt_dim: int, hidden_dim: int = 32, scale_m: float = 60.0) -> None:
        super().__init__()
        self.scale_m = float(scale_m)
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prompt_dim),
        )

    def forward(self, altitude_m: Tensor, num_regions: int) -> Tensor:
        if altitude_m.ndim == 1:
            altitude_m = altitude_m[:, None]
        if altitude_m.ndim != 2 or altitude_m.shape[1] != 1:
            raise ValueError("altitude_m must have shape [B] or [B,1]")
        token = self.network(altitude_m.float() / self.scale_m)
        return token[:, None, :].expand(-1, num_regions, -1)


class MotionSemanticPromptModel(nn.Module):
    """Convert region motion into a soft semantic prompt.

    The implementation follows the paper's division of labor:
    region-level optical-flow statistics produce a soft role distribution,
    which weights a learnable semantic role bank and is fused with motion,
    tactile-path spatial, and altitude cues.
    """

    def __init__(
        self,
        prompt_dim: int = 128,
        hidden_dim: int = 128,
        num_roles: int = 3,
        num_direction_bins: int = 8,
    ) -> None:
        super().__init__()
        self.num_roles = num_roles
        self.motion_descriptor = RegionMotionDescriptor(
            num_direction_bins=num_direction_bins,
            normalize_magnitude=True,
            normalize_entropy=False,
        )
        self.role_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_roles),
        )
        self.role_projection = nn.Linear(num_roles, prompt_dim)
        self.stat_projection = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prompt_dim),
        )
        self.role_bank = SemanticRolePromptBank(prompt_dim, num_roles)
        self.spatial_encoder = RegionSpatialEncoder(prompt_dim, hidden_dim // 2)
        self.altitude_encoder = AltitudeEncoder(prompt_dim)
        self.fusion = nn.Sequential(
            nn.Linear(prompt_dim * 4, prompt_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(prompt_dim * 2, prompt_dim),
            nn.LayerNorm(prompt_dim),
        )

    def forward(
        self,
        flow: Tensor,
        masks: Tensor,
        tactile_roi: Tensor,
        altitude_m: Tensor,
    ) -> PromptOutput:
        descriptor = self.motion_descriptor(flow, masks)
        role_logits = self.role_mlp(descriptor)
        role_probabilities = F.softmax(role_logits, dim=-1)

        motion_token = self.role_projection(role_probabilities) + self.stat_projection(descriptor)
        role_token = self.role_bank(role_probabilities)
        spatial_token = self.spatial_encoder(masks, tactile_roi)
        altitude_token = self.altitude_encoder(altitude_m, masks.shape[1])

        fused_prompt = self.fusion(
            torch.cat((motion_token, role_token, spatial_token, altitude_token), dim=-1)
        )

        return PromptOutput(
            descriptor=descriptor,
            role_logits=role_logits,
            role_probabilities=role_probabilities,
            motion_token=motion_token,
            role_token=role_token,
            spatial_token=spatial_token,
            altitude_token=altitude_token,
            fused_prompt=fused_prompt,
        )
