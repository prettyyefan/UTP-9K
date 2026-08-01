from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .role_prompt import MotionSemanticPromptModel, PromptOutput


@dataclass
class RoleAwareOutput:
    role_logits: Tensor
    role_probabilities: Tensor
    binary_logits: Tensor
    fused_prompt: Tensor
    prompt_details: PromptOutput


class RoleAwareTactileHead(nn.Module):
    """Reference head for Task A and Task B.

    This module implements the paper-specific trainable components. SAM2 and
    RAFT are intentionally kept behind adapters so their encoders can remain
    frozen. In a full SAM2 integration, ``fused_prompt`` should be projected
    into the decoder prompt pathway. The optional appearance features make the
    standalone head directly trainable for smoke tests and ablations.
    """

    def __init__(
        self,
        prompt_dim: int = 128,
        hidden_dim: int = 128,
        appearance_dim: int = 256,
        num_roles: int = 3,
        num_direction_bins: int = 8,
    ) -> None:
        super().__init__()
        self.num_roles = num_roles
        self.prompt_model = MotionSemanticPromptModel(
            prompt_dim=prompt_dim,
            hidden_dim=hidden_dim,
            num_roles=num_roles,
            num_direction_bins=num_direction_bins,
        )
        self.appearance_projection = nn.Sequential(
            nn.Linear(appearance_dim, prompt_dim),
            nn.GELU(),
            nn.LayerNorm(prompt_dim),
        )
        self.role_head = nn.Sequential(
            nn.Linear(prompt_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_roles),
        )

    def forward(
        self,
        flow: Tensor,
        masks: Tensor,
        tactile_roi: Tensor,
        altitude_m: Tensor,
        appearance_features: Tensor | None = None,
    ) -> RoleAwareOutput:
        prompt = self.prompt_model(flow, masks, tactile_roi, altitude_m)
        batch_size, num_regions, prompt_dim = prompt.fused_prompt.shape

        if appearance_features is None:
            appearance_token = torch.zeros(
                batch_size,
                num_regions,
                prompt_dim,
                device=prompt.fused_prompt.device,
                dtype=prompt.fused_prompt.dtype,
            )
        else:
            if appearance_features.shape[:2] != (batch_size, num_regions):
                raise ValueError("appearance_features must have shape [B,N,D]")
            appearance_token = self.appearance_projection(appearance_features)

        residual_logits = self.role_head(
            torch.cat((appearance_token, prompt.fused_prompt), dim=-1)
        )
        role_logits = prompt.role_logits + residual_logits
        role_probabilities = torch.softmax(role_logits, dim=-1)

        # Task A treats stationary blockage (class 0) as positive and all
        # remaining roles as negative.
        negative_logit = torch.logsumexp(role_logits[..., 1:], dim=-1)
        positive_logit = role_logits[..., 0]
        binary_logits = torch.stack((negative_logit, positive_logit), dim=-1)

        return RoleAwareOutput(
            role_logits=role_logits,
            role_probabilities=role_probabilities,
            binary_logits=binary_logits,
            fused_prompt=prompt.fused_prompt,
            prompt_details=prompt,
        )
