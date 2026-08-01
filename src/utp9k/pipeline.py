from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .adapters.raft import TorchvisionRAFT
from .adapters.sam2 import SAM2AutomaticMaskProvider
from .models.role_aware_head import RoleAwareTactileHead


@dataclass
class PipelinePrediction:
    masks: np.ndarray
    role_probabilities: np.ndarray
    binary_probabilities: np.ndarray


class RoleAwareTactilePipeline:
    """End-to-end inference wrapper around frozen SAM2/RAFT adapters."""

    def __init__(
        self,
        raft: TorchvisionRAFT,
        sam2: SAM2AutomaticMaskProvider,
        head: RoleAwareTactileHead,
        device: str | torch.device = "cuda",
    ) -> None:
        self.raft = raft
        self.sam2 = sam2
        self.head = head.eval().to(device)
        self.device = torch.device(device)

    @staticmethod
    def _image_to_tensor(image_rgb: np.ndarray) -> Tensor:
        return torch.from_numpy(image_rgb).permute(2, 0, 1).float().div(255.0)

    def predict(
        self,
        previous_rgb: np.ndarray,
        current_rgb: np.ndarray,
        tactile_roi: np.ndarray,
        altitude_m: float,
    ) -> PipelinePrediction:
        candidates = self.sam2.generate(current_rgb, tactile_roi)
        if candidates.masks.shape[0] == 0:
            return PipelinePrediction(
                masks=candidates.masks,
                role_probabilities=np.zeros((0, 3), dtype=np.float32),
                binary_probabilities=np.zeros((0, 2), dtype=np.float32),
            )

        previous = self._image_to_tensor(previous_rgb)[None]
        current = self._image_to_tensor(current_rgb)[None]
        flow = self.raft(previous, current).to(self.device)
        masks = torch.from_numpy(candidates.masks.astype(np.float32))[None].to(self.device)
        roi = torch.from_numpy(tactile_roi.astype(np.float32))[None, None].to(self.device)
        altitude = torch.tensor([altitude_m], device=self.device)

        with torch.inference_mode():
            output = self.head(flow, masks, roi, altitude)
            role_probabilities = output.role_probabilities[0].cpu().numpy()
            binary_probabilities = output.binary_logits[0].softmax(dim=-1).cpu().numpy()
        return PipelinePrediction(candidates.masks, role_probabilities, binary_probabilities)
