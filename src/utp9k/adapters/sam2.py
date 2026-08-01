from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class CandidateMasks:
    masks: np.ndarray
    scores: np.ndarray


class SAM2AutomaticMaskProvider:
    """SAM2 automatic-mask adapter with tactile-ROI filtering.

    SAM2 is an optional external dependency and is not vendored by this
    repository. Install the official Meta SAM2 package first.
    """

    def __init__(
        self,
        model_cfg: str,
        checkpoint: str | Path,
        device: str = "cuda",
        max_candidates: int = 8,
        min_area: int = 64,
        min_roi_overlap: float = 0.05,
        **generator_kwargs: Any,
    ) -> None:
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "SAM2 is not installed. Follow the official installation guide "
                "at https://github.com/facebookresearch/sam2."
            ) from exc

        self.device = torch.device(device)
        self.max_candidates = int(max_candidates)
        self.min_area = int(min_area)
        self.min_roi_overlap = float(min_roi_overlap)

        model = build_sam2(model_cfg, str(checkpoint), device=str(self.device))
        model.eval().requires_grad_(False)
        self.generator = SAM2AutomaticMaskGenerator(model, **generator_kwargs)

    def generate(self, image_rgb: np.ndarray, tactile_roi: np.ndarray | None = None) -> CandidateMasks:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must be an HxWx3 RGB array")
        proposals = self.generator.generate(image_rgb)

        filtered: list[tuple[np.ndarray, float]] = []
        roi = None if tactile_roi is None else tactile_roi.astype(bool)
        for proposal in proposals:
            mask = np.asarray(proposal["segmentation"], dtype=bool)
            area = int(mask.sum())
            if area < self.min_area:
                continue
            if roi is not None:
                overlap = float((mask & roi).sum()) / max(float(area), 1.0)
                if overlap < self.min_roi_overlap:
                    continue
            score = float(proposal.get("predicted_iou", 0.0))
            score += 0.25 * float(proposal.get("stability_score", 0.0))
            filtered.append((mask, score))

        filtered.sort(key=lambda item: item[1], reverse=True)
        selected = filtered[: self.max_candidates]
        if not selected:
            h, w = image_rgb.shape[:2]
            return CandidateMasks(
                masks=np.zeros((0, h, w), dtype=bool),
                scores=np.zeros((0,), dtype=np.float32),
            )

        return CandidateMasks(
            masks=np.stack([item[0] for item in selected]),
            scores=np.asarray([item[1] for item in selected], dtype=np.float32),
        )
