from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


ROLE_TO_INDEX = {
    "stationary": 0,
    "moving": 1,
    "artifact": 2,
    "shadow": 2,
    "artifact/shadow": 2,
}


class UTP9KRegionDataset(Dataset[dict[str, Tensor]]):
    """Manifest-driven region dataset for the trainable motion-semantic head.

    Each JSONL row should contain:

    ``flow``: path to a ``[2,H,W]`` NumPy file;
    ``masks``: path to an ``[N,H,W]`` NumPy/NPZ file;
    ``roles``: list of role names or integer labels;
    ``tactile_roi``: path to a ``[H,W]`` NumPy file;
    ``altitude_m``: numeric flight altitude;
    optional ``appearance_features``: path to ``[N,D]`` features.
    """

    def __init__(self, manifest: str | Path, root: str | Path | None = None) -> None:
        self.manifest = Path(manifest)
        self.root = Path(root) if root is not None else self.manifest.parent
        if not self.manifest.exists():
            raise FileNotFoundError(self.manifest)

        self.records: list[dict[str, Any]] = []
        with self.manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self.records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {self.manifest}:{line_number}") from exc
        if not self.records:
            raise ValueError(f"No records found in {self.manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    @staticmethod
    def _load_array(path: Path, preferred_key: str | None = None) -> np.ndarray:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            key = preferred_key if preferred_key in loaded.files else loaded.files[0]
            array = loaded[key]
            loaded.close()
            return array
        return loaded

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        record = self.records[index]
        flow = self._load_array(self._resolve(record["flow"]), "flow").astype(np.float32)
        masks = self._load_array(self._resolve(record["masks"]), "masks").astype(np.float32)
        tactile_roi = self._load_array(
            self._resolve(record["tactile_roi"]), "tactile_roi"
        ).astype(np.float32)

        roles_raw = record["roles"]
        roles = [ROLE_TO_INDEX.get(str(value).lower(), int(value)) if not isinstance(value, str) else ROLE_TO_INDEX[value.lower()] for value in roles_raw]
        if masks.shape[0] != len(roles):
            raise ValueError(f"Record {index}: number of masks and role labels differs")

        sample: dict[str, Tensor] = {
            "flow": torch.from_numpy(flow),
            "masks": torch.from_numpy(masks),
            "tactile_roi": torch.from_numpy(tactile_roi)[None],
            "roles": torch.tensor(roles, dtype=torch.long),
            "altitude_m": torch.tensor(float(record["altitude_m"]), dtype=torch.float32),
        }

        if "appearance_features" in record:
            appearance = self._load_array(
                self._resolve(record["appearance_features"]), "features"
            ).astype(np.float32)
            sample["appearance_features"] = torch.from_numpy(appearance)
        return sample


def collate_variable_regions(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Pad variable region counts and return a ``valid_regions`` mask."""
    max_regions = max(item["masks"].shape[0] for item in batch)
    height, width = batch[0]["masks"].shape[-2:]
    appearance_dim = next(
        (item["appearance_features"].shape[-1] for item in batch if "appearance_features" in item),
        None,
    )

    flows, masks, rois, roles, altitudes, valid = [], [], [], [], [], []
    appearances = []
    for item in batch:
        n = item["masks"].shape[0]
        pad_n = max_regions - n
        flows.append(item["flow"])
        masks.append(torch.cat((item["masks"], torch.zeros(pad_n, height, width)), dim=0))
        rois.append(item["tactile_roi"])
        roles.append(torch.cat((item["roles"], torch.full((pad_n,), -100, dtype=torch.long))))
        altitudes.append(item["altitude_m"])
        valid.append(torch.cat((torch.ones(n, dtype=torch.bool), torch.zeros(pad_n, dtype=torch.bool))))

        if appearance_dim is not None:
            feature = item.get("appearance_features", torch.zeros(n, appearance_dim))
            appearances.append(torch.cat((feature, torch.zeros(pad_n, appearance_dim)), dim=0))

    output = {
        "flow": torch.stack(flows),
        "masks": torch.stack(masks),
        "tactile_roi": torch.stack(rois),
        "roles": torch.stack(roles),
        "altitude_m": torch.stack(altitudes),
        "valid_regions": torch.stack(valid),
    }
    if appearances:
        output["appearance_features"] = torch.stack(appearances)
    return output
