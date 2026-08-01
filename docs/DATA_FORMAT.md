# UTP-9K data interface

The public dataset remains hosted on Hugging Face. This code package uses a
manifest-driven interface so the repository does not assume a private storage
layout.

## Region-head training manifest

Each line of `train.jsonl` is a JSON object:

```json
{
  "flow": "precomputed/flow/clip_001_000123.npy",
  "masks": "precomputed/masks/clip_001_000123.npz",
  "roles": ["stationary", "moving", "artifact"],
  "tactile_roi": "annotations/roi/clip_001_000123.npy",
  "altitude_m": 30,
  "appearance_features": "precomputed/features/clip_001_000123.npy"
}
```

Shapes:

- `flow`: `[2, H, W]`, previous-to-current optical flow.
- `masks`: `[N, H, W]`, SAM2 candidate masks.
- `tactile_roi`: `[H, W]`, tactile-path region of interest.
- `appearance_features`: optional `[N, D]` region appearance features.
- `roles`: one label per candidate: `stationary`, `moving`, or `artifact`.

The dataloader pads variable candidate counts and returns `valid_regions`.
Labels for padded candidates use the ignore index `-100`.
