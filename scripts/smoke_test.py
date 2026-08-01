from __future__ import annotations

import torch

from utp9k.models.role_aware_head import RoleAwareTactileHead


def main() -> None:
    torch.manual_seed(7)
    batch, regions, height, width = 2, 5, 96, 128
    flow = torch.randn(batch, 2, height, width)
    masks = (torch.rand(batch, regions, height, width) > 0.82).float()
    tactile_roi = (torch.rand(batch, 1, height, width) > 0.35).float()
    altitude = torch.tensor([30.0, 60.0])
    appearance = torch.randn(batch, regions, 256)

    model = RoleAwareTactileHead()
    output = model(flow, masks, tactile_roi, altitude, appearance)

    assert output.role_logits.shape == (batch, regions, 3)
    assert output.binary_logits.shape == (batch, regions, 2)
    assert output.fused_prompt.shape == (batch, regions, 128)
    print("Smoke test passed")
    print("role_logits:", tuple(output.role_logits.shape))
    print("binary_logits:", tuple(output.binary_logits.shape))
    print("fused_prompt:", tuple(output.fused_prompt.shape))


if __name__ == "__main__":
    main()
