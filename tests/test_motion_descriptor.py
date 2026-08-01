import torch

from utp9k.models.motion_descriptor import RegionMotionDescriptor


def test_constant_horizontal_flow_has_zero_variance_and_entropy() -> None:
    flow = torch.zeros(1, 2, 8, 8)
    flow[:, 0] = 2.0
    masks = torch.ones(1, 1, 8, 8)
    descriptor = RegionMotionDescriptor(normalize_magnitude=False)(flow, masks)
    assert torch.allclose(descriptor[..., 0], torch.tensor([[2.0]]), atol=1e-5)
    assert torch.allclose(descriptor[..., 1], torch.zeros(1, 1), atol=1e-5)
    assert torch.allclose(descriptor[..., 2], torch.zeros(1, 1), atol=1e-5)


def test_empty_mask_returns_zero_descriptor() -> None:
    flow = torch.randn(1, 2, 8, 8)
    masks = torch.zeros(1, 1, 8, 8)
    descriptor = RegionMotionDescriptor()(flow, masks)
    assert torch.equal(descriptor, torch.zeros_like(descriptor))
