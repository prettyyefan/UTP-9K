import torch

from utp9k.metrics import role_f1_scores, region_confusion_false_positive_rate


def test_perfect_role_f1() -> None:
    target = torch.tensor([0, 1, 2, 0, 1, 2])
    result = role_f1_scores(target, target)
    assert torch.allclose(result["class_f1"], torch.ones(3))
    assert torch.allclose(result["macro_f1"], torch.tensor(1.0))


def test_region_confusion_fp() -> None:
    target = torch.tensor([1, 1, 1, 2])
    prediction = torch.tensor([0, 1, 0, 2])
    rate = region_confusion_false_positive_rate(target, prediction, source_role=1)
    assert torch.allclose(rate, torch.tensor(2.0 / 3.0))
