from __future__ import annotations

import torch
from torch import Tensor


def _safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    return torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))


def role_f1_scores(target: Tensor, prediction: Tensor, num_classes: int = 3) -> dict[str, Tensor]:
    """Return class-wise and macro F1 without requiring scikit-learn."""
    target = target.reshape(-1).long()
    prediction = prediction.reshape(-1).long()
    valid = target >= 0
    target = target[valid]
    prediction = prediction[valid]

    scores = []
    for class_index in range(num_classes):
        true_positive = ((target == class_index) & (prediction == class_index)).sum().float()
        false_positive = ((target != class_index) & (prediction == class_index)).sum().float()
        false_negative = ((target == class_index) & (prediction != class_index)).sum().float()
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        scores.append(f1)

    class_f1 = torch.stack(scores)
    return {"class_f1": class_f1, "macro_f1": class_f1.mean()}


def region_confusion_false_positive_rate(
    target_role: Tensor,
    predicted_role: Tensor,
    source_role: int,
    stationary_role: int = 0,
) -> Tensor:
    """Rate at which a source role is incorrectly predicted as stationary."""
    target_role = target_role.reshape(-1).long()
    predicted_role = predicted_role.reshape(-1).long()
    source = target_role == source_role
    denominator = source.sum().float()
    numerator = (source & (predicted_role == stationary_role)).sum().float()
    return _safe_divide(numerator, denominator)
