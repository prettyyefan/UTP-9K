from __future__ import annotations

import cv2
import numpy as np

ROLE_NAMES = ("stationary", "moving", "artifact/shadow")
ROLE_COLORS_BGR = (
    (68, 68, 220),    # red-ish
    (171, 132, 36),   # blue-ish
    (172, 111, 122),  # purple-ish
)


def overlay_role_masks(
    image_bgr: np.ndarray,
    masks: np.ndarray,
    role_probabilities: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay candidate masks and role labels on a BGR image."""
    output = image_bgr.copy()
    for mask, probabilities in zip(masks, role_probabilities):
        role = int(np.argmax(probabilities))
        color = ROLE_COLORS_BGR[role]
        mask_bool = mask.astype(bool)
        colored = np.zeros_like(output)
        colored[:] = color
        output[mask_bool] = cv2.addWeighted(
            output[mask_bool], 1.0 - alpha, colored[mask_bool], alpha, 0.0
        )

        contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, color, 2)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, _, _ = cv2.boundingRect(largest)
            label = f"{ROLE_NAMES[role]} {probabilities[role]:.2f}"
            cv2.putText(output, label, (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return output
