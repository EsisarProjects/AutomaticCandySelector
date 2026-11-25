"""HSV feature extraction for an optional V2 per-type color classifier.

A saturation mask may fail on white candies or smurf hats obscured by bodies.
Do not treat these features as a validated V2 color solution without testing.
"""
from __future__ import annotations
import math
import cv2
import numpy as np

def resize_for_analysis(rgb: np.ndarray, max_side: int = 800) -> np.ndarray:
    """Réduit la résolution pour éviter de traiter les photos 3000×4000."""
    if max_side < 32:
        raise ValueError("max_side doit être >= 32")
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return rgb.copy()
    return cv2.resize(
        rgb,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def segment_colored_candy(
    rgb: np.ndarray,
    min_saturation: int = 20,
    min_value: int = 15,
    min_area: int = 30,
) -> np.ndarray:
    """Masque (0/255) d'un objet *coloré* sur fond blanc/gris.

    Limite connue : un bonbon blanc sur fond blanc n'est généralement pas
    détectable par cette méthode. Aucune classe réelle n'est utilisée.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb doit être un tableau uint8 H×W×3")
    if not (0 <= min_saturation <= 255 and 0 <= min_value <= 255):
        raise ValueError("Seuils HSV hors de [0,255]")

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    candidate = cv2.inRange(
        hsv,
        (0, min_saturation, min_value),
        (179, 255, 255),
    )
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel_open)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(
        candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    h, w = rgb.shape[:2]
    best = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        cx, cy = x + bw / 2, y + bh / 2
        # Biais faible vers le centre : les bonbons sont placés au centre.
        distance = math.hypot((cx - w / 2) / (w / 2), (cy - h / 2) / (h / 2))
        score = area / (1.0 + distance)
        if score > best_score:
            best_score = score
            best = contour

    mask = np.zeros((h, w), dtype=np.uint8)
    if best is not None:
        cv2.drawContours(mask, [best], -1, 255, thickness=cv2.FILLED)
    return mask


def circular_hue_mean(h_values: np.ndarray) -> float:
    """Moyenne circulaire de la teinte OpenCV, intervalle [0,180)."""
    arr = np.asarray(h_values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    angles = arr * (2 * np.pi / 180.0)
    vector = np.mean(np.exp(1j * angles))
    if abs(vector) < 1e-8:
        return float("nan")
    return float((np.angle(vector) % (2 * np.pi)) * 180.0 / (2 * np.pi))


def compute_hsv_features(rgb: np.ndarray, mask: np.ndarray) -> dict:
    """Mesure la teinte circulaire et les médianes S/V dans le masque."""
    if mask.shape != rgb.shape[:2]:
        raise ValueError("Dimensions du masque incompatibles")
    selected = mask > 0
    fraction = float(selected.mean())
    if not np.any(selected):
        return {
            "status": "non_detecte",
            "mask_fraction": 0.0,
            "hue_mean": float("nan"),
            "s_median": float("nan"),
            "v_median": float("nan"),
        }
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    pixels = hsv[selected]
    return {
        "status": "detecte",
        "mask_fraction": fraction,
        "hue_mean": circular_hue_mean(pixels[:, 0]),
        "s_median": float(np.median(pixels[:, 1])),
        "v_median": float(np.median(pixels[:, 2])),
    }


