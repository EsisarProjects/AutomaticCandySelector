"""Tests pour la première analyse HSV du projet ACS."""
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.color.hsv_analysis import (
    circular_hue_mean,
    compute_hsv_features,
    resize_for_analysis,
    segment_colored_candy,
)


def test_colored_object():
    image = np.full((400, 300, 3), 255, dtype=np.uint8)
    # Rectangle vert saturé au centre, sur fond blanc.
    cv2.rectangle(image, (120, 150), (180, 250), (30, 170, 30), -1)
    mask = segment_colored_candy(image)
    assert mask.shape == (400, 300)
    assert mask[200, 150] == 255
    assert mask[20, 20] == 0
    features = compute_hsv_features(image, mask)
    assert features["status"] == "detecte"
    assert 0 < features["mask_fraction"] < 0.1
    assert 40 < features["hue_mean"] < 90
    assert features["s_median"] > 100


def test_white_limitation():
    image = np.full((400, 300, 3), 255, dtype=np.uint8)
    mask = segment_colored_candy(image)
    features = compute_hsv_features(image, mask)
    assert features["status"] == "non_detecte"
    assert np.isnan(features["hue_mean"])


def test_circular_red():
    # Le rouge est à la fois près de H=0 et de H=179 (bouclage du cercle).
    hue = circular_hue_mean(np.array([178, 179, 0, 1]))
    assert hue < 5 or hue > 175


def test_resize():
    image = np.full((4000, 3000, 3), 255, dtype=np.uint8)
    reduced = resize_for_analysis(image, max_side=800)
    assert reduced.shape == (800, 600, 3)


if __name__ == "__main__":
    test_colored_object()
    test_white_limitation()
    test_circular_red()
    test_resize()
    print("ACS — Tests HSV : OK")