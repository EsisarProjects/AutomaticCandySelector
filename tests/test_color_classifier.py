"""Tests ACS 1.6.3 : classification HSV par type (sans photos réelles)."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.color.color_classifier import (
    ABSTAIN, evaluate_color_model, extract_features_rgb, feature_vector,
    fit_color_models, predict_color, read_split,
)


VARIANTS = {
    "crocodile": ["bleu", "jaune", "orange", "rouge", "vert"],
    "ourson": ["blanc", "jaune", "orange", "rougefaible", "rougefort", "vert"],
    "fraise": ["rouge"],
    "oeuf": ["blanc"],
}

# Profils fictifs : uniquement pour tester le logiciel, pas un résultat ACS.
PROFILES = {
    ("crocodile", "bleu"): (102.0, 180.0, 175.0),
    ("crocodile", "jaune"): (25.0, 170.0, 213.0),
    ("crocodile", "orange"): (12.0, 180.0, 199.0),
    ("crocodile", "rouge"): (1.0, 177.0, 160.0),
    ("crocodile", "vert"): (49.0, 92.0, 166.0),
    ("ourson", "blanc"): (23.0, 58.0, 195.0),
    ("ourson", "jaune"): (22.0, 166.0, 195.0),
    ("ourson", "orange"): (15.0, 171.0, 188.0),
    ("ourson", "rougefaible"): (6.5, 172.0, 162.0),
    ("ourson", "rougefort"): (0.1, 183.0, 134.0),
    ("ourson", "vert"): (27.0, 124.0, 102.0),
    ("fraise", "rouge"): (8.0, 165.0, 175.0),
    ("oeuf", "blanc"): (21.0, 102.0, 198.0),
}


def make_features(repeats=5, session="S01"):
    rows = []
    for (candy_type, color), (h, s, v) in PROFILES.items():
        for i in range(repeats):
            rows.append({
                "image_path": f"example_{candy_type}_{color}_{i}.png",
                "type": candy_type, "couleur": color, "session_id": session,
                "status": "detecte", "mask_fraction": 0.03,
                "hue_mean": h, "s_median": s, "v_median": v,
            })
    return pd.DataFrame(rows)


def test_hue_is_circular():
    def row(h):
        return {"status": "detecte", "hue_mean": h,
                "s_median": 180, "v_median": 200}
    near_0 = feature_vector(row(1))
    near_179 = feature_vector(row(179))
    middle = feature_vector(row(90))
    assert np.linalg.norm(near_0 - near_179) < 0.1
    assert np.linalg.norm(near_0 - middle) > 1.5
    assert feature_vector({"status": "non_detecte"}) is None


def test_original_hsv_segmentation():
    image = np.full((350, 350, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (120, 130), (230, 220), (25, 180, 35), -1)
    features = extract_features_rgb(image)
    assert features["status"] == "detecte"
    assert 0.01 < features["mask_fraction"] < 0.15
    assert feature_vector(features) is not None


def test_conditional_classification():
    train_df = make_features()
    models = fit_color_models(train_df, VARIANTS, n_neighbors=3)
    for (typ, color), (h, s, v) in PROFILES.items():
        features = {"status": "detecte", "hue_mean": h,
                    "s_median": s, "v_median": v}
        predicted = predict_color(models, typ, features)
        assert predicted["prediction"] == color, (typ, color, predicted)
    assert predict_color(models, "ourson", {"status": "non_detecte"})["prediction"] == ABSTAIN
    assert predict_color(models, "oeuf", {"status": "non_detecte"})["prediction"] == "blanc"
    summary, predictions = evaluate_color_model(models, make_features(repeats=1, session="S02"))
    assert summary["images"] == len(PROFILES)
    assert summary["correct"] == len(PROFILES)
    assert summary["accuracy_variant_conditionnelle"] == 1.0
    assert predictions["source"].isin(["knn_hsv", "regle_du_pilote"]).all()


def test_split_protection():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        df = pd.DataFrame([{
            "image_path": "dummy.jpg", "type": "ourson",
            "couleur": "vert", "session_id": "S03"
        }])
        df.to_csv(root / "classification_val.csv", index=False)
        try:
            read_split("val", split_dir=root)
        except ValueError:
            pass
        else:
            raise AssertionError("Val avec session S03 aurait dû être refusé")
        try:
            read_split("test", split_dir=root)
        except ValueError:
            pass
        else:
            raise AssertionError("Le split test doit rester interdit")


if __name__ == "__main__":
    test_hue_is_circular()
    test_original_hsv_segmentation()
    test_conditional_classification()
    test_split_protection()
    print("ACS — Tests classificateur HSV : OK")
