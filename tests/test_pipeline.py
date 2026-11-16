"""Tests ACS 1.7 : pipeline bout en bout sur modèles et images synthétiques.

Exécution depuis la racine du projet : python tests/test_pipeline.py
Aucun téléchargement ImageNet ni accès à S03 nécessaire.
"""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.color.color_classifier import fit_color_models
from src.pipeline import ACSPipeline, evaluate_validation


class DummyClassifier(nn.Module):
    """Modèle synthétique prédisant toujours 'crocodile' (index 0)."""

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([5.0, -2.0]))

    def forward(self, images):
        return self.logits.unsqueeze(0).expand(len(images), -1)


def make_photo(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (400, 350), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((150, 110, 245, 240), fill=color)
    image.save(path)


def create_test_artifacts(tmp: Path) -> tuple[Path, Path]:
    variants = {"crocodile": ["bleu", "rouge"],
                "ourson": ["jaune", "vert"]}
    # Points fictifs juste pour valider les interfaces KNN.
    profiles = {
        ("crocodile", "bleu"): (110.0, 180.0, 200.0),
        ("crocodile", "rouge"): (0.0, 200.0, 200.0),
        ("ourson", "jaune"): (30.0, 200.0, 200.0),
        ("ourson", "vert"): (60.0, 200.0, 200.0),
    }
    rows = []
    for (typ, color), (h, s, v) in profiles.items():
        for _ in range(3):
            rows.append({"type": typ, "couleur": color, "status": "detecte",
                         "hue_mean": h, "s_median": s, "v_median": v})
    bundle = fit_color_models(pd.DataFrame(rows), variants, n_neighbors=3)
    color_path = tmp / "color.joblib"
    joblib.dump(bundle, color_path)

    type_path = tmp / "type.pth"
    model = DummyClassifier()
    torch.save({
        "architecture": "mobilenet_v3_small",
        "classes": ["crocodile", "ourson"],
        "class_to_idx": {"crocodile": 0, "ourson": 1},
        "preprocessing_method": "center_crop",
        "mode": "linear_probe",
        "epoch": 5,
        "model_state_dict": model.state_dict(),
    }, type_path)
    return type_path, color_path


def build_pipeline(tmp: Path) -> ACSPipeline:
    type_path, color_path = create_test_artifacts(tmp)
    # Injecte uniquement un classificateur synthétique dans les tests.
    with patch("src.pipeline.create_model", return_value=DummyClassifier()):
        return ACSPipeline(type_checkpoint=type_path, color_model=color_path,
                           device="cpu", root=tmp)


def test_rgb_input_and_type_predicted_only():
    with TemporaryDirectory() as directory:
        tmp = Path(directory)
        pipeline = build_pipeline(tmp)
        rgb = np.full((350, 400, 3), 255, dtype=np.uint8)
        rgb[110:240, 150:245] = (230, 30, 20)
        pred = pipeline.predict_rgb(rgb)
        assert pred["type_predit"] == "crocodile", pred
        assert pred["couleur_predite"] == "rouge", pred
        assert pred["variante_predite"] == "crocodile/rouge"
        assert pred["source_couleur"] == "knn_hsv"
        assert pred["segmentation_status"] == "detecte"
        assert 0 <= pred["confiance_type"] <= 1
        try:
            pipeline.predict_rgb(np.zeros((100, 100), dtype=np.uint8))
        except ValueError:
            pass
        else:
            raise AssertionError("Une image grayscale doit être refusée")


def test_validation_measures_joint_result_not_oracle_type():
    with TemporaryDirectory() as directory:
        tmp = Path(directory)
        pipeline = build_pipeline(tmp)
        make_photo(tmp / "red.png", (230, 30, 20))
        make_photo(tmp / "green.png", (30, 180, 30))
        splits = tmp / "datasets/metadata/splits"
        splits.mkdir(parents=True)
        pd.DataFrame([
            {"image_path": "red.png", "type": "crocodile",
             "couleur": "rouge", "session_id": "S02"},
            {"image_path": "green.png", "type": "ourson",
             "couleur": "vert", "session_id": "S02"},
        ]).to_csv(splits / "classification_val.csv", index=False)
        summary, paths = evaluate_validation(pipeline, split_dir=splits,
                                             results_dir=tmp / "results")
        assert summary["n_images"] == 2
        assert summary["n_type_correct"] == 1
        assert summary["n_variant_correct"] == 1
        assert summary["n_variant_errors"] == 1
        assert summary["variant_accuracy"] == 0.5
        predictions = pd.read_csv(paths["predictions"])
        assert predictions["type_predit"].tolist() == ["crocodile", "crocodile"]
        assert predictions["type_reel"].tolist() == ["crocodile", "ourson"]
        assert predictions["variante_correcte"].tolist() == [True, False]
        assert all(path.is_file() for path in paths.values())


def test_split_guard_rejects_test_session():
    with TemporaryDirectory() as directory:
        tmp = Path(directory)
        pipeline = build_pipeline(tmp)
        splits = tmp / "datasets/metadata/splits"
        splits.mkdir(parents=True)
        pd.DataFrame([{"image_path": "missing.jpg", "type": "crocodile",
                       "couleur": "rouge", "session_id": "S03"}]).to_csv(
            splits / "classification_val.csv", index=False
        )
        try:
            evaluate_validation(pipeline, split_dir=splits, results_dir=tmp / "results")
        except ValueError:
            pass
        else:
            raise AssertionError("S03 ne doit pas être évalué comme S02")


if __name__ == "__main__":
    test_rgb_input_and_type_predicted_only()
    test_validation_measures_joint_result_not_oracle_type()
    test_split_guard_rejects_test_session()
    print("ACS — Tests pipeline type + couleur : OK")
