"""Tests unitaires de l'analyse des performances ACS, sans dataset réel."""
from pathlib import Path
import sys
import tempfile

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import compute_classification_metrics, get_split_session
from src.evaluation.plots import plot_confusion_matrix, plot_prediction_errors


def test_evaluation():
    assert get_split_session("val") == "S02"
    assert get_split_session("test") == "S03"
    try:
        get_split_session("train")
    except ValueError:
        pass
    else:
        raise AssertionError("Le split train ne doit pas être accepté ici")

    classes = ["crocodile", "ourson", "fraise", "oeuf"]
    # 4 images ; une confusion : fraise -> ourson.
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 1, 3]
    matrix, report, summary = compute_classification_metrics(y_true, y_pred, classes)

    assert matrix.shape == (4, 4)
    assert matrix[2, 1] == 1
    assert matrix.trace() == 3
    assert summary["n_errors"] == 1
    assert abs(summary["accuracy"] - 0.75) < 1e-9
    assert set(classes).issubset(set(report["classe"]))

    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        image_path = tmp / "exemple.jpg"
        Image.new("RGB", (400, 300), (230, 40, 40)).save(image_path)
        matrix_path = tmp / "confusion.png"
        error_path = tmp / "errors.png"
        plot_confusion_matrix(matrix, classes, matrix_path, dataset_name="test S03")
        predictions = pd.DataFrame([{
            "image_path": str(image_path),
            "type": "fraise",
            "couleur": "rouge",
            "prediction": "ourson",
            "confidence": 0.85,
            "correct": False,
        }])
        plot_prediction_errors(predictions, tmp, "padding", error_path, dataset_name="test S03")
        assert matrix_path.is_file() and matrix_path.stat().st_size > 0
        assert error_path.is_file() and error_path.stat().st_size > 0

    print("Tests des métriques et des figures : OK")


if __name__ == "__main__":
    test_evaluation()