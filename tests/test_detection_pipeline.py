"""Tests ACS 2.5 : intégration des stages sur données SYNTHÉTIQUES.

    python tests/test_detection_pipeline.py

Aucun poids pré-entraîné ni accès aux images de test n'est nécessaire.
"""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.detection.integrated_pipeline import (  # noqa: E402
    crop_box, draw_detections, process_file, run_stages,
)


class TensorDummy:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class BoxesDummy:
    def __init__(self, coords, scores):
        self.xyxy = TensorDummy(np.asarray(coords, dtype=float).reshape(-1, 4))
        self.conf = TensorDummy(np.asarray(scores, dtype=float))
        self.scores = scores

    def __len__(self):
        return len(self.scores)


class DetectorDummy:
    def __init__(self, coords, scores):
        self.boxes = BoxesDummy(coords, scores)
        self.last_source = None

    def predict(self, **kwargs):
        self.last_source = kwargs["source"]
        assert kwargs["conf"] == 0.45
        assert self.last_source.dtype == np.uint8
        return [SimpleNamespace(boxes=self.boxes)]


class ClassificationDummy:
    def __init__(self):
        self.seen_shapes = []

    def predict_rgb(self, rgb):
        assert rgb.dtype == np.uint8 and rgb.shape[2] == 3
        self.seen_shapes.append(rgb.shape)
        return {
            "type_predit": "crocodile",
            "confiance_type": 0.93,
            "couleur_predite": "vert",
            "score_couleur": 0.81,
            "source_couleur": "knn_hsv",
            "variante_predite": "crocodile/vert",
            "segmentation_status": "detecte",
        }


def main():
    rgb = np.full((100, 160, 3), 255, dtype=np.uint8)
    rgb[12:40, 15:58, :] = (80, 180, 50)
    rgb[55:92, 80:130, :] = (150, 70, 20)

    crop, coords = crop_box(rgb, [12, 10, 60, 43], padding=0)
    assert coords == [12, 10, 60, 43]
    assert crop.shape == (33, 48, 3)

    detector = DetectorDummy([[12, 10, 60, 43], [78, 53, 134, 93]], [0.91, 0.83])
    classifier = ClassificationDummy()
    detections, crops = run_stages(rgb, detector=detector, classifier=classifier)
    assert len(detections) == len(crops) == 2
    assert classifier.seen_shapes[0][0] > 0
    assert detections[0]["variante_predite"] == "crocodile/vert"
    assert detections[0]["statut"] == "prediction_disponible"
    assert np.array_equal(detector.last_source, rgb[:, :, ::-1])
    annotated = draw_detections(rgb, detections)
    # La photo 160x100 est conservée à gauche ; les libellés sont à droite.
    assert annotated.width > 160
    assert annotated.height >= 100
    assert annotated.getpixel((160 + 30, 10)) != (255, 255, 255)
    # La fonction reste correcte en absence de détection.
    assert draw_detections(rgb, []).width > rgb.shape[1]

    # Aucun bonbon : image annotée et fichier JSON toujours générés.
    empty = DetectorDummy([], [])
    no_detections, no_crops = run_stages(rgb, detector=empty, classifier=classifier)
    assert no_detections == [] and no_crops == []
    with TemporaryDirectory() as temp:
        tmp = Path(temp)
        img = tmp / "photo.png"
        Image.fromarray(rgb).save(img)
        report = process_file(
            img, detector=detector, classifier=classifier,
            output=tmp / "sorties", detector_conf=0.45,
            imgsz=640, device="cpu", iou=0.50,
            padding=0.02, type_warning_threshold=0.60,
            save_crops=True,
        )
        assert report["nombre_detections"] == 2
        assert Path(report["image_annotee"]).is_file()
        assert len(list((tmp / "sorties").glob("*_predictions.json"))) == 1
        assert all(Path(row["image_crop"]).is_file() for row in report["detections"])
    print("ACS — Tests pipeline YOLO + MobileNet + HSV/KNN : OK")


if __name__ == "__main__":
    main()
