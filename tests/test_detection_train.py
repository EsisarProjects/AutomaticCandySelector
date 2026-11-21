"""Tests légers : aucun téléchargement de poids ni entraînement requis.

    python tests/test_detection_train.py
"""
from pathlib import Path
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.train import (  # noqa: E402
    augmentation_settings,
    inspect_dataset,
    parse_args,
)


def test_augmentation_modes():
    baseline = augmentation_settings(False)
    assert all(baseline[k] == 0 for k in (
        "hsv_h", "hsv_s", "hsv_v", "degrees", "scale", "translate",
        "fliplr", "flipud", "mosaic", "mixup", "copy_paste"
    ))
    assert baseline["augmentations"] == []

    augmented = augmentation_settings(True)
    assert augmented["degrees"] == 15.0
    assert augmented["fliplr"] == 0.50
    assert augmented["hsv_h"] == 0.0
    assert augmented["mosaic"] == 0.0


def test_dataset_paths():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for split in ("train", "valid", "test"):
            images = base / split / "images"
            labels = base / split / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (images / "photo.png").write_bytes(b"fake-image-for-path-test")
            (labels / "photo.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        data_yaml = base / "data.local.yaml"
        data_yaml.write_text(yaml.safe_dump({
            "path": str(base), "train": "train/images", "val": "valid/images",
            "test": "test/images", "nc": 1, "names": {0: "bonbon"}
        }), encoding="utf-8")
        result = inspect_dataset(data_yaml)
        assert result["images"] == {"train": 1, "val": 1, "test": 1}

        broken = yaml.safe_load(data_yaml.read_text())
        broken["names"] = {0: "bonbons"}
        data_yaml.write_text(yaml.safe_dump(broken), encoding="utf-8")
        try:
            inspect_dataset(data_yaml)
        except ValueError:
            pass
        else:
            raise AssertionError("La classe 'bonbons' aurait dû être rejetée")


def test_cli():
    args = parse_args(["--check-only", "--device", "cpu"])
    assert args.check_only
    assert args.device == "cpu"
    assert args.epochs == 30


if __name__ == "__main__":
    test_augmentation_modes()
    test_dataset_paths()
    test_cli()
    print("ACS — Tests préparation entraînement YOLO : OK")
