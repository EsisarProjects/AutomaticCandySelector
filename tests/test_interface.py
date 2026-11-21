"""Headless tests for the ACS graphical interface.

Run from the project root: python tests/test_interface.py
"""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.interface.service import Settings, analyze_batch, collect_images, validate_settings  # noqa: E402


class FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([[8, 10, 37, 42]])
        self.conf = FakeTensor([0.92])

    def __len__(self):
        return 1


class FakeDetector:
    def predict(self, **kwargs):
        assert kwargs["conf"] == 0.45
        assert kwargs["source"].shape[2] == 3
        return [SimpleNamespace(boxes=FakeBoxes())]


class FakeClassifier:
    def predict_rgb(self, rgb):
        assert rgb.dtype == np.uint8
        return {
            "type_predit": "crocodile", "confiance_type": 0.94,
            "couleur_predite": "vert", "variante_predite": "crocodile/vert",
            "source_couleur": "knn_hsv", "segmentation_status": "detecte",
            "score_couleur": 0.81,
        }


def run() -> None:
    with TemporaryDirectory() as temp:
        tmp = Path(temp)
        folder = tmp / "photos"
        folder.mkdir()
        for i in range(2):
            pixels = np.full((75, 100, 3), 225, dtype=np.uint8)
            pixels[10:42, 8:37] = (40, 128 + i, 180)
            Image.fromarray(pixels).save(folder / f"bonbon_{i}.png")
        (folder / "notes.txt").write_text("pas une image", encoding="utf-8")
        files = collect_images([folder, folder / "bonbon_0.png"])
        assert len(files) == 2  # Duplicates removed
        assert all(p.is_absolute() for p in files)

        settings = Settings(output=tmp / "resultats", confidence=0.45, device="cpu")
        validate_settings(settings, require_models=False)
        events = []
        loaded = []

        def loader(s):
            loaded.append(s.device)
            return FakeDetector(), FakeClassifier(), "cpu"

        reports = analyze_batch(
            files, settings,
            emit=lambda name, data: events.append((name, data)),
            model_loader=loader,
        )
        assert loaded == ["cpu"]  # One model load for multiple photos
        assert len(reports) == 2
        assert [event[0] for event in events].count("result") == 2
        assert events[-1][0] == "done"
        for report in reports:
            assert report["nombre_detections"] == 1
            assert Path(report["image_annotee"]).is_file()
            assert report["detections"][0]["variante_predite"] == "crocodile/vert"
            assert report["detections"][0]["statut"] == "prediction_disponible"
            with Image.open(report["image_annotee"]) as image:
                assert image.width > 100  # External legend panel
        assert len(list((tmp / "resultats").glob("*_predictions.json"))) == 2

        cancel = Event()
        cancelled_events = []

        def cancel_after_first(name, data):
            cancelled_events.append(name)
            if name == "result":
                cancel.set()

        partial = analyze_batch(files, settings, emit=cancel_after_first,
                                model_loader=loader, cancelled=cancel)
        assert len(partial) == 1
        assert cancelled_events[-1] == "cancelled"

        try:
            validate_settings(Settings(confidence=1.5), require_models=False)
            raise AssertionError("Invalid confidence was not rejected")
        except ValueError:
            pass
        try:
            collect_images([tmp / "inexistant"])
            raise AssertionError("Missing input image was not rejected")
        except FileNotFoundError:
            pass
        # GUI and service sources must remain ASCII-only to avoid glyph warnings.
        interface_dir = ROOT / "src" / "interface"
        for filename in ("app.py", "service.py", "__init__.py"):
            source = (interface_dir / filename).read_text(encoding="utf-8")
            assert source.isascii(), f"Non-ASCII characters in {filename}"
        app_source = (interface_dir / "app.py").read_text(encoding="utf-8")
        assert 'text="Previous"' in app_source
        assert 'text="Next"' in app_source
        assert '"Review" if item.get("avertissements")' in app_source
        print("ACS - Interface stage 2.6.1 tests: OK")


if __name__ == "__main__":
    run()
