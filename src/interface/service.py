"""Inference service for the ACS interface, independent from Tkinter.

Reuses the tested stage 2.5.1 functions and caches loaded models
across consecutive analyses.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Callable, Iterable

from src.detection.integrated_pipeline import (
    DEFAULT_COLOR,
    DEFAULT_DETECTOR,
    DEFAULT_OUTPUT,
    DEFAULT_TYPE,
    gather_images,
    process_file,
)


@dataclass(frozen=True)
class Settings:
    detector: Path = DEFAULT_DETECTOR
    type_checkpoint: Path = DEFAULT_TYPE
    color_model: Path = DEFAULT_COLOR
    output: Path = DEFAULT_OUTPUT / "interface"
    device: str = "cpu"
    confidence: float = 0.45
    iou: float = 0.50
    imgsz: int = 640
    padding: float = 0.02
    type_warning: float = 0.60
    save_crops: bool = False


def collect_images(sources: Iterable[Path | str]) -> list[Path]:
    """Collect image paths from files and folders (one level), without duplicates."""
    files: list[Path] = []
    seen: set[Path] = set()
    for source in sources:
        for image in gather_images(Path(source).expanduser().resolve()):
            resolved = image.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(resolved)
    return files


def validate_settings(settings: Settings, *, require_models: bool = True) -> None:
    if settings.device not in ("cpu", "auto", "cuda"):
        raise ValueError("Invalid device: expected cpu, auto, or cuda")
    if not (0 < settings.confidence <= 1 and 0 < settings.iou <= 1):
        raise ValueError("Confidence and IoU must be in (0, 1]")
    if not (0 <= settings.padding <= 0.5 and 0 <= settings.type_warning <= 1):
        raise ValueError("Padding or warning threshold out of range")
    if settings.imgsz < 32:
        raise ValueError("imgsz must be >= 32")
    if not str(settings.output).strip():
        raise ValueError("Output folder is not defined")
    if require_models:
        for label, path in (
            ("YOLO", settings.detector),
            ("MobileNetV3", settings.type_checkpoint),
            ("HSV/KNN", settings.color_model),
        ):
            if not Path(path).expanduser().is_file():
                raise FileNotFoundError(f"Model {label} was not found: {path}")


class ModelCache:
    """Cache the (YOLO, ACSPipeline) pair for a given model configuration."""

    def __init__(self) -> None:
        self.key: tuple[str, str, str, str] | None = None
        self.models: tuple[Any, Any, str] | None = None

    def get(self, settings: Settings) -> tuple[Any, Any, str]:
        validate_settings(settings)
        paths = [str(Path(p).expanduser().resolve()) for p in (
            settings.detector, settings.type_checkpoint, settings.color_model
        )]
        key = (*paths, settings.device)
        if self.key == key and self.models is not None:
            return self.models

        # Import heavy modules only when starting inference.
        if settings.device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = settings.device

        from ultralytics import YOLO
        from src.pipeline import ACSPipeline

        detector = YOLO(paths[0])
        classifier = ACSPipeline(
            type_checkpoint=Path(paths[1]),
            color_model=Path(paths[2]),
            device=device,
        )
        self.key = key
        self.models = (detector, classifier, device)
        return self.models


Emit = Callable[[str, dict[str, Any]], None]


def analyze_batch(
    files: list[Path],
    settings: Settings,
    *,
    emit: Emit,
    cache: ModelCache | None = None,
    cancelled: Event | None = None,
    model_loader: Callable[[Settings], tuple[Any, Any, str]] | None = None,
    processor: Callable[..., dict[str, Any]] = process_file,
) -> list[dict[str, Any]]:
    """Process images and emit events to the GUI thread.

    This function never calls Tkinter. Cancellation occurs between images.
    `model_loader` and `processor` can be injected for synthetic tests.
    """
    if not files:
        raise ValueError("No images selected")
    validate_settings(settings, require_models=(model_loader is None))
    if cancelled is not None and cancelled.is_set():
        emit("cancelled", {"processed": 0, "total": len(files)})
        return []
    emit("status", {"message": "Loading models..."})
    if model_loader is not None:
        detector, classifier, device = model_loader(settings)
    else:
        detector, classifier, device = (cache or ModelCache()).get(settings)

    output = Path(settings.output).expanduser().resolve()
    reports: list[dict[str, Any]] = []
    for position, image_path in enumerate(files, start=1):
        if cancelled is not None and cancelled.is_set():
            emit("cancelled", {"processed": len(reports), "total": len(files)})
            return reports
        emit("progress", {"current": position, "total": len(files),
                          "image": str(image_path)})
        report = processor(
            image_path, detector=detector, classifier=classifier,
            output=output, detector_conf=settings.confidence,
            imgsz=settings.imgsz, device=device, iou=settings.iou,
            padding=settings.padding, type_warning_threshold=settings.type_warning,
            save_crops=settings.save_crops,
        )
        reports.append(report)
        emit("result", {"current": position, "total": len(files), "report": report})
    emit("done", {"processed": len(reports), "total": len(files),
                  "output": str(output)})
    return reports
