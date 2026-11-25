"""ACS — Train YOLO and publish its best weights under models/detection/.

À lancer depuis la racine du projet :
    python -m src.detection.train --check-only
    python -m src.detection.train --smoke-test --device cpu
    python -m src.detection.train --device cpu

Le jeu TEST n'est jamais utilisé pour choisir le modèle.
Training parameters and paths come from configs/detection.yaml.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from src.config import load_config, path_from_root

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_config('detection')
TRAINING = CONFIG['training']
DEFAULT_DATA = path_from_root(CONFIG['dataset']['yaml'])
DEFAULT_OUTPUT = path_from_root(TRAINING['experiment_dir'])
DEFAULT_CHECKPOINT = path_from_root(CONFIG['checkpoint'])
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _class_names(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, dict):
        try:
            return [str(value[k]) for k in sorted(value, key=lambda x: int(x))]
        except (ValueError, TypeError):
            return []
    return []


def inspect_dataset(data_file: Path) -> dict[str, Any]:
    """Vérifie les chemins et le nombre de classes sans lancer d'entraînement."""
    data_file = data_file.expanduser().resolve()
    if not data_file.is_file():
        raise FileNotFoundError(
            f"YAML absent : {data_file}\n"
            "Create datasets/v2/detection/yolo/data.yaml from data.yaml.example"
        )

    config = yaml.safe_load(data_file.read_text(encoding="utf-8")) or {}
    names = _class_names(config.get("names"))
    if names != ["bonbon"] or int(config.get("nc", len(names))) != 1:
        raise ValueError(
            f"Une seule classe 'bonbon' est attendue, mais names={config.get('names')!r}, "
            f"nc={config.get('nc', '<absent>')!r}."
        )

    dataset_root = Path(config.get("path", data_file.parent))
    if not dataset_root.is_absolute():
        dataset_root = data_file.parent / dataset_root
    dataset_root = dataset_root.resolve()

    folders: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        value = config.get(split)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Chemin du split '{split}' manquant dans {data_file}.")
        image_dir = Path(value)
        if not image_dir.is_absolute():
            image_dir = dataset_root / image_dir
        image_dir = image_dir.resolve()
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Dossier images du split '{split}' absent : {image_dir}")
        images = [p for p in image_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        if not images:
            raise ValueError(f"Aucune image dans le split '{split}': {image_dir}")
        label_dir = image_dir.parent / "labels"
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Dossier annotations absent : {label_dir}")
        folders[split] = str(image_dir)
        counts[split] = len(images)

    return {
        "yaml": str(data_file),
        "root": str(dataset_root),
        "classes": names,
        "folders": folders,
        "images": counts,
    }


def augmentation_settings(enabled: bool) -> dict[str, Any]:
    """Training-time augmentation settings (configurable by flag)."""
    settings: dict[str, Any] = {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        # Désactive le petit ensemble d'augmentations Albumentations
        # par défaut, si cette option est prise en charge par Ultralytics.
        "augmentations": [],
    }
    if enabled:
        settings.update({
            "hsv_v": 0.10,
            "degrees": 15.0,
            "translate": 0.05,
            "scale": 0.10,
            "flipud": 0.50,
            "fliplr": 0.50,
            # Mosaic reste désactivé pour comparer une augmentation modérée.
        })
    return settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACS — Entraînement YOLOv8n")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Chemin data.local.yaml")
    parser.add_argument("--model", default=TRAINING["pretrained"], help="Initial YOLO pretrained weights")
    parser.add_argument("--device", default=str(TRAINING["device"]), help="cpu or GPU id (e.g. 0)")
    parser.add_argument("--epochs", type=int, default=int(TRAINING["epochs"]))
    parser.add_argument("--batch", type=int, default=int(TRAINING["batch_size"]))
    parser.add_argument("--imgsz", type=int, default=int(TRAINING["image_size"]))
    parser.add_argument("--patience", type=int, default=int(TRAINING["patience"]))
    parser.add_argument("--workers", type=int, default=int(TRAINING["workers"]))
    parser.add_argument("--seed", type=int, default=int(TRAINING["seed"]))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--name", default=str(TRAINING["name"]))
    parser.add_argument("--augment", action="store_true", default=bool(TRAINING.get("augment", False)), help="Enable YOLO training augmentation")
    parser.add_argument("--no-augment", dest="augment", action="store_false", help="Disable YOLO training augmentation")
    parser.add_argument("--check-only", action="store_true", help="Vérifie les chemins sans importer YOLO")
    parser.add_argument("--smoke-test", action="store_true", help="Une époque à 416 px pour tester le pipeline")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32 or args.patience < 0 or args.workers < 0:
        parser.error("Paramètres d'entraînement invalides.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = inspect_dataset(args.data)

    print("\n=== ACS — Entraînement du détecteur YOLOv8n ===")
    print(f"Dataset : {summary['yaml']}")
    print(f"Classes : {summary['classes']}")
    for split, count in summary["images"].items():
        print(f"{split:5} : {count} images")
    print(f"Modèle : {args.model}")
    print(f"Device : {args.device}")
    print(f"Augmentation locale : {'oui' if args.augment else 'non'}")

    if args.check_only:
        print("Vérification des chemins : OK. Aucun entraînement lancé.")
        return

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics absent. Installez-le avec : python -m pip install ultralytics"
        ) from exc

    epochs = 1 if args.smoke_test else args.epochs
    imgsz = 416 if args.smoke_test else args.imgsz
    run_name = "yolov8n_smoke" if args.smoke_test else args.name
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    train_options: dict[str, Any] = {
        "data": summary["yaml"],
        "epochs": epochs,
        "batch": args.batch,
        "imgsz": imgsz,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "deterministic": True,
        "patience": args.patience,
        "optimizer": "AdamW",
        "lr0": float(TRAINING["learning_rate"]),
        "weight_decay": float(TRAINING["weight_decay"]),
        "project": str(output),
        "name": run_name,
        "exist_ok": False,
        "plots": True,
        "val": True,
        "save": True,
        **augmentation_settings(args.augment),
    }

    print(f"\nLancement : {epochs} époque(s), batch={args.batch}, imgsz={imgsz}")
    print("Le split test n'est PAS utilisé pendant cet entraînement.\n")
    model = YOLO(args.model)
    model.train(**train_options)

    # Ultralytics incrémente automatiquement les noms si un dossier existe déjà.
    save_dir = Path(model.trainer.save_dir).resolve()
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    manifest = {
        "dataset": summary,
        "model_initial": args.model,
        "pretrained": True,
        "experiment": "smoke" if args.smoke_test else "training",
        "epochs_requested": epochs,
        "training_parameters": {k: v for k, v in train_options.items() if k != "augmentations"},
        "custom_albumentations": [],
        "best_weights": str(best) if best.exists() else None,
        "last_weights": str(last) if last.exists() else None,
    }
    (save_dir / "acs_experiment.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== ACS — Fin de l'entraînement ===")
    print(f"Résultats : {save_dir}")
    if best.exists() and not args.smoke_test:
        DEFAULT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, DEFAULT_CHECKPOINT)
        print(f"Deployed best detector checkpoint: {DEFAULT_CHECKPOINT}")
    elif not args.smoke_test:
        raise FileNotFoundError(f"YOLO did not generate expected best checkpoint: {best}")
    print(f"Meilleur modèle : {best if best.exists() else 'non trouvé'}")
    print("Le jeu test est réservé à l'évaluation finale.")


if __name__ == "__main__":
    main()