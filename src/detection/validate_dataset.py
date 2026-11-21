"""ACS — Validation d'un dataset de détection exporté par Roboflow (YOLOv8).

Exécuter depuis la racine du projet :
    python -m src.detection.validate_dataset
    python -m src.detection.validate_dataset --make-local-yaml
    python -m src.detection.validate_dataset --root datasets/detection/roboflow/v1_baseline

Dépendances : Pillow, PyYAML.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "datasets/detection/roboflow/v1_baseline"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/detection/validation_v1"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_ALIASES = {"train": ("train",), "val": ("valid", "val"), "test": ("test",)}
EPS = 1e-5


def find_split_dirs(root: Path, split: str) -> tuple[Path, Path, str]:
    for alias in SPLIT_ALIASES[split]:
        image_dir = root / alias / "images"
        if image_dir.is_dir():
            return image_dir, root / alias / "labels", alias
    raise FileNotFoundError(
        f"Pas de dossier images pour {split}. Attendu : "
        + " ou ".join(str(root / a / 'images') for a in SPLIT_ALIASES[split])
    )


def parse_yolo_label(path: Path, issues: list[str]) -> list[tuple[float, float, float, float]]:
    """Renvoie des boîtes valides en xywh normalisé ; consigne toute anomalie."""
    boxes = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        issues.append(f"ERREUR — lecture label : {path}: {exc}")
        return boxes

    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        label_ref = f"{path}:{lineno}"
        if len(fields) != 5:
            issues.append(f"ERREUR — attendu classe xc yc w h (5 champs): {label_ref}")
            continue
        try:
            class_value = float(fields[0])
            coords = tuple(float(value) for value in fields[1:])
        except ValueError:
            issues.append(f"ERREUR — valeurs non numériques: {label_ref}")
            continue
        if not all(math.isfinite(x) for x in (class_value, *coords)):
            issues.append(f"ERREUR — valeur non finie: {label_ref}")
            continue
        if class_value != 0:
            issues.append(f"ERREUR — classe {fields[0]} au lieu de 0 (bonbon): {label_ref}")
            continue
        xc, yc, width, height = coords
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < width <= 1 and 0 < height <= 1):
            issues.append(f"ERREUR — coordonnées hors plage: {label_ref}: {coords}")
            continue
        left, right = xc - width / 2, xc + width / 2
        top, bottom = yc - height / 2, yc + height / 2
        if left < -EPS or right > 1 + EPS or top < -EPS or bottom > 1 + EPS:
            issues.append(f"ERREUR — boîte déborde de l'image: {label_ref}: {coords}")
            continue
        boxes.append(coords)
    return boxes


def inspect_yaml(root: Path, issues: list[str]) -> dict:
    yaml_path = root / "data.yaml"
    if not yaml_path.is_file():
        issues.append(f"ERREUR — data.yaml absent: {yaml_path}")
        return {}
    try:
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError) as exc:
        issues.append(f"ERREUR — data.yaml illisible: {exc}")
        return {}
    names = config.get("names")
    if isinstance(names, list):
        names_list = names
    elif isinstance(names, dict):
        names_list = [value for _, value in sorted(names.items(), key=lambda p: int(p[0]))]
    else:
        names_list = []
    if names_list != ["bonbon"]:
        issues.append(f"ERREUR — une classe attendue ['bonbon'], trouvée: {names!r}")
    if config.get("nc", len(names_list)) != 1:
        issues.append(f"ERREUR — nc doit être égal à 1 : {config.get('nc')!r}")
    for split in ("train", "val", "test"):
        if split not in config:
            issues.append(f"AVERTISSEMENT — clé '{split}' absente de data.yaml")
    # Certaines exportations Roboflow utilisent '../train/images'.
    # Le fichier local généré en option aura des chemins explicites.
    return config


def digest(path: Path) -> str:
    result = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def draw_previews(
    output: Path,
    examples: dict[str, list[tuple[Path, list[tuple[float, float, float, float]]]]],
    max_per_split: int = 3,
) -> list[str]:
    """Crée une planche visuelle par split, sans modifier les images originales."""
    paths = []
    output.mkdir(parents=True, exist_ok=True)
    for split, images in examples.items():
        selected = images[:max_per_split]
        if not selected:
            continue
        width, height, padding = 480, 380, 14
        montage = Image.new("RGB", (width * len(selected), height), "white")
        for idx, (image_path, boxes) in enumerate(selected):
            with Image.open(image_path) as im:
                base = ImageOps.exif_transpose(im).convert("RGB")
            iw, ih = base.size
            thumb = base.copy()
            thumb.thumbnail((width - 2 * padding, height - 80), Image.Resampling.LANCZOS)
            scale_x, scale_y = thumb.width / iw, thumb.height / ih
            draw = ImageDraw.Draw(thumb)
            for xc, yc, bw, bh in boxes:
                x1 = (xc - bw / 2) * iw * scale_x
                y1 = (yc - bh / 2) * ih * scale_y
                x2 = (xc + bw / 2) * iw * scale_x
                y2 = (yc + bh / 2) * ih * scale_y
                draw.rectangle((x1, y1, x2, y2), outline="#00e676", width=3)
            offset_x = idx * width + (width - thumb.width) // 2
            offset_y = 12 + (height - 80 - thumb.height) // 2
            montage.paste(thumb, (offset_x, offset_y))
            caption = ImageDraw.Draw(montage)
            caption.text((idx * width + padding, height - 52), image_path.name[:62], fill="black")
            caption.text((idx * width + padding, height - 32), f"{len(boxes)} boxes", fill="black")
        dest = output / f"preview_{split}.jpg"
        montage.save(dest, quality=95)
        paths.append(str(dest))
    return paths


def validate_dataset(root: Path, output: Path, expected_total: int = 52, max_preview: int = 3) -> dict:
    root = root.expanduser().resolve()
    output = output.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset absent: {root}")
    issues: list[str] = []
    inspect_yaml(root, issues)
    stats: dict[str, dict] = {}
    all_hashes: dict[str, tuple[str, str]] = {}
    examples: dict[str, list] = {}

    for split in ("train", "val", "test"):
        image_dir, label_dir, _ = find_split_dirs(root, split)
        if not label_dir.is_dir():
            issues.append(f"ERREUR — dossier labels absent: {label_dir}")
        images = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            issues.append(f"ERREUR — aucune image dans {image_dir}")
        file_counts = Counter(p.stem for p in images)
        for stem, count in file_counts.items():
            if count > 1:
                issues.append(f"ERREUR — même nom de base '{stem}' dans {split} ({count} images)")
        annotated_count = 0
        empty_count = 0
        boxes_count = 0
        valid_examples = []
        image_stems = set(file_counts)
        if label_dir.is_dir():
            for lab in sorted(label_dir.glob("*.txt")):
                if lab.stem not in image_stems:
                    issues.append(f"ERREUR — annotation sans image: {lab}")

        for image_path in images:
            try:
                with Image.open(image_path) as im:
                    im.load()
                    w, h = im.size
                    if w < 1 or h < 1:
                        raise ValueError("dimensions nulles")
            except (OSError, ValueError) as exc:
                issues.append(f"ERREUR — image illisible: {image_path}: {exc}")
                continue

            image_hash = digest(image_path)
            if image_hash in all_hashes:
                other_split, other_name = all_hashes[image_hash]
                if split != other_split:
                    issues.append(
                        f"ERREUR — doublon exact entre {other_split}/{split}: "
                        f"{other_name} et {image_path.name}"
                    )
                else:
                    issues.append(f"AVERTISSEMENT — doublon exact dans {split}: {image_path.name}")
            else:
                all_hashes[image_hash] = (split, image_path.name)

            label_path = label_dir / (image_path.stem + ".txt")
            if label_path.is_file():
                boxes = parse_yolo_label(label_path, issues)
            else:
                boxes = []
                issues.append(
                    f"AVERTISSEMENT — annotation absente (image négative possible): {image_path.name}"
                )
            if boxes:
                annotated_count += 1
            else:
                empty_count += 1
                issues.append(f"AVERTISSEMENT — aucune boîte valide: {image_path.name}")
            boxes_count += len(boxes)
            valid_examples.append((image_path, boxes))

        stats[split] = {
            "images": len(images),
            "images_with_boxes": annotated_count,
            "images_without_boxes": empty_count,
            "boxes": boxes_count,
        }
        examples[split] = valid_examples

    total_images = sum(s["images"] for s in stats.values())
    total_boxes = sum(s["boxes"] for s in stats.values())
    if expected_total and total_images != expected_total:
        issues.append(f"AVERTISSEMENT — total images = {total_images}, attendu = {expected_total}")
    if stats["val"]["images"] < 1 or stats["test"]["images"] < 1:
        issues.append("ERREUR — validation et test doivent contenir des images")

    output.mkdir(parents=True, exist_ok=True)
    preview_paths = draw_previews(output, examples, max_preview)
    errors = [x for x in issues if x.startswith("ERREUR")]
    warnings = [x for x in issues if x.startswith("AVERTISSEMENT")]
    report = {
        "root": str(root),
        "total_images": total_images,
        "total_boxes": total_boxes,
        "splits": stats,
        "errors": errors,
        "warnings": warnings,
        "preview_paths": preview_paths,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== ACS — Validation du dataset YOLO ===")
    for name, s in stats.items():
        print(
            f"{name:5} | images={s['images']:2} | boxes={s['boxes']:3} "
            f"| sans boîte={s['images_without_boxes']:2}"
        )
    print(f"TOTAL | images={total_images} | boxes={total_boxes}")
    print(f"Erreurs={len(errors)}, avertissements={len(warnings)}")
    for msg in issues[:20]:
        print(msg)
    if len(issues) > 20:
        print(f"... {len(issues) - 20} autres messages dans {report_path}")
    print(f"Rapport : {report_path}")
    for path in preview_paths:
        print(f"Prévisualisation : {path}")
    return report


def write_local_yaml(root: Path) -> Path:
    """Crée un YAML local sans écraser celui exporté par Roboflow."""
    aliases = {}
    for split in ("train", "val", "test"):
        _, _, alias = find_split_dirs(root, split)
        aliases[split] = alias
    data = {
        "path": str(root.resolve()),
        "train": f"{aliases['train']}/images",
        "val": f"{aliases['val']}/images",
        "test": f"{aliases['test']}/images",
        "nc": 1,
        "names": {0: "bonbon"},
    }
    destination = root / "data.local.yaml"
    destination.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"YAML local généré : {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Vérifie le dataset YOLO (Roboflow) de ACS.")
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-total", type=int, default=52)
    parser.add_argument("--preview", type=int, default=3, help="Images par split dans les planches")
    parser.add_argument("--make-local-yaml", action="store_true")
    args = parser.parse_args()
    report = validate_dataset(args.root, args.output, args.expected_total, args.preview)
    if args.make_local_yaml and not report["errors"]:
        write_local_yaml(args.root.expanduser().resolve())
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
