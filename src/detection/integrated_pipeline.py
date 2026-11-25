"""ACS — Étape 2.5 : détection + classification du type + prédiction couleur.

Uses src.pipeline.ACSPipeline.predict_rgb with ConvNeXt-Tiny.
Lancement depuis la racine du projet :
    python -m src.detection.integrated_pipeline --source photo.jpg --device cpu
    python -m src.detection.integrated_pipeline --source dossier_images --save-crops

Aucun apprentissage. Les vérités terrain ne sont jamais transmises aux modèles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.config import load_config, path_from_root

ROOT = Path(__file__).resolve().parents[2]
_DETECTION_CONFIG = load_config('detection')
_APP_CONFIG = load_config('application')
DEFAULT_DETECTOR = path_from_root(_DETECTION_CONFIG['checkpoint'])
DEFAULT_TYPE = path_from_root(load_config('classification')['checkpoint'])
DEFAULT_COLOR = path_from_root(load_config('color')['checkpoint'])
DEFAULT_OUTPUT = path_from_root(_APP_CONFIG['output_dir'])
DEFAULT_INFERENCE = _DETECTION_CONFIG['inference']
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def crop_box(rgb: np.ndarray, xyxy: tuple[float, float, float, float] | list[float],
             padding: float = 0.02) -> tuple[np.ndarray, list[int]]:
    """Découpe la ROI dans les coordonnées ORIGINALES, sans déformation.

    Le padding est une fraction de la largeur/hauteur de la boîte.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Image attendue : RGB uint8 de dimensions H×W×3")
    if not (0.0 <= padding <= 0.5):
        raise ValueError("padding doit appartenir à [0, 0.5]")
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    if not all(np.isfinite([x1, y1, x2, y2])) or x2 <= x1 or y2 <= y1:
        raise ValueError(f"Boîte invalide : {xyxy}")
    extra_x, extra_y = (x2 - x1) * padding, (y2 - y1) * padding
    left = max(0, min(width, int(np.floor(x1 - extra_x))))
    top = max(0, min(height, int(np.floor(y1 - extra_y))))
    right = max(0, min(width, int(np.ceil(x2 + extra_x))))
    bottom = max(0, min(height, int(np.ceil(y2 + extra_y))))
    if right <= left or bottom <= top:
        raise ValueError("Boîte située hors de l'image")
    return np.ascontiguousarray(rgb[top:bottom, left:right]), [left, top, right, bottom]


def _numpy(tensor: Any) -> np.ndarray:
    """Convertit les tenseurs PyTorch Ultralytics (ou les doubles de tests)."""
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    return np.asarray(tensor)


def run_stages(rgb: np.ndarray, *, detector: Any, classifier: Any,
               detector_conf: float = 0.45, imgsz: int = 640,
               device: str = "cpu", iou: float = 0.50,
               padding: float = 0.02,
               type_warning_threshold: float = 0.60
               ) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    """Exécute YOLO et ACSPipeline sur CHAQUE boîte détectée.

    Retourne les prédictions et les crops RGB, sans dépendre des étiquettes.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Image attendue : RGB uint8 H×W×3")
    if not (0 < detector_conf <= 1 and 0 < iou <= 1):
        raise ValueError("conf et iou doivent appartenir à ]0,1]")
    if not (0 <= type_warning_threshold <= 1):
        raise ValueError("type_warning_threshold doit appartenir à [0,1]")

    # Ultralytics attend les arrays numpy en BGR. La même orientation/pixels
    # sont utilisés pour les boîtes, les recadrages et les annotations.
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    predictions = detector.predict(
        source=bgr, conf=detector_conf, iou=iou, imgsz=imgsz,
        device=device, verbose=False, max_det=int(DEFAULT_INFERENCE["max_det"]),
    )
    if len(predictions) != 1:
        raise RuntimeError("Une seule image est attendue par appel au détecteur")
    boxes = predictions[0].boxes
    if boxes is None or len(boxes) == 0:
        return [], []
    coords = _numpy(boxes.xyxy).reshape(-1, 4)
    scores = _numpy(boxes.conf).reshape(-1)
    if len(coords) != len(scores):
        raise RuntimeError("Incohérence entre boxes.xyxy et boxes.conf")

    # Lecture stable : de haut en bas, puis de gauche à droite.
    order = sorted(range(len(coords)), key=lambda i: (float(coords[i, 1]), float(coords[i, 0])))
    detections: list[dict[str, Any]] = []
    crops: list[np.ndarray] = []
    for rank, idx in enumerate(order, start=1):
        crop, bounds = crop_box(rgb, coords[idx].tolist(), padding=padding)
        # Classification ConvNeXt and optional color inference.
        try:
            prediction = classifier.predict_rgb(crop)
        except Exception as exc:
            raise RuntimeError(
                f"Échec classification de la détection {rank} (bbox={bounds})"
            ) from exc
        if not isinstance(prediction, dict):
            raise TypeError("ACSPipeline.predict_rgb doit renvoyer un dictionnaire")

        candy_type = prediction["type_predit"]
        color = prediction["couleur_predite"]
        type_conf = float(prediction["confiance_type"])
        color_source = str(prediction.get("source_couleur", ""))
        segmentation_status = str(prediction.get("segmentation_status", ""))
        warnings = []
        if type_conf < type_warning_threshold:
            warnings.append("type_faible_confiance")
        if not str(color).strip() or str(color).lower() in {"indetermine", "unknown", "inconnu"}:
            warnings.append("couleur_indeterminee")
        # La couleur à variante unique est une règle de catalogue, pas
        # une mesure visuelle de la couleur de l'objet.
        if color_source not in {"catalog_rule", "regle_du_pilote"} and segmentation_status not in {"detecte", "not_required"}:
            warnings.append("segmentation_couleur_incomplete")

        detections.append({
            "id": rank,
            "classe_detection": "bonbon",
            "confiance_detection": float(scores[idx]),
            "bbox_xyxy": bounds,
            "type_predit": str(candy_type),
            "confiance_type": type_conf,
            "couleur_predite": str(color),
            "score_couleur": prediction.get("score_couleur"),
            "source_couleur": color_source,
            "variante_predite": str(prediction.get("variante_predite", f"{candy_type}/{color}")),
            "segmentation_status": segmentation_status,
            "avertissements": warnings,
            "statut": "a_verifier" if warnings else "prediction_disponible",
        })
        crops.append(crop)
    return detections, crops


def _legend_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Utilise une police disponible sans imposer de dépendance supplémentaire."""
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(filename, size=size)
    except (OSError, ValueError):
        return ImageFont.load_default()


def draw_detections(rgb: np.ndarray, detections: list[dict[str, Any]]) -> Image.Image:
    """Boîtes numérotées sur la photo ; légende COMPLÈTE dans un panneau externe.

    Les libellés détaillés ne recouvrent jamais les bonbons ni d'autres
    libellés. La photo conserve ses dimensions et coordonnées d'origine ;
    seule la taille de l'image annotée est élargie par le panneau latéral.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Image attendue : RGB uint8 H×W×3")

    height, width = rgb.shape[:2]
    panel_width = max(360, min(480, int(width * 0.65)))
    header_height = 90
    row_height = 49
    bottom_margin = 20
    output_height = max(height, header_height + row_height * len(detections) + bottom_margin)
    canvas = Image.new("RGB", (width + panel_width, output_height), (250, 251, 253))
    canvas.paste(Image.fromarray(rgb), (0, 0))
    draw = ImageDraw.Draw(canvas)

    green = (13, 157, 91)
    orange = (241, 142, 28)
    dark = (29, 44, 61)
    soft = (95, 107, 121)
    font_number = _legend_font(24, bold=True)
    font_title = _legend_font(26, bold=True)
    font_main = _legend_font(20, bold=True)
    font_small = _legend_font(22)

    # Les boîtes restent sur l'image ; seulement de petits identifiants,
    # de préférence au-dessus de chaque boîte, sont ajoutés à la photo.
    for item in detections:
        x1, y1, x2, y2 = (int(v) for v in item["bbox_xyxy"])
        x1, x2 = max(0, x1), min(width - 1, x2)
        y1, y2 = max(0, y1), min(height - 1, y2)
        attention = item.get("statut") != "prediction_disponible"
        stroke = orange if attention else green
        
        # 1. TĂNG ĐỘ DÀY KHUNG LÊN 6 (Cho cực kỳ đậm và nổi bật)
        draw.rectangle((x1, y1, x2, y2), outline=stroke, width=10)

        # 2. CHỈ ĐÁNH SỐ THỨ TỰ (Ví dụ: 01, 02...) KHỚP VỚI BẢNG BÊN PHẢI
        index = f"{item['id']:02d}" # Định dạng 2 chữ số: 01, 02, 03...
        
        # Dùng font to cho số thứ tự sắc nét
        font_badge = _legend_font(20, bold=True)
        bounds = draw.textbbox((0, 0), index, font=font_badge)
        
        badge_w = (bounds[2] - bounds[0]) + 16
        badge_h = 30
        bx = min(x1, width - badge_w)
        by = (y1 - badge_h - 4) if y1 >= badge_h + 4 else y1
        
        # Vẽ hộp nền chứa số
        draw.rectangle((bx, by, bx + badge_w, by + badge_h), fill=stroke)
        # Vẽ số lên hộp
        draw.text((bx + 8, by + 3), index, fill=(255, 255, 255), font=font_badge)

    # La légende occupe son propre espace, à droite de l'image.
    panel_x = width
    draw.rectangle((panel_x, 0, panel_x + panel_width, output_height), fill=(250, 251, 253))
    draw.line((panel_x, 0, panel_x, output_height), fill=(200, 209, 219), width=2)
    margin = 18
    text_x = panel_x + margin
    draw.text((text_x, 14), "ACS | Résultats", fill=dark, font=font_title)
    draw.text((text_x, 42), "D = détection  |  T = type", fill=soft, font=font_small)
    draw.text((text_x, 61), "Orange = prédiction à vérifier", fill=soft, font=font_small)
    draw.line((text_x, 84, panel_x + panel_width - margin, 84),
              fill=(216, 222, 231), width=1)

    if not detections:
        draw.text((text_x, 108), "Aucun bonbon détecté", fill=soft, font=font_main)
        return canvas

    for position, item in enumerate(detections):
        top = header_height + row_height * position
        attention = item.get("statut") != "prediction_disponible"
        marker = orange if attention else green
        if position % 2 == 0:
            draw.rectangle((panel_x + 8, top, panel_x + panel_width - 8,
                            top + row_height - 2), fill=(240, 244, 248))
        draw.rectangle((panel_x + 12, top + 7, panel_x + 17,
                        top + row_height - 9), fill=marker)

        predicted = f"{item['type_predit']}/{item['couleur_predite']}"
        row_label = f"{item['id']:02d}  {predicted}"
        # Évite de sortir du panneau avec les variantes longues.
        available = panel_width - 46
        font = font_main
        for size in (15, 14, 13, 12, 11):
            candidate = _legend_font(size, bold=True)
            if draw.textbbox((0, 0), row_label, font=candidate)[2] <= available:
                font = candidate
                break
        draw.text((text_x + 7, top + 5), row_label, fill=dark, font=font)
        confidence_line = (f"D {item['confiance_detection']:.2f}  |  "
                           f"T {item['confiance_type']:.2f}")
        if attention:
            confidence_line += "  |  à vérifier"
        draw.text((text_x + 7, top + 27), confidence_line,
                  fill=orange if attention else soft, font=font_small)
    return canvas


def gather_images(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Format non pris en charge : {source.suffix}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Image ou dossier introuvable : {source}")
    images = sorted(p for p in source.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"Aucune image compatible dans {source}")
    return images


def process_file(image_path: Path, *, detector: Any, classifier: Any,
                 output: Path, detector_conf: float, imgsz: int,
                 device: str, iou: float, padding: float,
                 type_warning_threshold: float,
                 save_crops: bool) -> dict[str, Any]:
    image_path = image_path.resolve()
    with Image.open(image_path) as original:
        oriented = ImageOps.exif_transpose(original).convert("RGB")
        rgb = np.asarray(oriented, dtype=np.uint8)
    detections, crops = run_stages(
        rgb, detector=detector, classifier=classifier,
        detector_conf=detector_conf, imgsz=imgsz, device=device,
        iou=iou, padding=padding,
        type_warning_threshold=type_warning_threshold,
    )
    # L'empreinte évite les collisions entre plusieurs images de même nom.
    suffix = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:8]
    stem = f"{image_path.stem}_{suffix}"
    output.mkdir(parents=True, exist_ok=True)
    annotated_path = output / f"{stem}_annote.jpg"
    json_path = output / f"{stem}_predictions.json"
    draw_detections(rgb, detections).save(annotated_path, quality=95)
    if save_crops:
        crop_dir = output / "crops" / stem
        crop_dir.mkdir(parents=True, exist_ok=True)
        for item, crop in zip(detections, crops):
            crop_path = crop_dir / f"bonbon_{item['id']:03d}.png"
            Image.fromarray(crop).save(crop_path)
            item["image_crop"] = str(crop_path)
    report = {
        "image_source": str(image_path),
        "image_annotee": str(annotated_path),
        "nombre_detections": len(detections),
        "seuil_detection": detector_conf,
        "avertissement_general": (
            "Le classifieur ne connaît que les types présents lors de son "
            "entraînement ; les objets inconnus peuvent être mal classés, "
            "même avec une confiance élevée."
        ),
        "detections": detections,
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    print(f"{image_path.name}: {len(detections)} bonbon(s) — {annotated_path.name}")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACS — Pipeline YOLO + ConvNeXt-Tiny + optional color")
    parser.add_argument("--source", type=Path, required=True, help="Image ou dossier (non récursif)")
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--type-checkpoint", type=Path, default=DEFAULT_TYPE)
    parser.add_argument("--color-model", type=Path, default=DEFAULT_COLOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=str(_APP_CONFIG["device"]))
    parser.add_argument("--conf", type=float, default=float(DEFAULT_INFERENCE["conf"]))
    parser.add_argument("--iou", type=float, default=float(DEFAULT_INFERENCE["iou"]))
    parser.add_argument("--imgsz", type=int, default=int(DEFAULT_INFERENCE["image_size"]))
    parser.add_argument("--padding", type=float, default=float(DEFAULT_INFERENCE["padding"]))
    parser.add_argument("--type-warning", type=float, default=float(DEFAULT_INFERENCE["type_warning_threshold"]))
    parser.add_argument("--save-crops", action="store_true")
    args = parser.parse_args(argv)
    if not (0 < args.conf <= 1 and 0 < args.iou <= 1):
        parser.error("--conf et --iou doivent appartenir à ]0,1]")
    if not (0 <= args.padding <= 0.5 and 0 <= args.type_warning <= 1):
        parser.error("--padding ou --type-warning hors plage")
    if args.imgsz < 32:
        parser.error("--imgsz doit être >= 32")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    images = gather_images(args.source.expanduser().resolve())
    required = [args.detector, args.type_checkpoint]
    if load_config("color").get("enabled", False):
        required.append(args.color_model)
    for checkpoint in required:
        if not checkpoint.expanduser().resolve().is_file():
            raise FileNotFoundError(f"Modèle introuvable : {checkpoint}")

    # Imports tardifs : tests synthétiques possibles sans torch / ultralytics.
    from ultralytics import YOLO
    from src.pipeline import ACSPipeline

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = YOLO(str(args.detector.expanduser().resolve()))
    classifier = ACSPipeline(
        type_checkpoint=args.type_checkpoint.expanduser().resolve(),
        color_model=args.color_model.expanduser().resolve(),
        device=device,
    )
    output = args.output.expanduser().resolve()
    print(f"ACS intégration | {len(images)} image(s) | conf={args.conf} | device={device}")
    summary = []
    for image_path in images:
        report = process_file(
            image_path, detector=detector, classifier=classifier,
            output=output, detector_conf=args.conf, imgsz=args.imgsz,
            device=device, iou=args.iou, padding=args.padding,
            type_warning_threshold=args.type_warning,
            save_crops=args.save_crops,
        )
        summary.append({"image_source": report["image_source"],
                        "nombre_detections": report["nombre_detections"],
                        "image_annotee": report["image_annotee"]})
    (output / "resume.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Résultats : {output}")


if __name__ == "__main__":
    main()
