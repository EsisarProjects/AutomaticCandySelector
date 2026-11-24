import sys
import os
import time
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

print("[WORKER] Dang khoi dong AI...", flush=True)

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from src.pipeline import ACSPipeline
import src.detection.integrated_pipeline as ip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[WORKER] Thiet bi dang chay: {DEVICE.upper()}", flush=True)

detector_path = ROOT / "results/detection/training/yolov8n_baseline_reproduction/weights/best.pt"
if not detector_path.is_file():
    detector_path = ip.DEFAULT_DETECTOR

type_checkpoint = ROOT / "models/candy_convnext_tiny.pth"
if not type_checkpoint.is_file():
    type_checkpoint = ip.DEFAULT_TYPE

detector = YOLO(str(detector_path))
classifier = ACSPipeline(
    type_checkpoint=type_checkpoint,
    color_model=ip.DEFAULT_COLOR,
    device=DEVICE,
)

def custom_draw_detections(rgb: np.ndarray, detections: list[dict]) -> Image.Image:
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=16)
    except:
        try:
            font = ImageFont.truetype("arialbd.ttf", size=16)
        except:
            font = ImageFont.load_default()

    green = (13, 157, 91)
    orange = (241, 142, 28)

    for item in detections:
        x1, y1, x2, y2 = (int(v) for v in item["bbox_xyxy"])
        attention = item.get("statut") != "prediction_disponible"
        stroke = orange if attention else green

        draw.rectangle((x1, y1, x2, y2), outline=stroke, width=3)

        type_name = item.get("type_predit", "")
        color_name = item.get("couleur_predite", "")
        conf = int(item.get("confiance_type", 0) * 100)
        label = f"{type_name} {color_name} {conf}%".strip()

        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        badge_w = text_w + 14
        badge_h = text_h + 8

        bx = max(0, min(x1, rgb.shape[1] - badge_w))
        by = (y1 - badge_h - 2) if y1 >= badge_h + 2 else y1 + 2

        draw.rectangle((bx, by, bx + badge_w, by + badge_h), fill=stroke)
        draw.text((bx + 7, by + 3), label, fill=(255, 255, 255), font=font)

    return canvas

ip.draw_detections = custom_draw_detections

print("READY", flush=True)

while True:
    line = sys.stdin.readline()
    if not line:
        break
    image_path_str = line.strip()
    if not image_path_str:
        continue

    image_path = Path(image_path_str)
    if not image_path.is_file():
        print(f"Loi: Khong tim thay file {image_path_str}", flush=True)
        print("DONE", flush=True)
        continue

    try:
        t0 = time.time()
        report = ip.process_file(
            image_path=image_path,
            detector=detector,
            classifier=classifier,
            output=ip.DEFAULT_OUTPUT,
            detector_conf=0.99,   # Độ nhạy chuẩn
            imgsz=640,
            device=DEVICE,
            iou=0.0001,             # Chống chém đôi viên kẹo
            padding=0.08,         # Lấy trọn vẹn cả con kẹo
            type_warning_threshold=0.60,
            save_crops=False,
        )
        t_total = time.time() - t0
        print(f"⚡ Nhan dien thanh cong trong: {t_total:.2f} giay", flush=True)
        print(f"RESULT:{report['image_annotee']}", flush=True)
    except Exception as e:
        print(f"Loi xu ly: {e}", flush=True)
    
    print("DONE", flush=True)
