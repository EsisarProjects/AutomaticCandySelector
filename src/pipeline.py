"""ACS — Pipeline tích hợp phân loại loại kẹo (ConvNeXt-Tiny) và màu sắc (KNN)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from src.color.color_classifier import (
    ABSTAIN,
    extract_features_rgb,
    predict_color,
    read_split,
)
from src.color.hsv_analysis import resize_for_analysis
from src.data.preprocessing import load_image
from src.classification.convnext_adapter import load_convnext_model

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TYPE_MODEL = ROOT / "models/candy_convnext_tiny.pth"
DEFAULT_COLOR_MODEL = ROOT / "models/color_hsv_knn_v1.joblib"


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name not in ("cpu", "cuda"):
        raise ValueError("Le device doit être auto, cpu ou cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponible : utilisez --device cpu")
    return torch.device(device_name)


class ACSPipeline:
    """Dự đoán loại kẹo và màu sắc từ 1 ảnh cắt RGB sử dụng ConvNeXt-Tiny."""

    def __init__(
        self,
        type_checkpoint: Path = DEFAULT_TYPE_MODEL,
        color_model: Path = DEFAULT_COLOR_MODEL,
        device: str = "cpu",
        root: Path = ROOT,
    ):
        self.root = Path(root)
        self.type_checkpoint = self._resolve(type_checkpoint)
        self.color_model = self._resolve(color_model)
        self.device = resolve_device(device)

        if not self.type_checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint du type introuvable : {self.type_checkpoint}")
        if not self.color_model.is_file():
            raise FileNotFoundError(f"Modèle couleur introuvable : {self.color_model}")

        # 1. TỰ ĐỘNG ĐỌC ĐỦ 6 LOẠI KẸO TỪ MODEL CONVNEXT ĐÃ TRAIN
        state_dict_raw = torch.load(self.type_checkpoint, map_location="cpu")
        if isinstance(state_dict_raw, dict) and "classes" in state_dict_raw:
            self.classes = state_dict_raw["classes"]
        else:
            self.classes = ['crocodile', 'dragibus', 'fraise', 'oeuf', 'ourson', 'schtroumpf']

        # 2. KHỚP KÍCH THƯỚC 224x224 (Chuẩn của ConvNeXt-Tiny)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Nạp mô hình ConvNeXt-Tiny thông qua Adapter
        self.type_model = load_convnext_model(
            checkpoint_path=self.type_checkpoint,
            device=self.device,
            class_names=self.classes
        )

        # Nạp mô hình màu sắc (KNN)
        self.color_bundle = joblib.load(self.color_model)
        self.variants = self.color_bundle.get("variants", {})

        # An toàn cho nhãn màu
        self.variant_labels = [
            f"{candy_type}/{color}"
            for candy_type in self.classes
            if candy_type in self.variants
            for color in self.variants[candy_type]
        ]

    def _resolve(self, path: Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self.root / path

    def predict_rgb(self, rgb: np.ndarray) -> dict:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("L'entrée doit être une image RGB uint8 H×W×3")
        
        # --- KHÂU RESIZE CHUẨN: GIỮ NGUYÊN TỈ LỆ DÀI/NGẮN (LETTERBOX) ---
        raw_img = Image.fromarray(rgb)
        target_size = 224  # Khớp 224x224 cho ConvNeXt
        
        # 1. Co ảnh theo tỉ lệ gốc
        raw_img.thumbnail((target_size, target_size), Image.Resampling.BILINEAR)
        
        # 2. Đặt vào khung vuông với nền trắng
        padded_img = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        offset = ((target_size - raw_img.width) // 2, (target_size - raw_img.height) // 2)
        padded_img.paste(raw_img, offset)
        # -------------------------------------------------------------

        inputs = self.transform(padded_img).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = self.type_model(inputs).softmax(dim=1)[0].cpu()
        predicted_index = int(probabilities.argmax().item())
        predicted_type = self.classes[predicted_index]

        # Trích xuất đặc trưng màu HSV
        features = extract_features_rgb(
            rgb,
            max_side=int(self.color_bundle.get("max_side", 200)),
            min_saturation=int(self.color_bundle.get("min_saturation", 25)),
        )

        if predicted_type in self.variants:
            color_result = predict_color(self.color_bundle, predicted_type, features)
            predicted_color = color_result.get("prediction", "")
            score = float(color_result["score"]) if (color_result.get("score") is not None and math.isfinite(color_result["score"])) else None
            source_color = color_result.get("source", "")
        else:
            predicted_color = ""
            score = None
            source_color = "auto"

        return {
            "type_predit": predicted_type,
            "confiance_type": float(probabilities[predicted_index].item()),
            "couleur_predite": predicted_color,
            "score_couleur": score,
            "source_couleur": source_color,
            "variante_predite": f"{predicted_type}/{predicted_color}" if predicted_color else predicted_type,
            "segmentation_status": features.get("status", ""),
            "hue_mean": features.get("hue_mean", 0),
            "s_median": features.get("s_median", 0),
            "v_median": features.get("v_median", 0),
            "mask_fraction": features.get("mask_fraction", 0),
        }

    def predict_image(self, path: Path) -> dict:
        path = self._resolve(path)
        if not path.is_file():
            raise FileNotFoundError(f"Photographie introuvable : {path}")
        return self.predict_rgb(np.asarray(load_image(path), dtype=np.uint8))
