"""ACS inference: ConvNeXt-Tiny type classification + REQUIRED V2 color model.

The old MobileNet checkpoint is NOT supported or silently substituted.
"""
from __future__ import annotations
import math
from pathlib import Path
import joblib
import numpy as np
import torch
from PIL import Image, ImageOps
from src.config import catalog, load_config, path_from_root
from src.classification.convnext_adapter import load_convnext_model, read_checkpoint
from src.classification.transforms import build_transform

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TYPE_MODEL = path_from_root(load_config('classification')['checkpoint'])
DEFAULT_COLOR_MODEL = path_from_root(load_config('color')['checkpoint'])


def resolve_device(device_name: str) -> torch.device:
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_name not in ('cpu', 'cuda'):
        raise ValueError('Device must be auto, cpu or cuda')
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    return torch.device(device_name)


class ACSPipeline:
    def __init__(self, type_checkpoint=None, color_model=None, device='auto', root=ROOT):
        self.root = Path(root)
        type_settings = load_config('classification')
        color_settings = load_config('color')
        self.type_checkpoint = self._resolve(type_checkpoint or type_settings['checkpoint'])
        self.color_model = self._resolve(color_model or color_settings['checkpoint'])
        self.device = resolve_device(device)
        self.catalog_types, self.catalog_variants = catalog()
        bundle = read_checkpoint(self.type_checkpoint)
        self.classes = bundle['classes']
        if self.classes != self.catalog_types:
            raise ValueError('ConvNeXt checkpoint classes differ from configs/classes.yaml; '
                             'retrain or use a checkpoint with the full V2 catalog.')
        preprocessing = bundle.get('preprocessing', type_settings['preprocessing'])
        self.transform = build_transform(preprocessing, training=False)
        self.type_model = load_convnext_model(self.type_checkpoint, self.device, self.classes)
        if color_settings.get('enabled') is not True:
            raise ValueError('Variant classification is mandatory: configs/color.yaml must set enabled: true')
        if not self.color_model.is_file():
            raise FileNotFoundError(f'Required color classifier checkpoint missing: {self.color_model}')
        # Only load joblib from a trusted local training process.
        self.color_bundle = joblib.load(self.color_model)
        if not isinstance(self.color_bundle, dict):
            raise ValueError('Unsupported color classifier format')
        self._validate_color_model()

    def _validate_color_model(self):
        """Reject models incompatible with the complete six-type/21-variant catalog."""
        observed = self.color_bundle.get('variants', {})
        if not isinstance(observed, dict) or observed != self.catalog_variants:
            raise ValueError(f'Color checkpoint does not cover the complete {sum(map(len, self.catalog_variants.values()))}-variant catalog')
        estimators = self.color_bundle.get('models', {})
        required = {typ for typ, variants in self.catalog_variants.items() if len(variants) > 1}
        if not isinstance(estimators, dict) or not required.issubset(estimators):
            raise ValueError(f'Color checkpoint is missing trained multi-variant models: {required - set(estimators)}')

    def _resolve(self, path):
        path = Path(path).expanduser()
        return path if path.is_absolute() else self.root / path

    def predict_rgb(self, rgb: np.ndarray) -> dict:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError('Expected RGB uint8 HxWx3')
        with torch.inference_mode():
            input_image = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
            probabilities = self.type_model(input_image).softmax(dim=1)[0].cpu()
        index = int(probabilities.argmax().item())
        candy_type = self.classes[index]
        options = self.catalog_variants[candy_type]
        color = ''
        score = None
        segmentation_status = 'not_evaluated'
        source = 'unavailable'
        if len(options) == 1:
            color = options[0]
            source = 'catalog_rule'    # NOT a visual color prediction
            segmentation_status = 'not_required'
        else:
            from src.color.color_classifier import extract_features_rgb, predict_color
            features = extract_features_rgb(
                rgb,
                max_side=int(self.color_bundle.get('max_side', 800)),
                min_saturation=int(self.color_bundle.get('min_saturation', 45)),
            )
            result = predict_color(self.color_bundle, candy_type, features)
            candidate = str(result.get('prediction', '')).strip()
            color = candidate if candidate in options else ''
            candidate_score = result.get('score')
            if candidate_score is not None and math.isfinite(float(candidate_score)):
                score = float(candidate_score)
            source = str(result.get('source', 'hsv_knn'))
            segmentation_status = str(features.get('status', 'unknown'))
        return {
            'type_predit': candy_type,
            'confiance_type': float(probabilities[index].item()),
            'couleur_predite': color,
            'score_couleur': score,
            'source_couleur': source,
            'variante_predite': f'{candy_type}/{color}' if color else 'unknown',
            'variante_resolvida': bool(color),
            'segmentation_status': segmentation_status,
        }

    def predict_image(self, path):
        path = self._resolve(path)
        if not path.is_file():
            raise FileNotFoundError(f'Image not found: {path}')
        with Image.open(path) as source:
            rgb = np.asarray(ImageOps.exif_transpose(source).convert('RGB'), dtype=np.uint8)
        return self.predict_rgb(rgb)