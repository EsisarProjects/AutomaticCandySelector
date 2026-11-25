"""Per-type HSV/KNN color prediction; trained only on independently split V2 images."""
from __future__ import annotations
import math
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.color.hsv_analysis import (resize_for_analysis, segment_colored_candy,
                                    compute_hsv_features)

ABSTAIN = 'indetermine'
FEATURE_NAMES = ('hue_sin', 'hue_cos', 'saturation', 'value')


def extract_features_rgb(rgb: np.ndarray, *, max_side: int = 800,
                         min_saturation: int = 45) -> dict:
    resized = resize_for_analysis(rgb, max_side=max_side)
    mask = segment_colored_candy(resized, min_saturation=min_saturation)
    return compute_hsv_features(resized, mask)


def feature_vector(features: dict) -> np.ndarray | None:
    if features.get('status') != 'detecte':
        return None
    values = np.array([features.get('hue_mean', np.nan),
                       features.get('s_median', np.nan),
                       features.get('v_median', np.nan)], dtype=np.float64)
    if not np.isfinite(values).all():
        return None
    h,s,v = values
    if not (0 <= h < 180 and 0 <= s <= 255 and 0 <= v <= 255):
        return None
    angle = 2 * math.pi * h / 180.0
    return np.array([math.sin(angle), math.cos(angle), s/255.0, v/255.0])


def fit_color_models(rows: list[dict], variants: dict[str, list[str]],
                     n_neighbors: int = 3) -> dict:
    if n_neighbors < 1:
        raise ValueError('n_neighbors must be positive')
    models = {}
    for candy_type, colors in variants.items():
        if len(colors) == 1:
            continue
        group = [row for row in rows if row['type'] == candy_type]
        vectors, labels = [], []
        for row in group:
            v = feature_vector(row['features'])
            if v is not None:
                vectors.append(v)
                labels.append(row['variant'])
        missing = set(colors) - set(labels)
        if missing:
            raise ValueError(f'Missing usable color training examples for {candy_type}: {missing}')
        estimator = make_pipeline(StandardScaler(),
                  KNeighborsClassifier(n_neighbors=min(n_neighbors, len(vectors)),
                                       weights='distance'))
        estimator.fit(np.stack(vectors), labels)
        models[candy_type] = estimator
    return {'format_version': 2, 'method': 'HSV KNN per candy type',
            'variants': variants, 'models': models,
            'feature_names': FEATURE_NAMES, 'n_neighbors': n_neighbors}


def predict_color(bundle: dict, candy_type: str, features: dict) -> dict:
    variants = bundle['variants']
    if candy_type not in variants:
        raise ValueError(f'Unknown candy type: {candy_type}')
    choices = variants[candy_type]
    if len(choices) == 1:
        return {'prediction': choices[0], 'score': None, 'source': 'catalog_rule'}
    vector = feature_vector(features)
    if vector is None:
        return {'prediction': ABSTAIN, 'score': None, 'source': 'segmentation_unreliable'}
    estimator = bundle['models'][candy_type]
    proba = estimator.predict_proba(vector.reshape(1, -1))[0]
    idx = int(np.argmax(proba))
    return {'prediction': str(estimator.classes_[idx]),
            'score': float(proba[idx]), 'source': 'knn_hsv'}
