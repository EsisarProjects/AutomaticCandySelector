"""Smoke tests for the new single-model/config/worker contract (no real checkpoints)."""
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from src.config import catalog, load_config, path_from_root
from src.classification.train import check_splits
from src.classification.transforms import build_transform
from src.detection.integrated_pipeline import DEFAULT_DETECTOR, DEFAULT_TYPE, DEFAULT_OUTPUT


def test_catalog_covers_21_variants():
    types, variants = catalog()
    assert types == ['crocodile', 'dragibus', 'fraise', 'oeuf', 'ourson', 'schtroumpf']
    assert sum(map(len, variants.values())) == 21
    assert variants['ourson'] == ['blanc', 'jaune', 'orange', 'rouge', 'vert']


def test_all_models_deploy_under_models_directory():
    classification = load_config('classification')
    detection = load_config('detection')
    color = load_config('color')
    for cfg in (classification, detection, color):
        assert cfg['checkpoint'].startswith('models/')
    assert DEFAULT_TYPE == path_from_root(classification['checkpoint'])
    assert DEFAULT_DETECTOR == path_from_root(detection['checkpoint'])
    assert DEFAULT_OUTPUT == path_from_root(load_config('application')['output_dir'])


def test_geometric_preprocessing_is_deterministic_for_inference():
    config = load_config('classification')['preprocessing']
    transform = build_transform(config, training=False)
    img = Image.new('RGB', (180, 100), (255, 0, 0))
    a, b = transform(img), transform(img)
    assert tuple(a.shape) == (3, 224, 224)
    assert torch.allclose(a, b)


def test_fixed_splits_and_duplicate_guard(tmp_path):
    cls = load_config('classification')
    types, variants = catalog()
    root = tmp_path / 'processed'
    # The fixture covers all catalog variants, as the real training check must.
    for typ in types:
        for variant in variants[typ]:
            for split, color in [('train', (255, 0, 0)),
                                 ('val', (0, 255, 0)), ('test', (0, 0, 255))]:
                folder = root / typ / variant / split
                folder.mkdir(parents=True)
                Image.new('RGB', (20, 20), color).save(folder / 'img.jpg')
    # Intentionally identical bytes in two splits: must reject leakage.
    Image.new('RGB', (20, 20), (255, 0, 0)).save(root / 'crocodile' / 'bleu' / 'val' / 'img.jpg')
    with pytest.raises(ValueError, match='Identical images across'):
        check_splits(root, cls)
    # Give every image distinct pixels to isolate the expected pass case.
    for ti, typ in enumerate(types):
        for vi, variant in enumerate(variants[typ]):
            for si, split in enumerate(('train', 'val', 'test')):
                rgb = ((ti * 37 + vi * 3 + si) % 256,
                       (vi * 45 + ti + si * 19) % 256,
                       (si * 71 + ti * 13 + vi) % 256)
                Image.new('RGB', (20, 20), rgb).save(root / typ / variant / split / 'img.jpg')
    counts = check_splits(root, cls)
    assert {name: len(data) for name, data in counts.items()} == {
        'train': 21, 'val': 21, 'test': 21,
    }
    (root / 'ourson' / 'rouge' / 'val' / 'img.jpg').unlink()
    with pytest.raises(ValueError, match='is missing 1 catalog variants'):
        check_splits(root, cls)


def test_missing_required_color_checkpoint_fails(monkeypatch, tmp_path):
    from src import pipeline
    types, _ = catalog()
    monkeypatch.setattr(pipeline, 'read_checkpoint', lambda _: {'classes': types})
    monkeypatch.setattr(pipeline, 'load_convnext_model', lambda *a, **k: object())
    with pytest.raises(FileNotFoundError, match='Required color classifier checkpoint'):
        pipeline.ACSPipeline(device='cpu', root=tmp_path)


def test_required_color_pipeline_can_predict_single_variant(monkeypatch, tmp_path):
    from src import pipeline
    types, variants = catalog()
    class DummyModel:
        def __call__(self, inputs):
            logits = torch.zeros((1, len(types)))
            logits[0, types.index('fraise')] = 10
            return logits
    monkeypatch.setattr(pipeline, 'read_checkpoint', lambda _: {'classes': types})
    monkeypatch.setattr(pipeline, 'load_convnext_model', lambda *a, **k: DummyModel())
    color_model = tmp_path/'color.joblib'
    color_model.touch()
    monkeypatch.setattr(pipeline.joblib, 'load', lambda _: {
        'variants': variants, 'models': {
            name: object() for name, allowed in variants.items() if len(allowed) > 1
        },
    })
    svc = pipeline.ACSPipeline(device='cpu', root=tmp_path, color_model=color_model)
    result = svc.predict_rgb(np.full((20, 30, 3), 255, dtype=np.uint8))
    assert result['type_predit'] == 'fraise'
    assert result['couleur_predite'] == 'rouge'
    assert result['variante_predite'] == 'fraise/rouge'
    assert result['variante_resolvida'] is True
