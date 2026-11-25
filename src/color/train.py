"""Optional V2 HSV/KNN color training. Uses same session-level splits as ConvNeXt.

This is an experimental baseline for COLOR ONLY, not the retired V1 classifier.
Validate carefully: white gummies and Schtroumpf bonnet colors may be challenging.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
from PIL import Image, ImageOps
from src.config import catalog, load_config, path_from_root
from src.classification.train import check_splits
from src.color.color_classifier import extract_features_rgb, fit_color_models, predict_color


def extract_rows(dataset, max_side: int, min_saturation: int):
    rows=[]
    for file, label in dataset.samples:
        typ = dataset.classes[label]
        variant = file.parent.parent.name
        with Image.open(file) as source:
            rgb = np.asarray(ImageOps.exif_transpose(source).convert('RGB'), dtype=np.uint8)
        features = extract_features_rgb(rgb, max_side=max_side,
                                         min_saturation=min_saturation)
        rows.append({'image': str(file), 'type': typ, 'variant': variant, 'features': features})
    return rows


def evaluate(rows, bundle):
    report = defaultdict(lambda: {'n': 0, 'correct': 0, 'abstained': 0})
    for row in rows:
        result = predict_color(bundle, row['type'], row['features'])
        item = report[row['type']]
        item['n'] += 1
        item['correct'] += (result['prediction'] == row['variant'])
        item['abstained'] += (result['prediction'] == 'indetermine')
    return {k: {**v, 'accuracy': v['correct']/v['n']} for k,v in report.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train optional V2 color HSV/KNN')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--evaluate-test', action='store_true', help='Final held-out test after training')
    parser.add_argument('--test-only', action='store_true', help='Final held-out test of existing model')
    args = parser.parse_args(argv)
    cls = load_config('classification')
    cfg = load_config('color')
    processed = path_from_root(cls['dataset']['processed'])
    splits = check_splits(processed, cls)
    for split, ds in splits.items():
        print(f'{split}: {len(ds)}')
    if args.check_only:
        return
    train = cfg['training']
    options = {'max_side': int(train['max_side']), 'min_saturation': int(train['min_saturation'])}
    checkpoint = path_from_root(cfg['checkpoint'])
    output = path_from_root(train['experiment_dir'])
    if args.test_only:
        if not splits['test'].samples:
            raise ValueError('Independent test images required')
        bundle = joblib.load(checkpoint)  # Trust only locally created model files.
        final = evaluate(extract_rows(splits['test'], **options), bundle)
        output.mkdir(parents=True, exist_ok=True)
        (output/'final_test.json').write_text(json.dumps(final, indent=2), encoding='utf-8')
        print('FINAL TEST of frozen color model:', final)
        return
    train_rows = extract_rows(splits['train'], **options)
    _, variants = catalog()
    bundle = fit_color_models(train_rows, variants, n_neighbors=int(train['n_neighbors']))
    bundle.update(options)
    val_rows = extract_rows(splits['val'], **options)
    validation = evaluate(val_rows, bundle)
    print('Validation by type:', validation)
    output.mkdir(parents=True, exist_ok=True)
    (output/'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, checkpoint)
    print(f'Color model written to {checkpoint}; enable only after reviewing validation.')
    if args.evaluate_test:
        if not splits['test'].samples:
            raise ValueError('Independent test images required')
        final = evaluate(extract_rows(splits['test'], **options), bundle)
        (output/'final_test.json').write_text(json.dumps(final, indent=2), encoding='utf-8')
        print('FINAL TEST:', final)


if __name__ == '__main__':
    main()
