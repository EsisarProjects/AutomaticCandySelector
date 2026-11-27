"""Independent V2 color/variant evaluation using GROUND-TRUTH candy types.

This isolates variant recognition from ConvNeXt errors. The joint evaluator
uses predicted types instead and therefore reports a different metric.

  python -m src.evaluation.color --split val
  python -m src.evaluation.color --split test
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import joblib
from src.classification.train import check_splits
from src.color.train import extract_rows
from src.color.color_classifier import predict_color, ABSTAIN
from src.config import catalog, load_config, path_from_root
from src.evaluation.variant_metrics import UNKNOWN, save_variant_report


def run(split: str):
    if split not in ('val', 'test'):
        raise ValueError('Split must be val or test')
    cfg = load_config('color')
    if cfg.get('enabled') is not True:
        raise ValueError('Color/variant classification is mandatory')
    checkpoint = path_from_root(cfg['checkpoint'])
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Fitted color checkpoint not found: {checkpoint}')
    variants = catalog()[1]
    bundle = joblib.load(checkpoint)  # Only locally produced trusted files.
    if bundle.get('variants') != variants:
        raise ValueError('Color checkpoint does not match the current catalog')
    sets = check_splits(path_from_root(load_config('classification')['dataset']['processed']), load_config('classification'))
    dataset = sets[split]
    if not dataset.samples:
        raise ValueError(f'No independent {split} images')
    settings = cfg['training']
    rows = extract_rows(dataset, max_side=int(bundle.get('max_side', settings['max_side'])),
                        min_saturation=int(bundle.get('min_saturation', settings['min_saturation'])))
    labels_true, labels_pred, files = [], [], []
    data_rows = []
    for row in rows:
        pred = predict_color(bundle, row['type'], row['features'])
        expected = f"{row['type']}/{row['variant']}"
        candidate = str(pred['prediction'])
        actual = f"{row['type']}/{candidate}" if candidate in variants[row['type']] else UNKNOWN
        labels_true.append(expected)
        labels_pred.append(actual)
        files.append(row['image'])
        data_rows.append({'image': row['image'], 'true_type': row['type'],
                          'true_variant': row['variant'], 'predicted_variant': candidate,
                          'correct': expected == actual, 'source': pred.get('source', ''),
                          'segmentation': row['features'].get('status', '')})
    output = path_from_root(f'results/evaluation/color/{split}')
    result = save_variant_report(labels_true, labels_pred, output, split=split, files=files)
    multi = [r for r in data_rows if len(variants[r['true_type']]) > 1]
    result['conditional_on_ground_truth_type'] = True
    result['multivariant_samples'] = len(multi)
    result['multivariant_accuracy'] = (sum(r['correct'] for r in multi)/len(multi)) if multi else None
    result['catalog_rule_samples'] = len(data_rows) - len(multi)
    result['checkpoint'] = str(checkpoint)
    (output/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    with (output/'predictions_detailed.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f, fieldnames=list(data_rows[0])); writer.writeheader(); writer.writerows(data_rows)
    print(f'COLOR [{split}] independent (true type provided): {result["accuracy"]:.4f}; '
          f'multi-variant accuracy={result["multivariant_accuracy"]}')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument('--split',required=True,choices=('val','test'))
    args=parser.parse_args(argv)
    run(args.split)

if __name__ == '__main__':
    main()
