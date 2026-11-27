"""Evaluate a frozen ConvNeXt type classifier on the fixed validation/test split.

    python -m src.evaluation.classification --split val
    python -m src.evaluation.classification --split test

This module never fits or updates weights. The test split must be reserved until
all model and threshold selections have been finalized.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.classification.convnext_adapter import load_convnext_model, read_checkpoint
from src.classification.train import check_splits
from src.classification.transforms import build_transform
from src.config import catalog, load_config, path_from_root
from src.evaluation.reporting import save_report


def evaluate_split(split: str, device: str = 'auto') -> dict:
    if split not in ('val', 'test'):
        raise ValueError('Only val and held-out test are supported')
    cfg = load_config('classification')
    types, _ = catalog()
    sets = check_splits(path_from_root(cfg['dataset']['processed']), cfg)
    dataset = sets[split]
    checkpoint = path_from_root(cfg['checkpoint'])
    bundle = read_checkpoint(checkpoint)
    if bundle['classes'] != types:
        raise ValueError('ConvNeXt checkpoint classes differ from the catalog')
    dataset.transform = build_transform(bundle.get('preprocessing', cfg['preprocessing']), training=False)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    network = load_convnext_model(checkpoint, torch.device(device), types)
    network.eval()
    loader = DataLoader(dataset, batch_size=int(cfg['training']['batch_size']), shuffle=False,
                        num_workers=int(cfg['training'].get('num_workers', 0)))
    y_true, y_pred, files = [], [], [str(path) for path, _ in dataset.samples]
    with torch.inference_mode():
        for images, targets in loader:
            logits = network(images.to(device))
            y_true.extend(targets.tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    output = path_from_root(f'results/evaluation/classification/{split}')
    metrics = save_report(y_true, y_pred, types, output, split=split, files=files)
    print(f'ConvNeXt {split}: accuracy={metrics["accuracy"]:.4f}; '
          f'macro-F1={metrics["macro_f1"]:.4f}; reports={output}')
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description='ACS independent type-classification evaluator')
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args(argv)
    evaluate_split(args.split, args.device)


if __name__ == '__main__':
    main()
