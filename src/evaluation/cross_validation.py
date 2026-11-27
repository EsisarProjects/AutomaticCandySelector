"""ConvNeXt type-classification cross-validation on development data only.

Uses processed/<type>/<variant>/{train,val} and never trains/evaluates on test.
By default, exact duplicate files are grouped using SHA256; this does NOT
prevent leakage from different photos of the same physical candy or session.
For rigorous group-aware CV, supply --groups-csv with columns file,group_id.
All development photographs must be represented in that CSV.

Examples:
    python -m src.evaluation.cross_validation --check-only
    python -m src.evaluation.cross_validation --folds 3
    python -m src.evaluation.cross_validation --folds 3 --groups-csv capture_groups.csv
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader

from src.classification.convnext_adapter import create_convnext
from src.classification.train import SessionDataset, check_splits
from src.classification.transforms import build_transform
from src.config import ROOT, catalog, load_config, path_from_root
from src.evaluation.reporting import save_history, save_report


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_groups_csv(filename: Path) -> dict[Path, str]:
    """Map absolute file paths to independent capture identifiers."""
    mapping: dict[Path, str] = {}
    with filename.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {'file', 'group_id'}.issubset(reader.fieldnames):
            raise ValueError('Groups CSV requires columns: file,group_id')
        for row in reader:
            if not row['file'] or not row['group_id']:
                raise ValueError('Groups CSV has blank file or group_id')
            path = Path(row['file']).expanduser()
            if not path.is_absolute():
                path = ROOT / path
            path = path.resolve()
            if path in mapping and mapping[path] != row['group_id']:
                raise ValueError(f'Conflicting groups for {path}')
            mapping[path] = row['group_id']
    return mapping


def make_folds(samples: list[tuple[Path, int]], *, folds: int, seed: int,
               names: list[str], group_map: dict[Path, str] | None = None):
    """Return list of (train_indices, validation_indices) using ONLY supplied samples.

    Identical image bytes cannot appear across folds. If `group_map` is given,
    group IDs may span candy types and original train/val directories.
    """
    if folds < 2:
        raise ValueError('At least two folds required')
    if not samples:
        raise ValueError('No development images found')
    labels = np.asarray([label for _, label in samples], dtype=np.int64)
    counts = Counter(labels.tolist())
    if set(counts) != set(range(len(names))):
        raise ValueError('Every candy type must appear in development data')
    if min(counts.values()) < folds:
        raise ValueError(f'Not enough photographs per type for {folds} folds')

    groups: list[str] = []
    hashes: dict[str, str] = {}
    for file, _ in samples:
        file = file.resolve()
        digest = _hash_file(file)
        if group_map is None:
            group = f'hash:{digest}'
        else:
            if file not in group_map:
                raise ValueError(f'Missing group_id for development image: {file}')
            group = f'capture:{group_map[file]}'
        prior = hashes.setdefault(digest, group)
        if prior != group:
            raise ValueError(f'Identical images assigned to different groups: {file}')
        groups.append(group)

    group_counts = defaultdict(set)
    for label, group in zip(labels, groups):
        group_counts[int(label)].add(group)
    for label, distinct in group_counts.items():
        if len(distinct) < folds:
            raise ValueError(f'Class {names[label]} has only {len(distinct)} independent groups; '
                             f'{folds} folds require at least {folds}')

    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    planned = []
    for index, (train_idx, val_idx) in enumerate(
            splitter.split(np.zeros(len(samples)), labels, np.asarray(groups)), start=1):
        train_labels = set(labels[train_idx].tolist())
        val_labels = set(labels[val_idx].tolist())
        expected = set(range(len(names)))
        if train_labels != expected or val_labels != expected:
            missing = [names[n] for n in sorted(expected - val_labels)]
            raise ValueError(f'Fold {index} does not contain all candy types in validation: '
                             f'{missing}. Use more independent groups or fewer folds.')
        train_groups = {groups[int(i)] for i in train_idx}
        val_groups = {groups[int(i)] for i in val_idx}
        if train_groups & val_groups:
            raise RuntimeError('Group leakage detected in fold plan')
        planned.append((train_idx.tolist(), val_idx.tolist()))
    return planned


def run(*, folds: int = 3, check_only: bool = False, no_pretrained: bool = False,
        device_override: str | None = None, group_csv: str | None = None,
        epoch_override: int | None = None):
    cfg = load_config('classification')
    processed = path_from_root(cfg['dataset']['processed'])
    # Validate the configured train/val/test structure and check exact duplicates.
    all_sets = check_splits(processed, cfg)
    names, variants = catalog()
    dev_names = [*cfg['dataset']['splits']['train'], *cfg['dataset']['splits']['val']]
    if set(dev_names) & set(cfg['dataset']['splits']['test']):
        raise ValueError('The test split must never enter cross-validation')
    # Keep precisely the same sorted sample ordering for training/evaluation transforms.
    samples = SessionDataset(processed, names, variants, dev_names).samples
    if not samples:
        raise ValueError('No train/val images found')
    settings = cfg['training']
    seed = int(settings['seed'])
    groups = _read_groups_csv(Path(group_csv).expanduser().resolve()) if group_csv else None
    plan = make_folds(samples, folds=folds, seed=seed, names=names, group_map=groups)
    print(f'ConvNeXt {folds}-fold development cross-validation | '
          f'{len(samples)} development images | TEST EXCLUDED', flush=True)
    if group_csv is None:
        print('WARNING: Grouping identical files only. Different photos of the same '
              'candy/session may leak across folds. Use --groups-csv for grouped CV.', flush=True)
    for index, (train_idx, val_idx) in enumerate(plan, 1):
        print(f'Fold {index}: train={len(train_idx)} val={len(val_idx)}', flush=True)
    if check_only:
        return {'folds': folds, 'development_images': len(samples),
                'test_used_for_training_or_scoring': False, 'planned': True}
    if epoch_override is not None and epoch_override < 1:
        raise ValueError('--epochs must be >= 1')

    device_name = device_override or settings['device']
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device(device_name)
    out = path_from_root('results/evaluation/classification/cross_validation')
    out.mkdir(parents=True, exist_ok=True)
    n_epochs = int(epoch_override or settings['epochs'])
    patience = int(settings['patience'])
    results = []

    for fold_index, (train_idx, val_idx) in enumerate(plan, 1):
        # Independent initial weights and optimizer for every fold.
        random.seed(seed + fold_index)
        np.random.seed(seed + fold_index)
        torch.manual_seed(seed + fold_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + fold_index)
        train_dataset = SessionDataset(
            processed, names, variants, dev_names,
            transform=build_transform(cfg['preprocessing'], training=True))
        val_dataset = SessionDataset(
            processed, names, variants, dev_names,
            transform=build_transform(cfg['preprocessing'], training=False))
        if train_dataset.samples != samples or val_dataset.samples != samples:
            raise RuntimeError('Dataset changed while creating cross-validation folds')
        loader_train = DataLoader(torch.utils.data.Subset(train_dataset, train_idx),
                                  batch_size=int(settings['batch_size']), shuffle=True,
                                  num_workers=int(settings['num_workers']))
        loader_val = DataLoader(torch.utils.data.Subset(val_dataset, val_idx),
                                batch_size=int(settings['batch_size']), shuffle=False,
                                num_workers=int(settings['num_workers']))
        model = create_convnext(len(names),
                                pretrained=bool(settings['pretrained']) and not no_pretrained).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=float(settings['learning_rate']),
                                      weight_decay=float(settings['weight_decay']))
        criterion = nn.CrossEntropyLoss()
        best_score, best_epoch, best_state = -1.0, 0, None
        stale, history = 0, []
        fold_dir = out / f'fold_{fold_index:02d}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, n_epochs + 1):
            model.train()
            train_loss = train_correct = train_count = 0
            for images, targets in loader_train:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item()) * len(targets)
                train_correct += int((logits.argmax(1) == targets).sum().item())
                train_count += len(targets)
            model.eval()
            val_loss = val_correct = val_count = 0
            with torch.inference_mode():
                for images, targets in loader_val:
                    images, targets = images.to(device), targets.to(device)
                    logits = model(images)
                    val_loss += float(criterion(logits, targets).item()) * len(targets)
                    val_correct += int((logits.argmax(1) == targets).sum().item())
                    val_count += len(targets)
            score = val_correct / max(1, val_count)
            history.append({'epoch': epoch, 'train_loss': train_loss / max(1, train_count),
                            'train_accuracy': train_correct / max(1, train_count),
                            'val_loss': val_loss / max(1, val_count), 'val_accuracy': score})
            if score > best_score:
                best_score, best_epoch = score, epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        save_history(history, fold_dir)
        model.load_state_dict(best_state)
        model.eval()
        actual, predicted = [], []
        with torch.inference_mode():
            for images, targets in loader_val:
                predictions = model(images.to(device)).argmax(1)
                actual.extend(targets.tolist())
                predicted.extend(predictions.cpu().tolist())
        files = [samples[i][0] for i in val_idx]
        report = save_report(actual, predicted, names, fold_dir,
                             split=f'fold_{fold_index:02d}', files=files)
        row = {'fold': fold_index, 'n_train': len(train_idx), 'n_val': len(val_idx),
               'best_epoch': best_epoch, 'accuracy': report['accuracy'],
               'macro_f1': report['macro_f1']}
        results.append(row)
        print(f'Fold {fold_index}/{folds}: accuracy={row["accuracy"]:.4f}, '
              f'macro-F1={row["macro_f1"]:.4f}', flush=True)

    summary = {
        'mode': 'capture-grouped' if group_csv else 'exact-duplicate-grouped',
        'folds': results,
        'test_used_for_training_or_scoring': False,
        'accuracy_mean': statistics.mean(row['accuracy'] for row in results),
        'accuracy_std': statistics.stdev(row['accuracy'] for row in results),
        'macro_f1_mean': statistics.mean(row['macro_f1'] for row in results),
        'macro_f1_std': statistics.stdev(row['macro_f1'] for row in results),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (out / 'fold_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f'Reports: {out}', flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description='ACS ConvNeXt development-only k-fold CV')
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default=None)
    parser.add_argument('--groups-csv', default=None,
                        help='CSV with file,group_id; independent captures stay together')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override configured epochs for CV only')
    args = parser.parse_args(argv)
    run(folds=args.folds, check_only=args.check_only,
        no_pretrained=args.no_pretrained, device_override=args.device,
        group_csv=args.groups_csv, epoch_override=args.epochs)


if __name__ == '__main__':
    main()
