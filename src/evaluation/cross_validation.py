"""Session-grouped cross-validation for ConvNeXt on DEVELOPMENT images only.

    python -m src.evaluation.cross_validation

S04/test is NEVER used for model selection or in any cross-validation fold.
Each S01/S02/S03 session becomes one validation fold; all remaining
DEVELOPMENT sessions train the fold. Must use independent physical candies
and photography conditions across sessions to prevent leakage.

The final deployable checkpoint must be trained/selected separately with
`python -m src.classification.train` and evaluated only once on S04.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import random
import statistics
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.classification.convnext_adapter import create_convnext
from src.classification.train import SessionDataset, check_splits
from src.classification.transforms import build_transform
from src.config import catalog, load_config, path_from_root
from src.evaluation.reporting import save_history, save_report


def run(*, no_pretrained=False, device_override=None):
    cfg = load_config('classification')
    # Check split configuration and exact duplicates across splits first.
    check_splits(path_from_root(cfg['dataset']['processed']), cfg)
    dev = cfg['dataset']['splits']
    sessions = list(dict.fromkeys(dev['train'] + dev['val']))
    if len(sessions) < 3:
        raise ValueError('At least 3 independent development sessions required for session-fold CV')
    if set(sessions) & set(dev['test']):
        raise ValueError('Cross-validation is not allowed to use test sessions')
    names, variants = catalog()
    training = cfg['training']
    data_root = path_from_root(cfg['dataset']['processed'])
    out = path_from_root('results/evaluation/classification/cross_validation')
    device_name = device_override or training['device']
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    seed = int(training['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    xforms = cfg['preprocessing']
    results = []
    for index, held_out in enumerate(sessions, start=1):
        training_sessions = [s for s in sessions if s != held_out]
        train_ds = SessionDataset(data_root, names, variants, training_sessions,
                                  build_transform(xforms, training=True))
        val_ds = SessionDataset(data_root, names, variants, [held_out],
                                build_transform(xforms, training=False))
        for label, ds in [('train', train_ds), ('val', val_ds)]:
            seen = {index for _, index in ds.samples}
            if len(seen) != len(names):
                missing = [name for i, name in enumerate(names) if i not in seen]
                raise ValueError(f'Fold {held_out} missing classes in {label}: {missing}')
        loader_train = DataLoader(train_ds, batch_size=int(training['batch_size']),
                                  shuffle=True, num_workers=int(training['num_workers']))
        loader_val = DataLoader(val_ds, batch_size=int(training['batch_size']),
                                shuffle=False, num_workers=int(training['num_workers']))
        model = create_convnext(len(names), pretrained=training['pretrained'] and not no_pretrained).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(training['learning_rate']),
                                      weight_decay=float(training['weight_decay']))
        criterion = nn.CrossEntropyLoss()
        best_accuracy = -1.
        best_state = None
        stale = 0
        history = []
        fold_dir = out / f'fold_{index:02d}_val_{held_out}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, int(training['epochs'])+1):
            model.train()
            train_loss = correct = total = 0
            for images, targets in loader_train:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())*len(targets)
                correct += int((logits.argmax(1) == targets).sum().item())
                total += len(targets)
            model.eval()
            val_loss = val_correct = val_total = 0
            with torch.inference_mode():
                for images, targets in loader_val:
                    images, targets = images.to(device), targets.to(device)
                    logits = model(images)
                    val_loss += float(criterion(logits, targets).item())*len(targets)
                    val_correct += int((logits.argmax(1) == targets).sum().item())
                    val_total += len(targets)
            score = val_correct/val_total
            history.append({'epoch': epoch, 'train_loss': train_loss/total,
                            'train_accuracy': correct/total,
                            'val_loss': val_loss/val_total, 'val_accuracy': score})
            if score > best_accuracy:
                best_accuracy = score
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= int(training['patience']):
                break
        save_history(history, fold_dir)
        model.load_state_dict(best_state)
        model.eval()
        truth, prediction = [], []
        with torch.inference_mode():
            for images, targets in loader_val:
                logits = model(images.to(device))
                truth.extend(targets.tolist())
                prediction.extend(logits.argmax(1).cpu().tolist())
        report = save_report(truth, prediction, names, fold_dir, split=f'fold_{held_out}',
                             files=[p for p, _ in val_ds.samples])
        report.update({'held_out_session': held_out, 'train_sessions': training_sessions,
                       'best_epoch': history[[x['val_accuracy'] for x in history].index(best_accuracy)]['epoch']})
        (fold_dir/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        results.append({'fold': index, 'val_session': held_out, 'n_val': len(truth),
                        'accuracy': report['accuracy'], 'macro_f1': report['macro_f1']})
        print(f'Fold {index}/{len(sessions)} validation={held_out} accuracy={report["accuracy"]:.4f}')
    summary = {'folds': results, 'test_data_used': False,
               'accuracy_mean': statistics.mean(x['accuracy'] for x in results),
               'accuracy_std': statistics.stdev(x['accuracy'] for x in results),
               'macro_f1_mean': statistics.mean(x['macro_f1'] for x in results),
               'macro_f1_std': statistics.stdev(x['macro_f1'] for x in results)}
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (out/'fold_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print('Cross-validation summary (NO TEST):', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--device', default=None)
    args = parser.parse_args(argv)
    run(no_pretrained=args.no_pretrained, device_override=args.device)


if __name__ == '__main__':
    main()
