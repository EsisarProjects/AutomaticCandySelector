"""Train the active ConvNeXt-Tiny only; split by photography session, never image shuffle."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from src.config import catalog, load_config, path_from_root
from src.classification.convnext_adapter import ARCHITECTURE, create_convnext, read_checkpoint
from src.classification.transforms import build_transform

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


class SessionDataset(Dataset):
    def __init__(self, root: Path, class_names: list[str], variants: dict, sessions: list[str], transform=None):
        self.classes = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.transform = transform
        self.samples: list[tuple[Path,int]] = []
        for name in class_names:
            for variant in variants[name]:
                for session in sessions:
                    folder = root / name / variant / session
                    if folder.is_dir():
                        for file in sorted(folder.iterdir()):
                            if file.is_file() and file.suffix.lower() in EXTENSIONS:
                                self.samples.append((file, self.class_to_idx[name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        with Image.open(filename) as image:
            converted = image.convert('RGB')
        return self.transform(converted) if self.transform else converted, label


def check_splits(processed_dir: Path, settings: dict):
    class_names, variants = catalog()
    sessions = settings['dataset']['splits']
    all_sessions = [session for names in sessions.values() for session in names]
    if set(sessions) != {'train', 'val', 'test'} or len(set(all_sessions)) != len(all_sessions):
        raise ValueError('train/val/test must exist and must not share a photography session')
    datasets = {
        split: SessionDataset(processed_dir, class_names, variants, sessions[split])
        for split in ('train', 'val', 'test')
    }
    if not datasets['train'].samples or not datasets['val'].samples:
        raise ValueError('Independent train and val images required. Prepare processed V2 images first.')
    if not datasets['test'].samples:
        print('WARNING: No independent test images available yet; no final evaluation is possible.')
    # An identical file in two splits indicates leakage. Near-duplicates must be audited manually.
    hashes = {}
    for split, ds in datasets.items():
        for file, _ in ds.samples:
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest][0] != split:
                raise ValueError(f'Identical images across {split} and {hashes[digest][0]}: '
                                 f'{file} and {hashes[digest][1]}')
            hashes[digest] = (split, file)
    return datasets


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    with torch.inference_mode():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            total_loss += float(criterion(logits, targets).item()) * len(targets)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += len(targets)
    return {'loss': total_loss / max(1, total), 'accuracy': correct / max(1, total), 'n': total}


def main(argv=None):
    parser = argparse.ArgumentParser(description='ACS ConvNeXt-Tiny training')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--evaluate-test', action='store_true', help='Evaluate held-out test after THIS run (use only once)')
    parser.add_argument('--test-only', action='store_true', help='Evaluate an EXISTING frozen checkpoint without retraining')
    args = parser.parse_args(argv)
    settings = load_config('classification')
    if settings['architecture'] != ARCHITECTURE:
        raise ValueError('Only ConvNeXt-Tiny is supported')
    training = settings['training']
    processed = path_from_root(settings['dataset']['processed'])
    splits = check_splits(processed, settings)
    for name, dataset in splits.items():
        print(f'{name}: {len(dataset)} photographs | sessions={settings["dataset"]["splits"][name]}')
    if args.check_only:
        return
    if (args.evaluate_test or args.test_only) and not len(splits['test']):
        raise ValueError('Cannot evaluate test: no independent S04 images')
    device_name = training['device']
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device(device_name)
    seed = int(training['seed'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    class_names, _ = catalog()
    xforms = settings['preprocessing']
    loaders = {}
    for name, ds in splits.items():
        ds.transform = build_transform(xforms, training=(name == 'train'))
        if len(ds):
            loaders[name] = DataLoader(ds, batch_size=int(training['batch_size']),
                                       shuffle=(name == 'train'), num_workers=int(training['num_workers']))
    destination = path_from_root(settings['checkpoint'])
    experiment_dir = path_from_root(training['experiment_dir'])
    if args.test_only:
        bundle = read_checkpoint(destination)
        if bundle['classes'] != class_names:
            raise ValueError('Frozen checkpoint classes do not match V2 catalog')
        model = create_convnext(len(class_names), pretrained=False).to(device)
        model.load_state_dict(bundle['model_state_dict'])
        model.eval()
        splits['test'].transform = build_transform(bundle.get('preprocessing', xforms), training=False)
        test_loader = DataLoader(splits['test'], batch_size=int(training['batch_size']), shuffle=False)
        final = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        (experiment_dir / 'final_test.json').write_text(json.dumps(final, indent=2), encoding='utf-8')
        print(f'FINAL TEST of frozen checkpoint {destination}: {final}')
        return
    model = create_convnext(len(class_names), pretrained=training['pretrained'] and not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training['learning_rate']),
                                  weight_decay=float(training['weight_decay']))
    destination.parent.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    unimproved = 0
    history = []
    from src.evaluation.reporting import save_history
    for epoch in range(1, int(training['epochs']) + 1):
        model.train()
        running = count = train_correct = 0
        for images, targets in loaders['train']:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            train_correct += int((logits.argmax(dim=1) == targets).sum().item())
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(targets)
            count += len(targets)
        validation = evaluate(model, loaders['val'], criterion, device)
        item = {'epoch': epoch, 'train_loss': running/max(1,count),
                'train_accuracy': train_correct/max(1,count),
                'val_loss': validation['loss'], 'val_accuracy': validation['accuracy']}
        history.append(item)
        print(item, flush=True)
        if validation['accuracy'] > best_accuracy:
            best_accuracy = validation['accuracy']
            unimproved = 0
            torch.save({
                'architecture': ARCHITECTURE,
                'classes': class_names,
                'class_to_idx': {name: i for i,name in enumerate(class_names)},
                'model_state_dict': model.state_dict(),
                'preprocessing': xforms,
                'epoch': epoch,
                'val_accuracy': best_accuracy,
            }, destination)
        else:
            unimproved += 1
        if unimproved >= int(training['patience']):
            break
    (experiment_dir / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    save_history(history, experiment_dir)
    print(f'Best ConvNeXt saved to {destination}; val accuracy={best_accuracy:.4f}')
    if args.evaluate_test:
        bundle = read_checkpoint(destination)
        model.load_state_dict(bundle['model_state_dict'])
        final = evaluate(model, loaders['test'], criterion, device)
        (experiment_dir / 'final_test.json').write_text(json.dumps(final, indent=2), encoding='utf-8')
        print(f'FINAL TEST (use once): {final}')


if __name__ == '__main__':
    main()