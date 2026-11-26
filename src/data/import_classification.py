"""Import a manually curated ACS dataset and create reproducible train/val/test splits.

Input (outside the project):
    Classification/{Train_Val,Test}/<type>/<variant>/*.{jpg,jpeg,png}

Output (inside the project):
    datasets/v2/classification/processed/<type>/<variant>/{train,val,test}/*

The held-out Test folder is never randomized into train or validation.
Exact duplicate files are grouped during the Train_Val split, and exact duplicates
across Test and development are refused. Other near-duplicate photographs can
still leak; curate independent photography sessions for trustworthy evaluation.

Run without --apply to preview. Applying only COPIES files, never moves/deletes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from src.config import catalog, load_config, path_from_root

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def folder_by_name(parent: Path, choices: tuple[str, ...]) -> Path:
    if not parent.is_dir():
        raise FileNotFoundError(f'Dataset root does not exist: {parent}')
    dirs = {p.name.lower(): p for p in parent.iterdir() if p.is_dir()}
    for choice in choices:
        found = dirs.get(choice.lower())
        if found:
            return found
    raise FileNotFoundError(f'Expected a directory named {choices} under {parent}')


def image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f'Missing variant: {folder}')
    return sorted((p for p in folder.rglob('*')
                   if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS),
                  key=lambda p: str(p).lower())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_plan(source: Path, output: Path, val_fraction: float, seed: int,
              class_names: list[str], variants: dict[str, list[str]]):
    """Return planned file copies and stratified counts without changing disk."""
    if not 0 < val_fraction < 0.5:
        raise ValueError('Use a validation fraction greater than 0 and below 0.5')
    dev_root = folder_by_name(source, ('Train_Val', 'TrainVal', 'Train', 'train_val'))
    test_root = folder_by_name(source, ('Test',))
    rows = []
    counts = Counter()
    # Track duplicates across ALL types, variants and splits, including Test.
    seen_hash: dict[str, tuple[str, str, str, Path]] = {}
    for typ in class_names:
        for variant in variants[typ]:
            development = image_files(dev_root / typ / variant)
            test = image_files(test_root / typ / variant)
            if len(development) < 2:
                raise ValueError(f'{typ}/{variant}: need >= 2 development images; found {len(development)}')
            if not test:
                raise ValueError(f'{typ}/{variant}: Test has no images')

            groups = defaultdict(list)
            for image in development:
                groups[sha256(image)].append(image)
            if len(groups) < 2:
                raise ValueError(f'{typ}/{variant}: need >= 2 distinct development photographs')

            # Deterministic for fixed seed and input, regardless of Python hash salt.
            rng = random.Random(f'{seed}:{typ}:{variant}')
            distinct_groups = list(groups.values())
            rng.shuffle(distinct_groups)
            desired = max(1, round(len(development) * val_fraction))
            val_groups, train_groups = [], []
            val_count = 0
            for index, group in enumerate(distinct_groups):
                remaining = len(distinct_groups) - index - 1
                if val_count < desired and remaining >= 1:
                    val_groups.append(group)
                    val_count += len(group)
                else:
                    train_groups.append(group)

            assignments = {
                'train': [p for group in train_groups for p in group],
                'val': [p for group in val_groups for p in group],
                'test': test,
            }
            if not assignments['train'] or not assignments['val']:
                raise ValueError(f'{typ}/{variant}: unable to create nonempty train and val')
            for split, images in assignments.items():
                for idx, original in enumerate(sorted(images, key=lambda p: str(p).lower()), start=1):
                    digest = sha256(original)
                    prior = seen_hash.get(digest)
                    if prior:
                        prior_split, prior_type, prior_variant, prior_path = prior
                        if (prior_type, prior_variant) != (typ, variant):
                            raise ValueError('Identical image assigned different labels: '
                                             f'{prior_path} and {original}')
                        if prior_split != split:
                            raise ValueError(f'Exact duplicate would leak between {prior_split}/{split}: '
                                             f'{prior_path} and {original}')
                    else:
                        seen_hash[digest] = (split, typ, variant, original)
                    dest_name = f'{typ}_{variant}_{split}_{idx:04d}{original.suffix.lower()}'
                    target = output / typ / variant / split / dest_name
                    rows.append({
                        'split': split, 'type': typ, 'variant': variant,
                        'original_path': str(original), 'new_path': str(target),
                        'sha256': digest,
                    })
                counts[split] += len(images)
            print(f'{typ:12} {variant:13} | train={len(assignments["train"]):4} '
                  f'val={len(assignments["val"]):4} test={len(test):4}')
    return rows, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description='ACS Classification: Train_Val shuffle/split + isolated Test')
    parser.add_argument('--source', type=Path, required=True,
                        help='External directory containing Train_Val/ and Test/')
    parser.add_argument('--val-fraction', type=float, default=0.20,
                        help='Proportion of Train_Val used as val; default 0.20')
    parser.add_argument('--seed', type=int, default=42, help='Deterministic shuffle seed')
    parser.add_argument('--apply', action='store_true', help='Copy images and write manifest')
    args = parser.parse_args(argv)
    source = args.source.expanduser().resolve()
    dataset_cfg = load_config('classification')['dataset']
    output = path_from_root(dataset_cfg['processed'])
    manifest = path_from_root('datasets/v2/metadata/classification_split_manifest.csv')
    class_names, variants = catalog()

    if source == output or output in source.parents or source in output.parents:
        parser.error('Source and destination must not contain each other')
    if output.exists() and any(p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
                               for p in output.rglob('*')):
        parser.error(f'Output already contains images: {output}. '
                     'Archive/rename the previous generated dataset before importing.')
    if manifest.exists():
        parser.error(f'Existing manifest would be overwritten: {manifest}')

    rows, counts = make_plan(source, output, args.val_fraction, args.seed, class_names, variants)
    print('\nTOTAL:', ', '.join(f'{split}={counts[split]}' for split in ('train', 'val', 'test')))
    print('Destination:', output)
    print('Test images remain untouched. Files are COPIED, not moved.')
    if not args.apply:
        print('DRY RUN. Add --apply after reviewing the counts.')
        return

    created = []
    try:
        for row in rows:
            target = Path(row['new_path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(f'Will not overwrite: {target}')
            shutil.copy2(row['original_path'], target)
            created.append(target)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open('x', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    except Exception:
        # Remove only copied destination files from THIS invocation; do not touch originals.
        for copied in reversed(created):
            copied.unlink(missing_ok=True)
        raise
    print(f'Copied {len(rows)} images. Manifest: {manifest}')


if __name__ == '__main__':
    main()
