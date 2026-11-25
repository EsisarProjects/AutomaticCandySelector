"""Convert V2 raw images to JPEG, preserving type/variant/session folders.

Raw HEIC/JPEG/PNG files are never changed. When formats share a basename,
HEIC is preferred to avoid counting the same photograph twice.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image, ImageOps
from src.config import load_config, path_from_root, catalog

SUPPORTED = {'.heic', '.heif', '.jpg', '.jpeg', '.png', '.webp'}


def convert_source(src: Path, dst: Path):
    if src.suffix.lower() in {'.heic', '.heif'}:
        try:
            from pillow_heif import register_heif_opener
        except ImportError as exc:
            raise RuntimeError('HEIC conversion requires: pip install pillow-heif') from exc
        register_heif_opener()
    with Image.open(src) as handle:
        rgb = ImageOps.exif_transpose(handle).convert('RGB')
        dst.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(dst, 'JPEG', quality=95)


def prepare(raw: Path, processed: Path, overwrite: bool = False) -> dict:
    classes, variants = catalog()
    counts = {'converted': 0, 'skipped': 0, 'missing_sessions': []}
    for typ in classes:
        for variant in variants[typ]:
            folder = raw / typ / variant
            if not folder.is_dir():
                continue
            for session in sorted(p for p in folder.iterdir() if p.is_dir()):
                unique = {}
                # Prefer HEIC when both original HEIC and JPG conversion exist.
                priority = {'.heic': 0, '.heif': 1, '.jpg': 2, '.jpeg': 3, '.png': 4, '.webp': 5}
                for src in sorted(session.iterdir(), key=lambda p: (p.stem.lower(), priority.get(p.suffix.lower(), 99))):
                    if src.is_file() and src.suffix.lower() in SUPPORTED:
                        unique.setdefault(src.stem.lower(), src)
                for src in unique.values():
                    dst = processed / typ / variant / session.name / (src.stem + '.jpg')
                    if dst.exists() and not overwrite:
                        counts['skipped'] += 1
                        continue
                    convert_source(src, dst)
                    counts['converted'] += 1
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description='ACS V2 HEIC/JPG -> processed JPGs')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    config = load_config('classification')['dataset']
    print(prepare(path_from_root(config['raw']), path_from_root(config['processed']), args.overwrite))


if __name__ == '__main__':
    main()
