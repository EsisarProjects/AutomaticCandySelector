"""Centralized ACS YAML configuration and portable project-root path resolution."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / 'configs'


def load_config(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / f'{name}.yaml'
    if not path.is_file():
        raise FileNotFoundError(f'Missing ACS configuration: {path}')
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'ACS configuration must be a YAML mapping: {path}')
    return data


def path_from_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def catalog() -> tuple[list[str], dict[str, list[str]]]:
    config = load_config('classes')
    types = config.get('types', [])
    variants = config.get('variantes', {})
    if not isinstance(types, list) or len(set(types)) != len(types) or not types:
        raise ValueError('configs/classes.yaml must define unique candy types')
    if not isinstance(variants, dict) or set(types) != set(variants):
        raise ValueError('Each configured candy type must have its variants')
    for typ, choices in variants.items():
        if not choices or len(choices) != len(set(choices)):
            raise ValueError(f'Missing or duplicate variants for {typ}')
    return types, variants
