"""Check wiring/configuration without loading heavy neural network weights."""
from __future__ import annotations
import argparse
from src.config import catalog, load_config, path_from_root


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--strict', action='store_true', help='Exit nonzero if a required checkpoint/dataset is absent')
    args = parser.parse_args(argv)
    types, variants = catalog()
    print(f'ACS catalog: {len(types)} types, {sum(map(len, variants.values()))} variants')
    missing = []
    for key, label in [('classification', 'ConvNeXt checkpoint'),
                       ('detection', 'YOLO checkpoint')]:
        checkpoint = path_from_root(load_config(key)['checkpoint'])
        exists = checkpoint.is_file()
        print(f'{"OK" if exists else "MISSING"}: {label}: {checkpoint}')
        if not exists:
            missing.append(label)
    color = load_config('color')
    color_path = path_from_root(color['checkpoint'])
    if color.get('enabled') is not True:
        print('INVALID: variant recognition must be enabled in configs/color.yaml')
        missing.append('mandatory variant classifier enabled')
    print(f'{"OK" if color_path.is_file() else "MISSING"}: required color checkpoint: {color_path}')
    if not color_path.is_file():
        missing.append('color checkpoint')
    classification = path_from_root(load_config('classification')['dataset']['processed'])
    detection = path_from_root(load_config('detection')['dataset']['yaml'])
    print(f'{"OK" if any(classification.rglob("*.jpg")) else "MISSING"}: V2 processed classification images: {classification}')
    print(f'{"OK" if detection.is_file() else "MISSING"}: V2 YOLO data.yaml: {detection}')
    print('INFO: Java worker uses application.yaml and detection.yaml inference settings')
    if missing and args.strict:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())