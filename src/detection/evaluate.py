"""Independent YOLO validation / final test, using a frozen checkpoint.

    python -m src.detection.evaluate --split val
    python -m src.detection.evaluate --split test

Do not use test results to tune thresholds or model selection.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from src.config import load_config, path_from_root
from src.detection.train import inspect_dataset


def run(split: str, device: str | None = None) -> dict:
    if split not in ('val', 'test'):
        raise ValueError('Only validation or held-out test are allowed')
    cfg = load_config('detection')
    dataset = inspect_dataset(path_from_root(cfg['dataset']['yaml']))
    checkpoint = path_from_root(cfg['checkpoint'])
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Frozen detector checkpoint not found: {checkpoint}')
    from ultralytics import YOLO
    out = path_from_root('results/evaluation/detection')
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(checkpoint))
    metrics = model.val(data=dataset['yaml'], split=split,
                        imgsz=int(cfg['training']['image_size']),
                        batch=int(cfg['training']['batch_size']),
                        device=device or str(cfg['training']['device']),
                        workers=int(cfg['training']['workers']),conf=0.767,
                        plots=True, project=str(out), name=split,
                        exist_ok=True, verbose=False)
    box = metrics.box
    result = {'split': split, 'images': dataset['images'][split],
              'checkpoint': str(checkpoint),
              'precision': float(box.mp), 'recall': float(box.mr),
              'map50': float(box.map50), 'map50_95': float(box.map),
              'classes': dataset['classes']}
    report_dir = Path(getattr(metrics, 'save_dir', out/split))
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'{split.upper()} YOLO: {result} | figures: {report_dir}')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--device', default=None)
    args = parser.parse_args(argv)
    run(args.split, args.device)


if __name__ == '__main__':
    main()
