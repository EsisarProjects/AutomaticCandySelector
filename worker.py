"""Persistent stdin/stdout bridge for the Java Swing application.

Protocol: one absolute image path per input line. Emit READY once, then
RESULT:<annotated_image_path> followed by DONE per request, or ERROR:... + DONE.
Do not print fake RESULT after an error. Models remain loaded between requests.
"""
from __future__ import annotations
import sys
from pathlib import Path
import time
from src.config import load_config, path_from_root


def main():
    try:
        import torch
        from ultralytics import YOLO
        from src.pipeline import ACSPipeline
        import src.detection.integrated_pipeline as integration
        app = load_config('application')
        inference = load_config('detection')['inference']
        device = app.get('device', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        detector_file = integration.DEFAULT_DETECTOR
        type_file = integration.DEFAULT_TYPE
        for path in [detector_file, type_file]:
            if not path.is_file():
                raise FileNotFoundError(f'Required model missing: {path}')
        detector = YOLO(str(detector_file))
        classifier = ACSPipeline(type_checkpoint=type_file, device=device)
        output = path_from_root(app['output_dir'])
    except Exception as exc:
        print(f'ERROR:Cannot start ACS worker: {exc}', flush=True)
        return 2

    print('READY', flush=True)
    for request in sys.stdin:
        filename = request.strip()
        if not filename:
            continue
        try:
            path = Path(filename).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f'Image not found: {path}')
            start = time.perf_counter()
            report = integration.process_file(
                image_path=path,
                detector=detector,
                classifier=classifier,
                output=output,
                detector_conf=float(inference['conf']),
                imgsz=int(inference['image_size']),
                device=device,
                iou=float(inference['iou']),
                padding=float(inference['padding']),
                type_warning_threshold=float(inference['type_warning_threshold']),
                save_crops=bool(app.get('save_crops', False)),
            )
            print(f'INFO:Processed {report["nombre_detections"]} candies in '
                  f'{time.perf_counter()-start:.2f}s', flush=True)
            print(f'RESULT:{report["image_annotee"]}', flush=True)
        except Exception as exc:
            print(f'ERROR:{type(exc).__name__}: {exc}', flush=True)
        finally:
            print('DONE', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
