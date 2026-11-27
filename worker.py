"""Persistent stdin/stdout bridge for the Java Swing application.

Protocol: one absolute image path per input line. Emit READY once, then
RESULT:<annotated_image_path> followed by DONE per request, or ERROR:... + DONE.
Do not print fake RESULT after an error. Models remain loaded between requests.
"""
from __future__ import annotations
import sys
from collections import deque
from queue import Queue
from threading import Thread
from pathlib import Path
import time
import cv2
import numpy as np
from src.config import load_config, path_from_root


def run_camera_stream(detector, classifier, inference, commands, pending, device, integration):
    """Live preview on the main thread; MODE:FILE can stop it immediately."""
    from queue import Empty
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR:Cannot open webcam', flush=True)
        cap.release()
        return True

    print("INFO:Live Camera started. Press 'q' or select File Mode to exit.", flush=True)
    try:
        while True:
            # Reader thread keeps stdin responsive while frames are processed.
            try:
                while True:
                    command = commands.get_nowait()
                    if command is None:
                        return False  # stdin was closed
                    command = command.strip()
                    if command == 'MODE:FILE':
                        return True
                    if command and command != 'MODE:CAMERA':
                        pending.append(command)  # handle image request after camera closes
            except Empty:
                pass

            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections, _ = integration.run_stages(
                rgb, detector=detector, classifier=classifier,
                detector_conf=float(inference['conf']),
                imgsz=int(inference['image_size']), device=device,
                iou=float(inference['iou']),
                padding=float(inference['padding']),
                type_warning_threshold=float(inference['type_warning_threshold']),
            )
            annotated = np.asarray(integration.draw_detections(rgb, detections))
            cv2.imshow('ACS - Live Camera Pipeline', cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as exc:
        print(f'ERROR:Camera stream failed: {exc}', flush=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print('INFO:Live Camera stopped', flush=True)
    return True


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
    
    commands = Queue()
    pending = deque()

    def read_stdin():
        for line in sys.stdin:
            commands.put(line.strip())
        commands.put(None)

    Thread(target=read_stdin, daemon=True).start()
    while True:
        command = pending.popleft() if pending else commands.get()
        if command is None:
            break
        if not command:
            continue
        if command == 'MODE:CAMERA':
            print('INFO:Switched to Camera Mode', flush=True)
            keep_running = run_camera_stream(detector, classifier, inference,
                                             commands, pending, device, integration)
            print('INFO:Switched to File Mode', flush=True)
            if not keep_running:
                break
            continue
        if command == 'MODE:FILE':
            print('INFO:Switched to File Mode', flush=True)
            continue

        # Xử lý file ảnh tĩnh thông qua thao tác kéo-thả hoặc chọn tệp
        filename = command
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
