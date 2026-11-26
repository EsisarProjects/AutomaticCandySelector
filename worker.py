"""Persistent stdin/stdout bridge for the Java Swing application.

Protocol: one absolute image path per input line. Emit READY once, then
RESULT:<annotated_image_path> followed by DONE per request, or ERROR:... + DONE.
Do not print fake RESULT after an error. Models remain loaded between requests.
"""
from __future__ import annotations
import sys
from pathlib import Path
import time
import cv2
import numpy as np
from src.config import load_config, path_from_root


def run_camera_stream(detector, classifier, inference, output):
    """Mở webcam thời gian thực, nhận diện kẹo trực tiếp qua từng frame."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR:Cannot open webcam", flush=True)
        return

    print("INFO:Live Camera started. Press 'q' in the camera window to exit.", flush=True)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Chuyển đổi không gian màu từ BGR (OpenCV) sang RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Chạy pipeline nhận diện trên frame hiện tại
            detections, _ = integration.run_stages(
                rgb,
                detector=detector,
                classifier=classifier,
                detector_conf=float(inference['conf']),
                imgsz=int(inference['image_size']),
                device=device,
                iou=float(inference['iou']),
                padding=float(inference['padding']),
                type_warning_threshold=float(inference['type_warning_threshold']),
            )

            # Vẽ kết quả (khung và số thứ tự) trực tiếp lên frame
            annotated_rgb = np.asarray(integration.draw_detections(rgb, detections))
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            # Hiển thị cửa sổ video trực tiếp
            cv2.imshow("ACS - Live Camera Pipeline", annotated_bgr)

            # Nhấn phím 'q' trên bàn phím để tắt cửa sổ camera
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("INFO:Live Camera stopped", flush=True)


def main():
    global integration, device
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
    
    current_mode = "FILE"

    for request in sys.stdin:
        command = request.strip()
        if not command:
            continue

        # Lắng nghe lệnh chuyển đổi chế độ từ giao diện Java Swing
        if command == "MODE:CAMERA":
            current_mode = "CAMERA"
            print("INFO:Switched to Camera Mode", flush=True)
            # Kích hoạt luồng chạy camera real-time
            run_camera_stream(detector, classifier, inference, output)
            continue
        elif command == "MODE:FILE":
            current_mode = "FILE"
            print("INFO:Switched to File Mode", flush=True)
            continue

        if current_mode == "CAMERA":
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
