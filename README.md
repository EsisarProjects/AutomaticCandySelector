# ACS — Automatic Candy Selector (refactored source)

A single Python inference pipeline with a **ConvNeXt-Tiny type classifier** and
**YOLO candy detector**, exposed to a Java Swing UI through a persistent Python
worker. 

## Project responsibilities

| Location | Purpose |
| --- | --- |
| `configs/classes.yaml` | Six candy types, 22 allowed variants |
| `configs/classification.yaml` | ConvNeXt dataset splits, training, preprocessing, deployed checkpoint |
| `configs/detection.yaml` | YOLO dataset, training, inference and deployed checkpoint |
| `configs/color.yaml` | Optional HSV/KNN V2 color model; disabled until validated |
| `configs/application.yaml` | Inference output directory, device and crop saving |
| `models/classification/` | Deployed ConvNeXt checkpoint |
| `models/detection/` | Deployed YOLO checkpoint |
| `models/color/` | Optional deployed color-classification checkpoint |
| `results/` | Generated figures/metrics and inference images; NOT model storage |
| `datasets/v2/` | New raw/processed images and detection annotations |
| `src/pipeline.py` | Type + optional variant classification |
| `src/detection/integrated_pipeline.py` | Detection -> crop -> classification -> JSON/image |
| `worker.py` | Persistent line-based Java/Python communication |
| `java-app/src/main/java/MainFrame.java` | Java Swing GUI |

## Prepare environment

Python 3.11 or a compatible Python+PyTorch+Ultralytics environment is recommended.
Install PyTorch/torchvision appropriate to your CUDA/CPU setup before installing
additional Python dependencies if required by your platform.

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python tests/test_detection_pipeline.py
python -m src.verify
```

## Bring in existing trained models

**Weights are intentionally not included in this source-only bundle.** Place the
colleague's existing ConvNeXt checkpoint at:

```text
models/classification/convnext_tiny_best.pth
```

If you need to run the old trained YOLO while preparing a new detector, **copy only
its best.pt weight file**, not the historical plots/logs, to:

```text
models/detection/yolo_best.pt
```

The weights must be trained with the architecture/class mapping expected by the
pipeline. ConvNeXt checkpoint `classes` must match the 6 types in `classes.yaml`.
If the colleague's checkpoint was trained on only 4 types, it will be rejected.

## Dataset conversion and training

Place new ORIGINAL photography under
`datasets/v2/classification/raw/<type>/<variant>/<S01|S02|S03|S04>/`.
Do not mix old V1 photographs into V2. Keep the originals unchanged.

```bash
python -m src.data.prepare_classification
python -m src.classification.train --check-only
python -m src.classification.train
# Only AFTER final model selection (held-out evaluation):
python -m src.classification.train --test-only
```

The configuration currently assigns **S01+S02=train, S03=val, S04=test**.
These groups must also use independent physical candy samples / acquisition
conditions to avoid misleading scores. Validation and test are not created by
randomly shuffling correlated photographs. If only S01 is available, independent
validation is missing and the training check will fail intentionally.

`preprocessing.resize_mode: stretch` matches the colleague's original training
code. Both new training and inference use this same transform; do not change it
for an old checkpoint without retraining/validating.

For YOLO, prepare labeled data in the usual Ultralytics structure under
`datasets/v2/detection/yolo/` and copy `data.yaml.example` to `data.yaml`:

```bash
cp datasets/v2/detection/yolo/data.yaml.example datasets/v2/detection/yolo/data.yaml
python -m src.detection.train --check-only
python -m src.detection.train --device cpu
```

When YOLO finishes training, `best.pt` is copied automatically to
`models/detection/yolo_best.pt`. Training logs remain in `results/training/detection`.

## Optional color classifier

The previous V1 HSV/KNN is not automatically considered suitable for all 22
variants. The new V2 training path is:

```bash
python -m src.color.train --check-only
python -m src.color.train
# When model selection is complete:
python -m src.color.train --test-only
```

Examine the V2 validation report under `results/training/color/validation.json`
before setting `configs/color.yaml: enabled: true`. Until enabled, all types can
be predicted but multiple-variant colors are returned as **unknown**, never
fabricated. The single-variant `fraise/rouge` and `oeuf/blanc` are catalog rules,
not visual color measurements. HSV segmentation may be unreliable for white
candies and Schtroumpf bonnet colors; consider improving the variant model.

## Run inference

Once both REQUIRED weights exist:

```bash
python -m src.detection.integrated_pipeline --source path/to/image.jpg
python worker.py  # prints READY and waits for image paths on stdin
```

Java UI (Linux/macOS; Windows can set environment variables analogously):

```bash
export ACS_PROJECT_ROOT="$(pwd)"
export ACS_PYTHON="$(command -v python)"
mkdir -p java-app/build
javac -encoding UTF-8 -d java-app/build java-app/src/main/java/MainFrame.java
java -cp java-app/build MainFrame
```

The application starts Python in the project root and uses the environment's
Python interpreter. The worker prints `READY` once, and for each image prints
`RESULT:<annotated image path>` and `DONE`; errors use `ERROR:...` and `DONE`.
The generated image/JSON live under `results/inference/`.

## Limitations and test status

The included tests verify the configuration contract, session split behavior,
color fallback behavior, and a synthetic detector/classifier integration path.
They **do not validate real ConvNeXt or YOLO accuracy**. Real weights, V2 images,
and actual desktop UI interaction must be tested on the user's computer.
