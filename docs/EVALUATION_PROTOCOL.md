# ACS — Evaluation Protocol

This protocol evaluates **three required recognition stages** (detection, type, variant) and the final Java application. Test sets are reserved until model selection is complete.

## Experiments and ground truth

| ID | Name | Input / reference | Output |
| --- | --- | --- | --- |
| E1 | Type recognition | Isolated candy photographs: true type | Accuracy, per-type P/R/F1, 6×6 confusion matrix |
| E2 | Variant recognition independent of type classifier | Isolated candy photographs: **true type supplied** to variant classifier | Accuracy, per-variant P/R/F1, unknown rate, 21-variant confusion matrix |
| E3 | Joint type–variant classification | Isolated candy photographs: **predicted** type and then predicted variant | Exact 21-label accuracy, macro-F1, unknown rate, per-class matrix |
| E4 | YOLO detection | Test scene images and one-class `bonbon` bounding boxes | mAP50, mAP50:95, precision, recall |
| E5 | Full-scene end-to-end recognition | Test scenes + **per-object type, variant, and box** | Box matching, detection precision/recall, exact object recognition precision/recall/F1, exact-scene success |
| E6 | Final Java application | Independent images and scenarios | Worker/backend artifact integrity and timing + manual UI checklist |

### Why separate E2 and E3?

E2 receives the ground-truth type and measures the color/variant classifier **without any ConvNeXt errors**. E3 uses ConvNeXt predictions and measures how classification errors propagate. The single-variant `fraise/rouge` and `oeuf/blanc` rules are included in full-system accuracy but should be reported separately from the learned multivariant prediction accuracy.

An **unknown** color is a permissible rejection/abstention for a difficult image, but it is **not a correct full variant prediction**. Report its rate rather than silently dropping those samples.

## Experimental independence

1. The external `Train_Val` images are split reproducibly into training and validation, while the external `Test` images remain fully held out.
2. Distinct physical candies and acquisition conditions should remain in the same split; near-duplicates must not cross splits. A random split alone cannot guarantee this.
3. Grouped cross-validation is not available without group/session metadata. Do not describe the current holdout split as cross-validation.
4. Hyperparameters, confidence thresholds, IoU rules, and selected checkpoints are fixed **before** test evaluation.
5. YOLO detector evaluation uses the native one-class labels; full-scene recognition uses additional reference metadata described below.

## Per-object full-scene ground truth

The YOLO dataset can stay single-class (`bonbon`), but each evaluation candy must also have a true type and variant. Create:

```text
datasets/v2/detection/annotations/
├── val_objects.csv
└── test_objects.csv
```

Each CSV contains:

```csv
image,x1,y1,x2,y2,type,variant
```

- `image`: **filename only**, exactly matching a JPEG/PNG in `datasets/v2/detection/yolo/<split>/images/`.
- `x1,y1,x2,y2`: bounding-box corner coordinates **in original image pixels**, with `(0,0)` at the top left; ensure `x2>x1`, `y2>y1`.
- `type`: one of the six catalog types.
- `variant`: one of the allowed variants for that type.
- Each candy occupies **one row**; a scene with five candies has five rows.
- Negative images (no candies) are included in the images folder with **no corresponding CSV rows**.
- The reference CSV is read **only for evaluation**, never passed to the detector or classifiers.

The matching algorithm associates predicted and reference boxes one-to-one by descending IoU (default `0.50`). A correctly localized but wrongly classified candy is **not** fully correct. An undetected candy becomes a false negative for end-to-end recognition. A spurious box is a false positive. The exact-scene metric requires every candy to be correctly matched and labeled, with no extra predictions.

The detector's own mAP evaluation remains a separate E4 experiment. The end-to-end evaluator computes precision/recall at the configured **fixed detector threshold**, not mAP across thresholds.

## Report artifacts

The code creates evaluation reports under `results/evaluation/`:

- `classification/{val,test}/`: type accuracy, precision/recall/F1, confusion matrix and predictions.
- `color/{val,test}/`: conditional variant metrics, unknown rate and a breakdown for multivariant types.
- `joint_classification/{val,test}/`: 6-type and 21-variant performance using predicted types.
- `detection/{val,test}/`: native YOLO metrics and figures.
- `end_to_end/{val,test}/`: per-scene matches, exact object/scene metrics and runtime.
- `application/`: persistent worker/JSON functional checks and wall-clock runtime.

## Final Java Swing acceptance checklist

Perform these checks **through the graphical application** and record the outcome and observed behavior. They are manual UI evaluations, not covered by the Python worker benchmark:

| Test | Expected outcome | Status |
| --- | --- | --- |
| Choose a supported scene image | Image loads correctly; UI remains responsive | Not executed |
| Process a scene with multiple candies | Every result has a box, predicted type, and a variant or explicit unknown status | Not executed |
| Process a scene without candies | UI displays an empty result without an exception | Not executed |
| Invalid image path / corrupted file | Error appears in the UI; application remains usable | Not executed |
| Missing checkpoint | Clear startup error; no misleading success or fake classification | Not executed |
| Process consecutive images | Persistent worker handles all requests without model reload per image | Not executed |
| Verify annotated image and JSON | Displayed output agrees with serialized predictions | Not executed |
| Evaluate usability/responsiveness | Record interaction issues and total user-perceived latency | Not executed |

A completed application evaluation should report both **recognition accuracy** (E5) and **operational quality** (E6) without conflating them.