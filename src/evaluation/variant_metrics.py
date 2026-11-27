"""Catalog-driven variant reporting, with a separate unknown/rejection outcome."""
from __future__ import annotations
from pathlib import Path
from src.config import catalog
from src.evaluation.reporting import save_report
from sklearn.metrics import precision_recall_fscore_support

UNKNOWN = 'unknown'


def variant_names() -> list[str]:
    _, mapping = catalog()
    return [f'{kind}/{color}' for kind, colors in mapping.items() for color in colors]


def save_variant_report(true_labels, pred_labels, output: Path, *, split: str, files=None):
    names = variant_names() + [UNKNOWN]
    indices = {name: i for i, name in enumerate(names)}
    unknown = indices[UNKNOWN]
    if any(label not in indices or label == UNKNOWN for label in true_labels):
        raise ValueError('Ground truth must contain only catalog type/variant pairs')
    y_true = [indices[label] for label in true_labels]
    y_pred = [indices.get(label, unknown) for label in pred_labels]
    metrics = save_report(y_true, y_pred, names, output, split=split, files=files)
    # 'unknown' is a REJECTION outcome, not a catalog class.
    # Include unknown predictions as errors but exclude the unknown row
    # when computing macro scores for the actual catalog variants.
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(names)-1)), zero_division=0)
    metrics['macro_precision'] = float(precision.mean())
    metrics['macro_recall'] = float(recall.mean())
    metrics['macro_f1'] = float(f1.mean())
    metrics['weighted_f1'] = float((f1*support).sum()/max(1, support.sum()))
    metrics['unknown_predictions'] = sum(label == UNKNOWN for label in pred_labels)
    metrics['unknown_rate'] = metrics['unknown_predictions'] / len(y_true)
    metrics['evaluated_variants'] = len(names) - 1
    import json
    (Path(output)/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics
