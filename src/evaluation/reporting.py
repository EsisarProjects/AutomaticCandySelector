"""Tabular and graphical reports for classification performance."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def save_report(y_true, y_pred, names, output, *, split, files=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError('Expected nonempty, equally-sized ground truth and predictions')
    labels = list(range(len(names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                      zero_division=0, output_dict=True)
    results = {'split': split, 'samples': len(y_true),
               'accuracy': float(accuracy_score(y_true, y_pred)),
               'macro_precision': per_class['macro avg']['precision'],
               'macro_recall': per_class['macro avg']['recall'],
               'macro_f1': per_class['macro avg']['f1-score'],
               'weighted_f1': per_class['weighted avg']['f1-score'],
               'per_class': {name: per_class[name] for name in names},
               'confusion_matrix': matrix.tolist()}
    (output/'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    with (output/'confusion_matrix.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['true/predicted', *names])
        w.writerows([[name, *row] for name, row in zip(names, matrix.tolist())])
    with (output/'per_class_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['class', 'precision', 'recall', 'f1', 'support'])
        for name in names:
            row = per_class[name]
            w.writerow([name, row['precision'], row['recall'], row['f1-score'], row['support']])
    if files is not None:
        if len(files) != len(y_true):
            raise ValueError('Files/labels mismatch')
        with (output/'predictions.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['file', 'actual', 'predicted', 'correct'])
            for filename, truth, pred in zip(files, y_true, y_pred):
                w.writerow([str(filename), names[truth], names[pred], truth == pred])
    fig, ax = plt.subplots(figsize=(max(7, len(names)*.85), max(6, len(names)*.85)))
    img = ax.imshow(matrix)
    ax.set_xticks(labels, names, rotation=45, ha='right')
    ax.set_yticks(labels, names)
    ax.set(xlabel='Predicted', ylabel='Actual', title=f'{split} confusion matrix')
    for i in labels:
        for j in labels:
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center')
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    fig.savefig(output/'confusion_matrix.png', dpi=180)
    plt.close(fig)
    return results


def save_history(history, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with (output/'history.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    for name in ('loss', 'accuracy'):
        fig, ax = plt.subplots(figsize=(8, 5))
        for prefix in ('train', 'val'):
            key = prefix+'_'+name
            if key in history[0]:
                ax.plot([x['epoch'] for x in history], [x[key] for x in history], label=prefix)
        ax.set(xlabel='Epoch', ylabel=name, title=f'ConvNeXt training and validation: {name}')
        ax.legend()
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(output/f'{name}_curve.png', dpi=180)
        plt.close(fig)
