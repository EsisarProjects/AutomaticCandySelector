"""Évaluation du meilleur checkpoint ACS : validation S02 ou test final S03."""
from pathlib import Path
import argparse
import json
import hashlib

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.classification.models import create_model
from src.data.dataset import ACSDataset
from src.data.augmentation import get_evaluation_transform
from src.evaluation.plots import plot_confusion_matrix, plot_prediction_errors

ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "datasets/metadata/splits"
TABLES_DIR = ROOT / "results/tables"
FIGURES_DIR = ROOT / "results/figures"

SPLIT_SESSIONS = {"val": "S02", "test": "S03"}


def get_split_session(split):
    """Autorise uniquement validation S02 et test final S03."""
    if split not in SPLIT_SESSIONS:
        raise ValueError(f"Split inconnu : {split!r}. Choisir val ou test.")
    return SPLIT_SESSIONS[split]


def sha256_file(path):
    """Empreinte pour tracer exactement le checkpoint et le CSV utilisés."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_classification_metrics(y_true, y_pred, classes):
    """Matrice de confusion, rapport par classe et résumé."""
    labels = list(range(len(classes)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=classes,
        output_dict=True, zero_division=0,
    )
    rows = []
    for name in [*classes, "macro avg", "weighted avg"]:
        values = report[name]
        rows.append({
            "classe": name,
            "precision": values["precision"],
            "recall": values["recall"],
            "f1_score": values["f1-score"],
            "support": int(values["support"]),
        })
    table = pd.DataFrame(rows)
    summary = {
        "n_images": int(len(y_true)),
        "n_correct": int(np.trace(matrix)),
        "n_errors": int(len(y_true) - np.trace(matrix)),
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
    }
    return matrix, table, summary


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponible. Utilisez --device cpu.")
    return torch.device(requested)


def evaluate_checkpoint(checkpoint_path, batch_size=4, device_name="cpu", max_errors=12, split="val"):
    """Évalue sans augmentation le checkpoint fixé sur le split demandé."""
    expected_session = get_split_session(split)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")
    if batch_size < 1:
        raise ValueError("batch_size doit être >= 1")

    # Le fichier a été créé avec torch.save par notre script de training.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "mobilenet_v3_small":
        raise ValueError("Architecture du checkpoint non prise en charge.")
    classes = checkpoint["classes"]
    class_to_idx = checkpoint["class_to_idx"]
    method = checkpoint["preprocessing_method"]

    # Création sans téléchargement de poids ImageNet : tous les poids sont
    # reconstruits depuis model_state_dict dans le checkpoint.
    model = create_model(
        num_classes=len(classes), pretrained=False,
        mode=checkpoint.get("mode", "linear_probe")
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = resolve_device(device_name)
    model = model.to(device).eval()

    csv_path = SPLITS_DIR / f"classification_{split}.csv"
    dataset = ACSDataset(csv_path=csv_path, transform=get_evaluation_transform(method))
    if dataset.classes != classes or dataset.class_to_idx != class_to_idx:
        raise ValueError(
            "L'ordre des classes dans classes.yaml diffère du checkpoint. "
            "Vérifiez les fichiers avant d'évaluer."
        )
    actual_sessions = set(dataset.df["session_id"].astype(str))
    if actual_sessions != {expected_session}:
        raise ValueError(
            f"Sessions incorrectes pour {split} : {actual_sessions}; "
            f"attendu : {expected_session}."
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    records = []
    offset = 0
    with torch.inference_mode():
        for images, labels in loader:
            probabilities = model(images.to(device)).softmax(dim=1).cpu()
            predicted = probabilities.argmax(dim=1)
            labels = labels.cpu()

            for j in range(len(labels)):
                item = dataset.df.iloc[offset + j]
                real_idx = int(labels[j])
                predicted_idx = int(predicted[j])
                row = {
                    "image_id": item["image_id"],
                    "image_path": item["image_path"],
                    "session_id": item["session_id"],
                    "type": item["type"],
                    "couleur": item["couleur"],
                    "prediction": classes[predicted_idx],
                    "confidence": float(probabilities[j, predicted_idx]),
                    "true_class_probability": float(probabilities[j, real_idx]),
                    "correct": real_idx == predicted_idx,
                    "true_index": real_idx,
                    "predicted_index": predicted_idx,
                }
                for k, name in enumerate(classes):
                    row[f"prob_{name}"] = float(probabilities[j, k])
                records.append(row)
            offset += len(labels)

    predictions = pd.DataFrame(records)
    matrix, report_table, summary = compute_classification_metrics(
        predictions["true_index"].tolist(),
        predictions["predicted_index"].tolist(), classes,
    )
    summary.update({
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "preprocessing_method": method,
        "split": split,
        "session": expected_session,
        "test_set_used": split == "test",
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "split_csv_sha256": sha256_file(csv_path),
    })

    run_name = checkpoint_path.stem
    if run_name.endswith("_best"):
        run_name = run_name[:-5]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{run_name}_{split}"

    paths = {
        "predictions": TABLES_DIR / f"{prefix}_predictions.csv",
        "report": TABLES_DIR / f"{prefix}_classification_report.csv",
        "confusion_csv": TABLES_DIR / f"{prefix}_confusion.csv",
        "summary": TABLES_DIR / f"{prefix}_evaluation.json",
        "confusion_png": FIGURES_DIR / f"{prefix}_confusion.png",
        "errors_png": FIGURES_DIR / f"{prefix}_errors.png",
    }
    predictions.to_csv(paths["predictions"], index=False)
    report_table.to_csv(paths["report"], index=False)
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        paths["confusion_csv"], index_label="classe_reelle"
    )
    with open(paths["summary"], "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    display_name = f"{'validation' if split == 'val' else 'test'} {expected_session}"
    plot_confusion_matrix(matrix, classes, paths["confusion_png"], dataset_name=display_name)
    plot_prediction_errors(
        predictions, ROOT, method, paths["errors_png"],
        max_errors=max_errors, dataset_name=display_name
    )

    print(f"\n=== ACS — Évaluation ({display_name}) ===")
    print(f"Checkpoint : {checkpoint_path.name} (époque {checkpoint['epoch']})")
    print(f"Images : {summary['n_images']} | Erreurs : {summary['n_errors']}")
    print(f"Accuracy : {summary['accuracy']:.4f}")
    print(f"Macro F1 : {summary['macro_f1']:.4f}\n")
    print(report_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    errors = predictions.loc[~predictions["correct"]]
    if not errors.empty:
        print("\n=== Images incorrectement classées ===")
        print(errors[["image_path", "type", "prediction", "confidence"]].to_string(index=False))
    for label, path in paths.items():
        print(f"{label} : {path}")
    if split == "test":
        # Comparaison purement descriptive : aucune modification du modèle.
        val_path = TABLES_DIR / f"{run_name}_val_evaluation.json"
        if val_path.is_file():
            with open(val_path, encoding="utf-8") as file:
                val_summary = json.load(file)
            if (Path(val_summary.get("checkpoint", "")).name == checkpoint_path.name
                    and int(val_summary.get("checkpoint_epoch", -1)) == int(checkpoint["epoch"])):
                comparison = pd.DataFrame([
                    {"ensemble": "validation S02", "images": val_summary["n_images"],
                     "erreurs": val_summary["n_errors"], "accuracy": val_summary["accuracy"],
                     "macro_f1": val_summary["macro_f1"]},
                    {"ensemble": "test S03", "images": summary["n_images"],
                     "erreurs": summary["n_errors"], "accuracy": summary["accuracy"],
                     "macro_f1": summary["macro_f1"]},
                ])
                comparison_path = TABLES_DIR / f"{run_name}_val_vs_test.csv"
                comparison.to_csv(comparison_path, index=False)
                paths["comparison"] = comparison_path
                print("\n=== Comparaison S02 / S03 ===")
                print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
                print(f"Comparaison : {comparison_path}")
            else:
                print("\nAttention : le résumé S02 ne correspond pas au checkpoint actuel.")
        print("\nTest S03 évalué. Ne pas ajuster le modèle en fonction de ses résultats.")
    else:
        print("\nJeu de test S03 non utilisé.")
    return summary, paths


def main():
    parser = argparse.ArgumentParser(description="ACS - Évaluation S02 ou test final S03")
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="val = S02 ; test = S03 (évaluation finale)")
    parser.add_argument("--checkpoint", default="models/mobilenetv3_linear_probe_v1_best.pth")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-errors", type=int, default=12)
    args = parser.parse_args()
    evaluate_checkpoint(
        args.checkpoint, batch_size=args.batch_size,
        device_name=args.device, max_errors=args.max_errors, split=args.split,
    )


if __name__ == "__main__":
    main()