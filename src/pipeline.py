"""ACS — Étape 1.7 : pipeline bout en bout type + couleur.

Utilise le checkpoint MobileNetV3 Small (type) et le KNN HSV S01 (couleur).
Le TYPE PRÉDIT par MobileNetV3 est transmis au KNN, jamais le type réel.

Depuis la racine du projet :
    python -m src.pipeline --evaluate-val --device cpu
    python -m src.pipeline --image datasets/classification/raw/.../photo.jpg --device cpu

Ce module n'entraîne aucun modèle et n'accède pas au jeu de test S03.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.classification.models import create_model
from src.color.color_classifier import (
    ABSTAIN,
    extract_features_rgb,
    predict_color,
    read_split,
)
from src.color.hsv_analysis import resize_for_analysis
from src.data.augmentation import get_evaluation_transform
from src.data.preprocessing import load_image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TYPE_MODEL = ROOT / "models/mobilenetv3_linear_probe_v1_best.pth"
DEFAULT_COLOR_MODEL = ROOT / "models/color_hsv_knn_v1.joblib"
PREFIX = "acs_pipeline_v1_val"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name not in ("cpu", "cuda"):
        raise ValueError("Le device doit être auto, cpu ou cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponible : utilisez --device cpu")
    return torch.device(device_name)


class ACSPipeline:
    """Prédit type et couleur à partir d'une seule photographie RGB."""

    def __init__(
        self,
        type_checkpoint: Path = DEFAULT_TYPE_MODEL,
        color_model: Path = DEFAULT_COLOR_MODEL,
        device: str = "cpu",
        root: Path = ROOT,
    ):
        self.root = Path(root)
        self.type_checkpoint = self._resolve(type_checkpoint)
        self.color_model = self._resolve(color_model)
        self.device = resolve_device(device)

        if not self.type_checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint du type introuvable : {self.type_checkpoint}")
        if not self.color_model.is_file():
            raise FileNotFoundError(f"Modèle couleur introuvable : {self.color_model}")

        # Poids issus de notre entraînement, chargés sans téléchargement ImageNet.
        checkpoint = torch.load(
            self.type_checkpoint, map_location="cpu", weights_only=True
        )
        if checkpoint.get("architecture") != "mobilenet_v3_small":
            raise ValueError("Architecture du checkpoint non prise en charge")
        self.classes = list(checkpoint["classes"])
        if len(self.classes) != len(set(self.classes)) or not self.classes:
            raise ValueError("Liste de classes invalide dans le checkpoint")
        mapping = {name: index for index, name in enumerate(self.classes)}
        if checkpoint["class_to_idx"] != mapping:
            raise ValueError("Mapping des classes incohérent dans le checkpoint")

        method = checkpoint["preprocessing_method"]
        self.transform = get_evaluation_transform(method=method)
        self.preprocessing_method = method
        self.type_epoch = int(checkpoint.get("epoch", -1))
        self.type_model = create_model(
            num_classes=len(self.classes), pretrained=False,
            mode=checkpoint.get("mode", "linear_probe"),
        )
        self.type_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.type_model = self.type_model.to(self.device).eval()

        # ATTENTION : joblib charge du pickle. Ouvrir UNIQUEMENT le modèle
        # que vous avez généré vous-même dans votre dépôt de projet.
        self.color_bundle = joblib.load(self.color_model)
        if self.color_bundle.get("format_version") != 1:
            raise ValueError("Format du modèle couleur non pris en charge")
        if self.color_bundle.get("trained_on") != "S01":
            raise ValueError("Le modèle couleur doit avoir été entraîné sur S01")
        self.variants = self.color_bundle["variants"]
        if set(self.classes) != set(self.variants):
            raise ValueError("Les types du modèle MobileNet et du KNN diffèrent")
        for candy_type, colors in self.variants.items():
            if not colors or len(colors) != len(set(colors)):
                raise ValueError(f"Variantes invalides : {candy_type}")
            if len(colors) > 1 and candy_type not in self.color_bundle["models"]:
                raise ValueError(f"KNN manquant pour : {candy_type}")
        self.variant_labels = [
            f"{candy_type}/{color}"
            for candy_type in self.classes
            for color in self.variants[candy_type]
        ]

    def _resolve(self, path: Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self.root / path

    def predict_rgb(self, rgb: np.ndarray) -> dict:
        """Inférence intégrale sans accès à une étiquette réelle."""
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("L'entrée doit être une image RGB uint8 H×W×3")
        image = Image.fromarray(rgb)
        inputs = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = self.type_model(inputs).softmax(dim=1)[0].cpu()
        predicted_index = int(probabilities.argmax().item())
        predicted_type = self.classes[predicted_index]

        # La segmentation d'origine est appliquée aux pixels RGB, pas au
        # tenseur normalisé et PAS à la couleur/type réel de la photographie.
        features = extract_features_rgb(
            rgb,
            max_side=int(self.color_bundle["max_side"]),
            min_saturation=int(self.color_bundle["min_saturation"]),
        )
        color_result = predict_color(self.color_bundle, predicted_type, features)
        predicted_color = color_result["prediction"]
        score = float(color_result["score"])
        score = score if math.isfinite(score) else None

        return {
            "type_predit": predicted_type,
            "confiance_type": float(probabilities[predicted_index].item()),
            "couleur_predite": predicted_color,
            "score_couleur": score,
            "source_couleur": color_result["source"],
            "variante_predite": f"{predicted_type}/{predicted_color}",
            "segmentation_status": features["status"],
            "hue_mean": features["hue_mean"],
            "s_median": features["s_median"],
            "v_median": features["v_median"],
            "mask_fraction": features["mask_fraction"],
        }

    def predict_image(self, path: Path) -> dict:
        """Charge une photographie, puis applique le pipeline bout en bout."""
        path = self._resolve(path)
        if not path.is_file():
            raise FileNotFoundError(f"Photographie introuvable : {path}")
        return self.predict_rgb(np.asarray(load_image(path), dtype=np.uint8))


def plot_confusion(matrix: np.ndarray, labels: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Variante prédite")
    ax.set_ylabel("Variante réelle")
    ax.set_title("ACS — Matrice de confusion des variantes (pipeline complet, S02)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j]:
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() * 0.55 else "black",
                        fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Nombre d'images")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_errors(predictions: pd.DataFrame, path: Path, root: Path,
                max_errors: int = 12) -> None:
    errors = predictions.loc[~predictions["variante_correcte"]].head(max_errors)
    if errors.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "Aucune erreur sur S02", ha="center", va="center")
        ax.axis("off")
    else:
        ncols = min(3, len(errors))
        nrows = math.ceil(len(errors) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                                 squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, row in zip(axes.flat, errors.to_dict(orient="records")):
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = root / image_path
            rgb = np.asarray(load_image(image_path), dtype=np.uint8)
            ax.imshow(resize_for_analysis(rgb, max_side=550))
            ax.set_title(
                f"Réel : {row['variante_reelle']}\n"
                f"Prédit : {row['variante_predite']}\n"
                f"Type : {'OK' if row['type_correct'] else 'ERREUR'}",
                fontsize=10,
            )
    fig.suptitle("ACS — Erreurs des variantes (pipeline complet, validation S02)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_validation(pipeline: ACSPipeline, *, split_dir: Path | None = None,
                        results_dir: Path | None = None,
                        max_errors: int = 12) -> tuple[dict, dict[str, Path]]:
    """Évalue S02; les vrais labels servent aux métriques, jamais aux prédictions."""
    if max_errors < 1:
        raise ValueError("max_errors doit être >= 1")
    split_dir = Path(split_dir) if split_dir else pipeline.root / "datasets/metadata/splits"
    results_dir = Path(results_dir) if results_dir else pipeline.root / "results"
    metadata = read_split("val", split_dir=split_dir)  # refuse S03 par conception
    if not set(metadata["type"]).issubset(set(pipeline.classes)):
        raise ValueError("Type réel absent du checkpoint")

    records = []
    for item in metadata.to_dict(orient="records"):
        typ, color = item["type"], item["couleur"]
        if color not in pipeline.variants[typ]:
            raise ValueError(f"Variante réelle inconnue : {typ}/{color}")
        prediction = pipeline.predict_image(Path(item["image_path"]))
        records.append({
            "image_path": item["image_path"],
            "session_id": item["session_id"],
            "type_reel": typ,
            "couleur_reelle": color,
            "variante_reelle": f"{typ}/{color}",
            **prediction,
            "type_correct": typ == prediction["type_predit"],
            "couleur_correcte": color == prediction["couleur_predite"],
            "variante_correcte": (f"{typ}/{color}" == prediction["variante_predite"]),
        })
    predictions = pd.DataFrame(records)
    y_true = predictions["variante_reelle"].tolist()
    y_pred = predictions["variante_predite"].tolist()
    known = pipeline.variant_labels
    abstention_labels = [
        f"{typ}/{ABSTAIN}" for typ in pipeline.classes
        if f"{typ}/{ABSTAIN}" in set(y_pred)
    ]
    matrix_labels = known + abstention_labels
    if not set(y_pred).issubset(set(matrix_labels)):
        raise ValueError("Variante prédite inconnue")
    matrix = confusion_matrix(y_true, y_pred, labels=matrix_labels)
    report_dict = classification_report(
        y_true, y_pred, labels=known, output_dict=True, zero_division=0
    )
    report_rows = [
        {"variante": variant, "precision": report_dict[variant]["precision"],
         "recall": report_dict[variant]["recall"],
         "f1_score": report_dict[variant]["f1-score"],
         "support": int(report_dict[variant]["support"])}
        for variant in known
    ]
    type_ok = predictions["type_correct"]
    summary = {
        "note": "Pipeline bout en bout : le KNN reçoit exclusivement le type prédit par MobileNetV3.",
        "split": "val", "session": "S02",
        "n_images": len(predictions),
        "n_variants": len(known),
        "n_type_correct": int(type_ok.sum()),
        "n_variant_correct": int(predictions["variante_correcte"].sum()),
        "n_variant_errors": int((~predictions["variante_correcte"]).sum()),
        "n_abstentions": int((predictions["couleur_predite"] == ABSTAIN).sum()),
        "type_accuracy": float(type_ok.mean()),
        "color_accuracy_given_predicted_type_correct": float(
            predictions.loc[type_ok, "couleur_correcte"].mean()
        ) if type_ok.any() else None,
        "variant_accuracy": float(accuracy_score(y_true, y_pred)),
        "variant_macro_f1": float(f1_score(
            y_true, y_pred, labels=known, average="macro", zero_division=0
        )),
        "type_checkpoint": str(pipeline.type_checkpoint),
        "color_model": str(pipeline.color_model),
        "type_checkpoint_epoch": pipeline.type_epoch,
        "type_checkpoint_sha256": sha256_file(pipeline.type_checkpoint),
        "color_model_sha256": sha256_file(pipeline.color_model),
        "validation_csv_sha256": sha256_file(split_dir / "classification_val.csv"),
        "preprocessing_method": pipeline.preprocessing_method,
    }

    tables = results_dir / "tables"
    figures = results_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": tables / f"{PREFIX}_predictions.csv",
        "classification_report": tables / f"{PREFIX}_classification_report.csv",
        "confusion_csv": tables / f"{PREFIX}_confusion.csv",
        "summary": tables / f"{PREFIX}_summary.json",
        "confusion_png": figures / f"{PREFIX}_confusion.png",
        "errors_png": figures / f"{PREFIX}_errors.png",
    }
    predictions.to_csv(paths["predictions"], index=False)
    pd.DataFrame(report_rows).to_csv(paths["classification_report"], index=False)
    pd.DataFrame(matrix, index=matrix_labels, columns=matrix_labels).to_csv(
        paths["confusion_csv"], index_label="variante_reelle"
    )
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_confusion(matrix, matrix_labels, paths["confusion_png"])
    plot_errors(predictions, paths["errors_png"], pipeline.root, max_errors=max_errors)

    print("=== ACS — Évaluation intégrée : validation S02 ===")
    print(f"Images : {summary['n_images']} | Variantes : {summary['n_variants']}")
    print(f"Type correct : {summary['n_type_correct']}/{summary['n_images']} "
          f"({summary['type_accuracy']:.2%})")
    print(f"Variante correcte : {summary['n_variant_correct']}/{summary['n_images']} "
          f"({summary['variant_accuracy']:.2%})")
    print(f"Macro F1 ({summary['n_variants']} variantes) : {summary['variant_macro_f1']:.4f}")
    print(f"Erreurs de variante : {summary['n_variant_errors']} | "
          f"Abstentions : {summary['n_abstentions']}")
    if (~predictions["variante_correcte"]).any():
        print("\n=== Erreurs de variante ===")
        print(predictions.loc[~predictions["variante_correcte"], [
            "image_path", "variante_reelle", "variante_predite",
            "confiance_type", "score_couleur",
        ]].to_string(index=False))
    print("\nFichiers :")
    for name, path in paths.items():
        print(f"{name} : {path}")
    print("\nJeu S03 non utilisé. Aucun ré-entraînement.")
    return summary, paths


def main() -> None:
    parser = argparse.ArgumentParser(description="ACS — Pipeline type + couleur")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--evaluate-val", action="store_true",
                       help="Évaluer les 13 variantes sur S02")
    group.add_argument("--image", type=Path, help="Classer une seule photographie")
    parser.add_argument("--type-checkpoint", type=Path, default=DEFAULT_TYPE_MODEL)
    parser.add_argument("--color-model", type=Path, default=DEFAULT_COLOR_MODEL)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--max-errors", type=int, default=12)
    args = parser.parse_args()
    pipeline = ACSPipeline(
        type_checkpoint=args.type_checkpoint, color_model=args.color_model,
        device=args.device,
    )
    if args.evaluate_val:
        evaluate_validation(pipeline, max_errors=args.max_errors)
    else:
        result = pipeline.predict_image(args.image)
        print("=== ACS — Prédiction type + couleur ===")
        print(f"Photographie : {args.image}")
        print(f"Type : {result['type_predit']} "
              f"(Softmax : {result['confiance_type']:.2%})")
        print(f"Couleur : {result['couleur_predite']} "
              f"(méthode : {result['source_couleur']})")
        if result["score_couleur"] is not None:
            print(f"Vote KNN non calibré : {result['score_couleur']:.3f}")
        print(f"Variante : {result['variante_predite']}")
        if result["couleur_predite"] == ABSTAIN:
            print("ATTENTION : segmentation couleur non fiable")


if __name__ == "__main__":
    main()