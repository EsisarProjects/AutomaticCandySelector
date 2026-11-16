"""ACS — Étape 1.6.3 : classification supervisée de la couleur (HSV).

Utilise exactement la segmentation de hsv_analysis.py (première méthode).
Apprentissage : S01 ; évaluation conditionnelle au type réel : S02.
Le jeu S03 n'est jamais lu par ce programme.

Depuis la racine du projet :
    python -m src.color.color_classifier
    python -m src.color.color_classifier --predict-image photo.jpg --type ourson
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.color.hsv_analysis import (
    compute_hsv_features,
    resize_for_analysis,
    segment_colored_candy,
)
from src.data.preprocessing import load_image

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "datasets/metadata/splits"
CLASSES_YAML = ROOT / "configs/classes.yaml"
MODEL_PATH = ROOT / "models/color_hsv_knn_v1.joblib"
TABLES_DIR = ROOT / "results/tables"
FIGURES_DIR = ROOT / "results/figures"

ABSTAIN = "indetermine"
FEATURE_NAMES = ("hue_sin", "hue_cos", "saturation", "value")


def load_variants(config_path: Path = CLASSES_YAML) -> dict[str, list[str]]:
    """Lit exclusivement les variantes autorisées dans classes.yaml."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or "variantes" not in config:
        raise ValueError("La clé 'variantes' est absente de classes.yaml")
    variants = {str(k): list(v) for k, v in config["variantes"].items()}
    if not variants or any(not colors for colors in variants.values()):
        raise ValueError("Liste de variantes vide")
    if len({c for colors in variants.values() for c in colors}) == 0:
        raise ValueError("Aucune couleur définie")
    return variants


def read_split(split: str, split_dir: Path = SPLIT_DIR) -> pd.DataFrame:
    """Lit S01 ou S02; refuse délibérément l'utilisation de S03 ici."""
    sessions = {"train": "S01", "val": "S02"}
    if split not in sessions:
        raise ValueError("Seuls les splits 'train' (S01) et 'val' (S02) sont autorisés")
    csv_path = Path(split_dir) / f"classification_{split}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)
    required = {"image_path", "type", "couleur", "session_id"}
    if df.empty or not required.issubset(df.columns):
        raise ValueError(f"Métadonnées manquantes ou vides : {csv_path}")
    if df[list(required)].isna().any().any():
        raise ValueError(f"Valeurs manquantes dans : {csv_path}")
    if set(df["session_id"]) != {sessions[split]}:
        raise ValueError(f"{split} doit correspondre uniquement à {sessions[split]}")
    if df["image_path"].duplicated().any():
        raise ValueError(f"Chemins dupliqués dans {csv_path}")
    return df


def extract_features_rgb(rgb: np.ndarray, *, max_side: int = 800,
                         min_saturation: int = 45) -> dict:
    """Segment HSV historique, puis statistiques HSV; n'utilise aucun label."""
    resized = resize_for_analysis(rgb, max_side=max_side)
    mask = segment_colored_candy(resized, min_saturation=min_saturation)
    return compute_hsv_features(resized, mask)


def extract_features_file(path: Path, *, max_side: int = 800,
                          min_saturation: int = 45) -> dict:
    rgb = np.asarray(load_image(path), dtype=np.uint8)
    return extract_features_rgb(rgb, max_side=max_side,
                                min_saturation=min_saturation)


def feature_vector(features: dict) -> np.ndarray | None:
    """Encode H sur le cercle (0 et 179 sont voisins), S et V dans [0,1]."""
    if features.get("status") != "detecte":
        return None
    values = np.array([
        features.get("hue_mean", np.nan),
        features.get("s_median", np.nan),
        features.get("v_median", np.nan),
    ], dtype=np.float64)
    if not np.isfinite(values).all():
        return None
    h, s, v = values
    if not (0 <= h < 180 and 0 <= s <= 255 and 0 <= v <= 255):
        return None
    angle = (2.0 * np.pi * h) / 180.0
    return np.array([math.sin(angle), math.cos(angle), s / 255.0, v / 255.0],
                    dtype=np.float64)


def build_feature_table(metadata: pd.DataFrame, *, root: Path = ROOT,
                        max_side: int = 800, min_saturation: int = 45) -> pd.DataFrame:
    rows = []
    for row in metadata.itertuples(index=False):
        path = Path(row.image_path)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(f"Photographie introuvable : {path}")
        features = extract_features_file(
            path, max_side=max_side, min_saturation=min_saturation
        )
        rows.append({
            "image_path": row.image_path,
            "type": row.type,
            "couleur": row.couleur,
            "session_id": row.session_id,
            **features,
        })
    return pd.DataFrame(rows)


def fit_color_models(train_features: pd.DataFrame,
                     variants: dict[str, list[str]],
                     n_neighbors: int = 3) -> dict:
    """Entraîne un KNN par type uniquement si plusieurs couleurs possibles."""
    if n_neighbors < 1:
        raise ValueError("n_neighbors doit être >= 1")
    known_types = set(variants)
    if not set(train_features["type"]).issubset(known_types):
        raise ValueError("Types du train inconnus dans classes.yaml")
    models = {}
    for candy_type, colors in variants.items():
        group = train_features[train_features["type"] == candy_type]
        if group.empty:
            raise ValueError(f"Type absent de S01 : {candy_type}")
        if not set(group["couleur"]).issubset(set(colors)):
            raise ValueError(f"Étiquette couleur inconnue : {candy_type}")
        missing_colors = set(colors) - set(group["couleur"])
        if missing_colors:
            raise ValueError(f"Variantes absentes de S01 pour {candy_type}: {missing_colors}")
        if len(colors) == 1:
            continue  # La couleur est définie par les variantes du pilote.
        vectors = []
        labels = []
        for row in group.to_dict(orient="records"):
            vector = feature_vector(row)
            if vector is None:
                raise ValueError(
                    f"Échec segmentation dans S01 ({candy_type}): {row['image_path']}"
                )
            vectors.append(vector)
            labels.append(row["couleur"])
        if len(vectors) < n_neighbors:
            raise ValueError(f"Pas assez d'images : {candy_type}")
        estimator = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance"),
        )
        estimator.fit(np.stack(vectors), labels)
        models[candy_type] = estimator
    return {
        "format_version": 1,
        "method": "HSV historique + KNN par type",
        "variants": variants,
        "models": models,
        "feature_names": FEATURE_NAMES,
        "max_side": 800,
        "min_saturation": 45,
        "n_neighbors": n_neighbors,
        "trained_on": "S01",
        "selection_split": "S02",
    }


def predict_color(bundle: dict, candy_type: str, features: dict) -> dict:
    """Prédiction *conditionnelle* au type fourni (non prédit ici)."""
    if candy_type not in bundle["variants"]:
        raise ValueError(f"Type inconnu : {candy_type}")
    allowed = bundle["variants"][candy_type]
    if len(allowed) == 1:
        return {
            "prediction": allowed[0],
            "score": float("nan"),
            "source": "regle_du_pilote",
        }
    vector = feature_vector(features)
    if vector is None:
        return {"prediction": ABSTAIN, "score": float("nan"),
                "source": "segmentation_non_fiable"}
    estimator = bundle["models"][candy_type]
    probabilities = estimator.predict_proba(vector.reshape(1, -1))[0]
    index = int(np.argmax(probabilities))
    return {
        "prediction": str(estimator.classes_[index]),
        "score": float(probabilities[index]),
        "source": "knn_hsv",
    }


def evaluate_color_model(bundle: dict, val_features: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Évalue sur S02 avec le type RÉEL connu; ce n'est pas un score bout en bout."""
    rows = []
    for row in val_features.to_dict(orient="records"):
        typ, true_color = row["type"], row["couleur"]
        if typ not in bundle["variants"] or true_color not in bundle["variants"][typ]:
            raise ValueError(f"Variante inconnue : {typ}/{true_color}")
        result = predict_color(bundle, typ, row)
        rows.append({
            "image_path": row["image_path"],
            "type_reel": typ,
            "couleur_reelle": true_color,
            "couleur_predite": result["prediction"],
            "correct": result["prediction"] == true_color,
            "score_knn": result["score"],
            "source": result["source"],
            "segmentation_status": row["status"],
            "hue_mean": row["hue_mean"],
            "s_median": row["s_median"],
            "v_median": row["v_median"],
            "mask_fraction": row["mask_fraction"],
        })
    predictions = pd.DataFrame(rows)
    true_variants = (predictions["type_reel"] + "/" + predictions["couleur_reelle"]).tolist()
    pred_variants = (predictions["type_reel"] + "/" + predictions["couleur_predite"]).tolist()
    labels = [f"{t}/{c}" for t, cs in bundle["variants"].items() for c in cs]
    summary = {
        "note": "Couleur prédite connaissant le TYPE RÉEL ; pas d'évaluation du pipeline complet.",
        "split": "val",
        "session": "S02",
        "images": len(predictions),
        "correct": int(predictions["correct"].sum()),
        "errors": int((~predictions["correct"]).sum()),
        "abstentions": int((predictions["couleur_predite"] == ABSTAIN).sum()),
        "accuracy_variant_conditionnelle": float(accuracy_score(true_variants, pred_variants)),
        "macro_f1_variantes": float(f1_score(true_variants, pred_variants,
                                               average="macro", labels=labels,
                                               zero_division=0)),
        "classes_variantes": labels,
        "n_neighbors": bundle["n_neighbors"],
    }
    return summary, predictions


def save_confusion_figures(bundle: dict, predictions: pd.DataFrame) -> Path:
    """Matrices par type multi-couleur. Une abstention apparaît comme colonne."""
    variable_types = [t for t, cs in bundle["variants"].items() if len(cs) > 1]
    fig, axes = plt.subplots(1, len(variable_types),
                             figsize=(7 * len(variable_types), 6), squeeze=False)
    for ax, typ in zip(axes[0], variable_types):
        subset = predictions[predictions["type_reel"] == typ]
        colors = bundle["variants"][typ]
        predicted_labels = colors + ([ABSTAIN] if ABSTAIN in set(subset["couleur_predite"]) else [])
        matrix = confusion_matrix(
            subset["couleur_reelle"], subset["couleur_predite"],
            labels=predicted_labels,
        )
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(predicted_labels)), predicted_labels, rotation=40, ha="right")
        ax.set_yticks(range(len(predicted_labels)), predicted_labels)
        ax.set_xlabel("Couleur prédite")
        ax.set_ylabel("Couleur réelle")
        ax.set_title(f"{typ} — validation S02")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.suptitle("ACS — Matrices de confusion des couleurs (type réel connu)")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "color_hsv_knn_v1_val_confusion.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_error_examples(predictions: pd.DataFrame, *, root: Path = ROOT,
                        max_examples: int = 12) -> Path | None:
    """Montre les photographies originales dont la couleur a été mal prédite."""
    errors = predictions.loc[~predictions["correct"]].head(max_examples)
    if errors.empty:
        return None
    ncols = min(3, len(errors))
    nrows = math.ceil(len(errors) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, errors.to_dict(orient="records")):
        path = Path(row["image_path"])
        if not path.is_absolute():
            path = root / path
        rgb = np.asarray(load_image(path), dtype=np.uint8)
        ax.imshow(resize_for_analysis(rgb, max_side=600))
        ax.set_title(
            f"{row['type_reel']}\nRéel : {row['couleur_reelle']}\n"
            f"Prédit : {row['couleur_predite']}", fontsize=11
        )
    fig.suptitle("ACS — Erreurs de couleur (validation S02, type réel connu)")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "color_hsv_knn_v1_val_errors.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def run(max_side: int = 800, min_saturation: int = 45,
        n_neighbors: int = 3) -> None:
    """Entraînement S01 + évaluation S02 (S03 jamais consulté)."""
    for folder in (MODEL_PATH.parent, TABLES_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    variants = load_variants()
    train_df = read_split("train")
    val_df = read_split("val")
    if not set(train_df["image_path"]).isdisjoint(set(val_df["image_path"])):
        raise ValueError("Chevauchement des photographies S01/S02")
    train_features = build_feature_table(train_df, max_side=max_side,
                                       min_saturation=min_saturation)
    val_features = build_feature_table(val_df, max_side=max_side,
                                     min_saturation=min_saturation)
    bundle = fit_color_models(train_features, variants, n_neighbors=n_neighbors)
    bundle["max_side"] = max_side
    bundle["min_saturation"] = min_saturation
    joblib.dump(bundle, MODEL_PATH)

    train_features.to_csv(TABLES_DIR / "color_hsv_knn_v1_train_features.csv", index=False)
    val_features.to_csv(TABLES_DIR / "color_hsv_knn_v1_val_features.csv", index=False)
    summary, predictions = evaluate_color_model(bundle, val_features)
    predictions.to_csv(TABLES_DIR / "color_hsv_knn_v1_val_predictions.csv", index=False)
    (TABLES_DIR / "color_hsv_knn_v1_val_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (TABLES_DIR / "color_hsv_knn_v1_val_report.txt").open("w", encoding="utf-8") as out:
        out.write(summary["note"] + "\n\n")
        for candy_type, colors in variants.items():
            group = predictions[predictions["type_reel"] == candy_type]
            out.write(f"=== {candy_type} (n={len(group)}) ===\n")
            out.write(classification_report(
                group["couleur_reelle"], group["couleur_predite"],
                labels=colors, zero_division=0,
            ))
            out.write("\n")
    figure = save_confusion_figures(bundle, predictions)
    errors_figure = save_error_examples(predictions)

    print("=== ACS — Classification des couleurs (HSV + KNN) ===")
    print(f"Apprentissage S01 : {len(train_features)} images")
    print(f"Validation S02 : {len(val_features)} images")
    print("Évaluation CONDITIONNELLE AU TYPE RÉEL (pas bout en bout)")
    print(f"Accuracy variantes : {summary['accuracy_variant_conditionnelle']:.4f}")
    print(f"Macro F1 variantes : {summary['macro_f1_variantes']:.4f}")
    print(f"Erreurs : {summary['errors']} | Abstentions : {summary['abstentions']}")
    if summary["errors"]:
        print("\n--- Erreurs de couleur ---")
        print(predictions.loc[~predictions["correct"], [
            "image_path", "type_reel", "couleur_reelle", "couleur_predite",
            "score_knn", "segmentation_status",
        ]].to_string(index=False))
    print("\nModèle :", MODEL_PATH)
    print("Métriques :", TABLES_DIR / "color_hsv_knn_v1_val_summary.json")
    print("Rapport :", TABLES_DIR / "color_hsv_knn_v1_val_report.txt")
    print("Prédictions :", TABLES_DIR / "color_hsv_knn_v1_val_predictions.csv")
    print("Figure :", figure)
    if errors_figure is not None:
        print("Figure des erreurs :", errors_figure)


def predict_single(image_path: Path, candy_type: str, model_path: Path = MODEL_PATH) -> None:
    # N'ouvrez que des fichiers joblib générés par vous-même : pickle n'est pas sûr.
    bundle = joblib.load(model_path)
    features = extract_features_file(
        image_path, max_side=bundle["max_side"],
        min_saturation=bundle["min_saturation"],
    )
    result = predict_color(bundle, candy_type, features)
    print(f"Type fourni : {candy_type}")
    print(f"Couleur prédite : {result['prediction']}")
    print(f"Méthode : {result['source']}")
    if math.isfinite(result["score"]):
        print(f"Score de vote KNN (non calibré) : {result['score']:.3f}")
    print("Caractéristiques HSV :", features)


def main():
    parser = argparse.ArgumentParser(description="ACS — Classification des couleurs HSV")
    parser.add_argument("--max-side", type=int, default=800)
    parser.add_argument("--min-saturation", type=int, default=45)
    parser.add_argument("--n-neighbors", type=int, default=3)
    parser.add_argument("--predict-image", type=Path, default=None)
    parser.add_argument("--type", type=str, default=None)
    args = parser.parse_args()
    if args.predict_image is not None:
        if args.type is None:
            parser.error("--predict-image exige --type (fourni manuellement)")
        predict_single(args.predict_image, args.type)
    else:
        run(max_side=args.max_side, min_saturation=args.min_saturation,
            n_neighbors=args.n_neighbors)


if __name__ == "__main__":
    main()
