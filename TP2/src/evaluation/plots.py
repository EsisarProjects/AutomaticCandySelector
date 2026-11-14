"""Visualisations pour l'évaluation du classificateur ACS."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.preprocessing import load_image, get_preprocessing_transform
from src.data.augmentation import denormalize_image


def plot_confusion_matrix(matrix, classes, output_path, dataset_name="validation S02"):
    """Matrice : lignes = étiquettes réelles, colonnes = prédictions."""
    matrix = np.asarray(matrix, dtype=int)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Nombre d'images")
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Classe prédite",
        ylabel="Classe réelle",
        title=f"ACS — Matrice de confusion ({dataset_name})",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    row_totals = matrix.sum(axis=1)
    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(len(classes)):
        for j in range(len(classes)):
            count = int(matrix[i, j])
            pct = (100 * count / row_totals[i]) if row_totals[i] else 0.0
            ax.text(
                j, i, f"{count}\n({pct:.1f} %)",
                ha="center", va="center",
                color="white" if count > threshold else "black",
                fontsize=10,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_prediction_errors(predictions, root, method, output_path,
                           max_errors=12, dataset_name="validation S02"):
    """Montre les photographies mal classées et leurs entrées CNN."""
    if max_errors < 1:
        raise ValueError("max_errors doit être >= 1")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors = predictions.loc[~predictions["correct"]].copy()
    errors = errors.sort_values("confidence", ascending=False).head(max_errors)

    if errors.empty:
        fig, ax = plt.subplots(figsize=(7, 2))
        ax.text(0.5, 0.5, f"Aucune erreur sur {dataset_name}",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return output_path

    transform = get_preprocessing_transform(method=method)
    fig, axes = plt.subplots(len(errors), 2, figsize=(10, 3.7 * len(errors)),
                             squeeze=False)

    for row_idx, (_, item) in enumerate(errors.iterrows()):
        path = Path(item["image_path"])
        if not path.is_absolute():
            path = Path(root) / path
        original = load_image(path)
        processed = denormalize_image(transform(original))

        axes[row_idx, 0].imshow(original)
        axes[row_idx, 0].set_title(
            f"Originale — réel : {item['type']} / {item['couleur']}", fontsize=10
        )
        axes[row_idx, 1].imshow(processed)
        axes[row_idx, 1].set_title(
            f"Entrée CNN — prédit : {item['prediction']} "
            f"({100 * item['confidence']:.1f} %)", fontsize=10
        )
        for ax in axes[row_idx]:
            ax.axis("off")

    fig.suptitle(f"ACS — Erreurs de classification ({dataset_name})", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path