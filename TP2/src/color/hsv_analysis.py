"""ACS — Étape 1.6.1 : analyse des couleurs HSV (train S01 uniquement).

Exécution depuis la racine du projet :
    python -m src.color.hsv_analysis
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.preprocessing import load_image

ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = ROOT / "datasets/metadata/splits/classification_train.csv"
TABLES_DIR = ROOT / "results/tables"
FIGURES_DIR = ROOT / "results/figures"


def resize_for_analysis(rgb: np.ndarray, max_side: int = 800) -> np.ndarray:
    """Réduit la résolution pour éviter de traiter les photos 3000×4000."""
    if max_side < 32:
        raise ValueError("max_side doit être >= 32")
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return rgb.copy()
    return cv2.resize(
        rgb,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def segment_colored_candy(
    rgb: np.ndarray,
    min_saturation: int = 45,
    min_value: int = 40,
    min_area: int = 40,
) -> np.ndarray:
    """Masque (0/255) d'un objet *coloré* sur fond blanc/gris.

    Limite connue : un bonbon blanc sur fond blanc n'est généralement pas
    détectable par cette méthode. Aucune classe réelle n'est utilisée.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb doit être un tableau uint8 H×W×3")
    if not (0 <= min_saturation <= 255 and 0 <= min_value <= 255):
        raise ValueError("Seuils HSV hors de [0,255]")

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    candidate = cv2.inRange(
        hsv,
        (0, min_saturation, min_value),
        (179, 255, 255),
    )
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel_open)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(
        candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    h, w = rgb.shape[:2]
    best = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        cx, cy = x + bw / 2, y + bh / 2
        # Biais faible vers le centre : les bonbons sont placés au centre.
        distance = math.hypot((cx - w / 2) / (w / 2), (cy - h / 2) / (h / 2))
        score = area / (1.0 + distance)
        if score > best_score:
            best_score = score
            best = contour

    mask = np.zeros((h, w), dtype=np.uint8)
    if best is not None:
        cv2.drawContours(mask, [best], -1, 255, thickness=cv2.FILLED)
    return mask


def circular_hue_mean(h_values: np.ndarray) -> float:
    """Moyenne circulaire de la teinte OpenCV, intervalle [0,180)."""
    arr = np.asarray(h_values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    angles = arr * (2 * np.pi / 180.0)
    vector = np.mean(np.exp(1j * angles))
    if abs(vector) < 1e-8:
        return float("nan")
    return float((np.angle(vector) % (2 * np.pi)) * 180.0 / (2 * np.pi))


def compute_hsv_features(rgb: np.ndarray, mask: np.ndarray) -> dict:
    """Mesure la teinte circulaire et les médianes S/V dans le masque."""
    if mask.shape != rgb.shape[:2]:
        raise ValueError("Dimensions du masque incompatibles")
    selected = mask > 0
    fraction = float(selected.mean())
    if not np.any(selected):
        return {
            "status": "non_detecte",
            "mask_fraction": 0.0,
            "hue_mean": float("nan"),
            "s_median": float("nan"),
            "v_median": float("nan"),
        }
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    pixels = hsv[selected]
    return {
        "status": "detecte",
        "mask_fraction": fraction,
        "hue_mean": circular_hue_mean(pixels[:, 0]),
        "s_median": float(np.median(pixels[:, 1])),
        "v_median": float(np.median(pixels[:, 2])),
    }


def read_train_metadata(path: Path = TRAIN_CSV) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV introuvable : {path}")
    df = pd.read_csv(path, dtype=str)
    columns = {"image_path", "type", "couleur", "session_id"}
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    if df.empty:
        raise ValueError("CSV d'entraînement vide")
    if df[list(columns)].isna().any().any():
        raise ValueError("Métadonnées manquantes")
    if set(df["session_id"]) != {"S01"}:
        raise ValueError("Cette première analyse est réservée à S01 (train)")
    return df


def save_segmentation_examples(examples: dict) -> Path:
    """Affiche une photo originale et son masque pour chaque variante."""
    keys = sorted(examples)
    columns = 4
    rows = math.ceil(len(keys) / columns)
    fig, axes = plt.subplots(rows, columns * 2, figsize=(16, 3.5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, key in enumerate(keys):
        rgb, mask = examples[key]
        row, pair = divmod(i, columns)
        a = axes[row, pair * 2]
        b = axes[row, pair * 2 + 1]
        a.imshow(rgb)
        a.set_title(f"{key[0]} / {key[1]}\noriginale", fontsize=9)
        if np.any(mask):
            isolated = np.full_like(rgb, 255)
            isolated[mask > 0] = rgb[mask > 0]
            b.imshow(isolated)
            b.set_title("masque HSV", fontsize=9)
        else:
            b.imshow(rgb)
            b.set_title("NON DÉTECTÉ", fontsize=9)
    fig.suptitle("ACS — Segmentation HSV : inspection des variantes (S01)")
    fig.tight_layout()
    output = FIGURES_DIR / "hsv_train_segmentation_examples.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def save_hsv_scatter(df: pd.DataFrame) -> Path:
    valid = df[df["status"] == "detecte"]
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, group in valid.groupby("couleur", sort=True):
        ax.scatter(group["hue_mean"], group["s_median"], label=name, alpha=0.8)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 255)
    ax.set_xlabel("Teinte moyenne circulaire H (OpenCV, 0–179)")
    ax.set_ylabel("Saturation médiane S (0–255)")
    ax.set_title("ACS — Caractéristiques HSV par couleur (train S01)")
    ax.grid(alpha=0.25)
    ax.legend(title="Couleur", loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    output = FIGURES_DIR / "hsv_train_hue_saturation.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["type", "couleur"], sort=True)
        .agg(
            n_images=("status", "size"),
            n_detected=("status", lambda x: int((x == "detecte").sum())),
            hue_mean=("hue_mean", circular_hue_mean),
            s_median=("s_median", "median"),
            v_median=("v_median", "median"),
            mask_fraction_median=("mask_fraction", "median"),
        )
        .reset_index()
    )
    return summary


def run(min_saturation: int = 45, max_side: int = 800) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = read_train_metadata()
    rows = []
    examples = {}

    for _, item in df.iterrows():
        path = Path(item["image_path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(path)
        rgb = np.asarray(load_image(path), dtype=np.uint8)
        rgb = resize_for_analysis(rgb, max_side=max_side)
        mask = segment_colored_candy(rgb, min_saturation=min_saturation)
        features = compute_hsv_features(rgb, mask)
        rows.append({
            "image_path": item["image_path"],
            "type": item["type"],
            "couleur": item["couleur"],
            "session_id": item["session_id"],
            **features,
        })
        key = (item["type"], item["couleur"])
        if key not in examples:
            examples[key] = (rgb, mask)

    measurements = pd.DataFrame(rows)
    summary = summarize(measurements)
    raw_output = TABLES_DIR / "hsv_train_features.csv"
    summary_output = TABLES_DIR / "hsv_train_summary.csv"
    measurements.to_csv(raw_output, index=False)
    summary.to_csv(summary_output, index=False)
    figure_examples = save_segmentation_examples(examples)
    figure_scatter = save_hsv_scatter(measurements)

    print("=== ACS — Analyse HSV (train S01) ===")
    print(f"Images étudiées : {len(measurements)}")
    print(f"Objets colorés détectés : {(measurements['status'] == 'detecte').sum()}")
    print(f"Non détectés : {(measurements['status'] == 'non_detecte').sum()}")
    print("\nRésumé par variante :")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nMesures : {raw_output}")
    print(f"Résumé : {summary_output}")
    print(f"Figure segmentation : {figure_examples}")
    print(f"Figure HSV : {figure_scatter}")

def main() -> None:
    parser = argparse.ArgumentParser(description="ACS — Analyse HSV sur S01")
    parser.add_argument("--min-saturation", type=int, default=45)
    parser.add_argument("--max-side", type=int, default=800)
    args = parser.parse_args()
    run(min_saturation=args.min_saturation, max_side=args.max_side)


if __name__ == "__main__":
    main()