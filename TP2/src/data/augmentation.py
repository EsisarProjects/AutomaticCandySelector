from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import torch

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.data.preprocessing import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_image,
    resize_center_crop,
    resize_with_padding,
    get_preprocessing_transform,
)


# ==========================================
# ACS - Configuration
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

DATASET_CSV = ROOT / "datasets/metadata/classification.csv"

OUTPUT_DIR = ROOT / "results/figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# Redimensionnement
# ==========================================

class ACSResize:
    """
    Redimensionnement adapté au projet ACS.

    Méthodes :
        center_crop
        padding
    """

    def __init__(self, method="center_crop"):

        if method not in ["center_crop", "padding"]:
            raise ValueError(
                f"Méthode inconnue : {method}"
            )

        self.method = method

    def __call__(self, image):

        if self.method == "center_crop":
            return resize_center_crop(
                image,
                size=IMAGE_SIZE
            )

        return resize_with_padding(
            image,
            size=IMAGE_SIZE
        )


# ==========================================
# Data Augmentation - Entraînement
# ==========================================

def get_training_transform(
    method="center_crop",
    enable_photometric=False
):
    """
    Construit les transformations utilisées
    exclusivement pendant l'entraînement.

    Paramètres :
        method :
            center_crop ou padding

        enable_photometric :
            Active les modifications légères
            de luminosité et de contraste.
    """

    augmentations = [

        # 1. Redimensionnement
        ACSResize(method),

        # 2. Symétrie horizontale aléatoire
        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        # 3. Rotation aléatoire
        transforms.RandomRotation(
            degrees=15,
            interpolation=InterpolationMode.BILINEAR,
            fill=(255, 255, 255)
        ),

        # 4. Translation et changement d'échelle
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.0),
            interpolation=InterpolationMode.BILINEAR,
            fill=(255, 255, 255)
        ),
    ]

    # ======================================
    # Transformations photométriques
    # ======================================

    if enable_photometric:

        augmentations.append(

            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.08,
                        contrast=0.08,
                        saturation=0.0,
                        hue=0.0
                    )
                ],
                p=0.4
            )
        )

    # ======================================
    # Conversion et normalisation
    # ======================================

    augmentations.extend([

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])

    return transforms.Compose(
        augmentations
    )


# ==========================================
# Transformations - Validation et Test
# ==========================================

def get_evaluation_transform(method="center_crop"):
    """
    Transformations déterministes utilisées
    pour la validation et le test.

    Aucune augmentation aléatoire.
    """

    return get_preprocessing_transform(
        method=method
    )


# ==========================================
# Dénormalisation pour visualisation
# ==========================================

def denormalize_image(tensor):
    """
    Transforme un tenseur normalisé
    en image visualisable.

    Format de sortie :
        H x W x C
    """

    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=tensor.dtype,
        device=tensor.device
    ).view(3, 1, 1)

    std = torch.tensor(
        IMAGENET_STD,
        dtype=tensor.dtype,
        device=tensor.device
    ).view(3, 1, 1)

    image = tensor * std + mean

    image = image.clamp(0, 1)

    return image.permute(1, 2, 0).cpu().numpy()


# ==========================================
# Chargement d'une image réelle
# ==========================================

def get_sample_image():

    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            "classification.csv introuvable. "
            "Exécutez preparation.py."
        )

    df = pd.read_csv(DATASET_CSV)

    if df.empty:
        raise ValueError(
            "Le dataset est vide."
        )

    for _, row in df.iterrows():

        image_path = ROOT / row["image_path"]

        if image_path.is_file():

            image = load_image(image_path)

            label = (
                f"{row['type']} - "
                f"{row['couleur']}"
            )

            return image, label

    raise FileNotFoundError(
        "Aucune image valide trouvée."
    )


# ==========================================
# Visualisation des augmentations
# ==========================================

def visualize_augmentations(
    method="center_crop",
    enable_photometric=False
):
    """
    Génère une figure contenant :
        - 1 image de référence
        - 8 transformations aléatoires
    """

    image, label = get_sample_image()

    train_transform = get_training_transform(
        method=method,
        enable_photometric=enable_photometric
    )

    eval_transform = get_evaluation_transform(
        method=method
    )

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(10, 10)
    )

    axes = axes.flatten()

    # Image de référence
    reference = eval_transform(image)

    axes[0].imshow(
        denormalize_image(reference)
    )

    axes[0].set_title(
        f"Référence\n{label}"
    )

    axes[0].axis("off")

    # Huit augmentations
    for i in range(1, 9):

        augmented = train_transform(image)

        axes[i].imshow(
            denormalize_image(augmented)
        )

        axes[i].set_title(
            f"Augmentation {i:02d}"
        )

        axes[i].axis("off")

    fig.suptitle(
        "ACS - Data Augmentation",
        fontsize=16
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR /
        "augmentation_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("=== ACS - Data Augmentation ===")

    print(f"Image : {label}")
    print(f"Méthode : {method}")

    print(
        "Modifications photométriques :",
        enable_photometric
    )

    print(f"Figure : {output_path}")

    print("Visualisation terminée !")


# ==========================================
# Programme principal
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        choices=["center_crop", "padding"],
        default="center_crop"
    )

    parser.add_argument(
        "--photometric",
        action="store_true",
        help="Activer brightness et contraste"
    )

    args = parser.parse_args()

    visualize_augmentations(
        method=args.method,
        enable_photometric=args.photometric
    )


if __name__ == "__main__":
    main()