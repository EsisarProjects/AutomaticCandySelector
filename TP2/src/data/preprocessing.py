from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageOps, ImageDraw

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# ==========================================
# ACS - Configuration
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

DATASET_CSV = ROOT / "datasets/metadata/classification.csv"

OUTPUT_DIR = ROOT / "results/figures"

IMAGE_SIZE = 224

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# Normalisation ImageNet
# ==========================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]

IMAGENET_STD = [0.229, 0.224, 0.225]


# ==========================================
# Chargement des images
# ==========================================

def load_image(image_path):
    """
    Charge une image, corrige son orientation
    EXIF et la convertit en RGB.
    """

    with Image.open(image_path) as img:

        image = ImageOps.exif_transpose(img)

        image = image.convert("RGB")

    return image


# ==========================================
# Méthode A - Resize + Center Crop
# ==========================================

def resize_center_crop(image, size=IMAGE_SIZE):
    """
    Redimensionne l'image en conservant
    son ratio, puis applique un recadrage
    central.

    Sortie : (size, size)
    """

    transform = transforms.Compose([
        transforms.Resize(
            256,
            interpolation=InterpolationMode.BILINEAR
        ),

        transforms.CenterCrop(size)
    ])

    return transform(image)


# ==========================================
# Méthode B - Resize with Padding
# ==========================================

def resize_with_padding(image, size=IMAGE_SIZE):
    """
    Conserve toutes les parties de l'image
    et ajoute des marges blanches.
    """

    resized = ImageOps.contain(
        image,
        (size, size),
        method=Image.Resampling.BILINEAR
    )

    canvas = Image.new(
        "RGB",
        (size, size),
        color=(255, 255, 255)
    )

    x = (size - resized.width) // 2
    y = (size - resized.height) // 2

    canvas.paste(resized, (x, y))

    return canvas


# ==========================================
# Prétraitement pour les réseaux CNN
# ==========================================

def get_preprocessing_transform(method="center_crop"):
    """
    Construit le pipeline de prétraitement
    pour les réseaux CNN pré-entraînés.

    Étapes :
        1. Redimensionnement
        2. Conversion en tenseur
        3. Normalisation ImageNet

    Paramètres :
        method : "center_crop" ou "padding"

    Retour :
        torchvision.transforms.Compose
    """

    if method == "center_crop":

        resize_transform = transforms.Lambda(
            lambda image: resize_center_crop(
                image,
                size=IMAGE_SIZE
            )
        )

    elif method == "padding":

        resize_transform = transforms.Lambda(
            lambda image: resize_with_padding(
                image,
                size=IMAGE_SIZE
            )
        )

    else:
        raise ValueError(
            f"Méthode inconnue : {method}"
        )

    transform = transforms.Compose([

        # 1. Redimensionnement
        resize_transform,

        # 2. Conversion PIL -> Tensor [0, 1]
        transforms.ToTensor(),

        # 3. Normalisation ImageNet
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )

    ])

    return transform


# ==========================================
# Image synthétique de démonstration
# ==========================================

def create_demo_image():
    """
    Crée une image artificielle permettant
    de comparer les méthodes de resize.
    """

    image = Image.new(
        "RGB",
        (1200, 600),
        color="white"
    )

    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(
        (100, 270, 1100, 330),
        radius=25,
        fill=(220, 50, 50)
    )

    return image


# ==========================================
# Recherche d'une image du dataset
# ==========================================

def get_sample_image():
    """
    Charge une photographie du dataset ACS.

    Si aucune photographie n'est disponible,
    utilise une image synthétique.
    """

    if DATASET_CSV.exists():

        df = pd.read_csv(DATASET_CSV)

        if not df.empty:

            for path in df["image_path"]:

                image_path = ROOT / path

                if image_path.is_file():

                    return load_image(image_path)

    print(
        "Aucune photographie disponible. "
        "Utilisation d'une image synthétique."
    )

    return create_demo_image()


# ==========================================
# Comparaison visuelle des méthodes
# ==========================================

def compare_methods(image):
    """
    Compare visuellement :
        - Image originale
        - Center Crop
        - Padding
    """

    image_crop = resize_center_crop(
        image,
        IMAGE_SIZE
    )

    image_padding = resize_with_padding(
        image,
        IMAGE_SIZE
    )

    images = [
        image,
        image_crop,
        image_padding
    ]

    titles = [
        f"Image originale\n{image.size}",
        f"Center Crop\n{image_crop.size}",
        f"Padding\n{image_padding.size}"
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13, 5)
    )

    for ax, img, title in zip(
        axes,
        images,
        titles
    ):

        ax.imshow(img)

        ax.set_title(title)

        ax.axis("off")

    fig.suptitle(
        "ACS - Comparaison des méthodes de redimensionnement",
        fontsize=13
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR / "preprocessing_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Figure enregistrée : {output_path}")


# ==========================================
# Vérification de la normalisation
# ==========================================

def verify_normalization(image, method="center_crop"):
    """
    Applique le prétraitement complet
    et affiche les informations du tenseur.
    """

    transform = get_preprocessing_transform(method)

    tensor = transform(image)

    print("\n=== Vérification de la normalisation ===")

    print(f"Méthode : {method}")

    print(f"Dimensions : {tuple(tensor.shape)}")

    print(f"Type : {tensor.dtype}")

    print(f"Minimum : {tensor.min().item():.4f}")

    print(f"Maximum : {tensor.max().item():.4f}")

    print(f"Moyenne : {tensor.mean().item():.4f}")

    print(f"Écart-type : {tensor.std().item():.4f}")

    # Vérification des dimensions
    assert tensor.shape == (
        3,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    # Vérification du type
    assert tensor.dtype == torch.float32

    # Vérification des valeurs
    assert torch.isfinite(tensor).all()

    print("Normalisation : OK")

    return tensor


# ==========================================
# Programme principal
# ==========================================

def main():

    print("===================================")
    print("ACS - Prétraitement des images")
    print("===================================")

    # Chargement d'une image
    image = get_sample_image()

    print(f"\nImage originale : {image.size}")

    # Comparaison des redimensionnements
    compare_methods(image)

    # Normalisation avec Center Crop
    tensor_crop = verify_normalization(
        image,
        method="center_crop"
    )

    # Normalisation avec Padding
    tensor_padding = verify_normalization(
        image,
        method="padding"
    )

    # Simulation d'un batch
    batch = tensor_crop.unsqueeze(0)

    print("\n=== Simulation d'un batch ===")

    print(f"Dimensions : {tuple(batch.shape)}")

    print("\nPrétraitement terminé avec succès !")


if __name__ == "__main__":
    main()