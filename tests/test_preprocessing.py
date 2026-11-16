from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageChops


# ==========================================
# ACS - Configuration
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from src.data.preprocessing import (
    resize_center_crop,
    resize_with_padding
)


# ==========================================
# Création d'une image de test
# ==========================================

def create_test_image():

    """
    Crée une image rectangulaire avec
    un objet rouge sur fond blanc.
    """

    image = Image.new(
        "RGB",
        (1200, 600),
        color=(255, 255, 255)
    )

    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(
        (60, 250, 1140, 350),
        radius=30,
        fill=(220, 50, 50)
    )

    return image


# ==========================================
# Test 1 - Center Crop
# ==========================================

def test_center_crop():

    print("\n=== Test Center Crop ===")

    image = create_test_image()

    result = resize_center_crop(
        image,
        size=224
    )

    assert result.size == (224, 224), (
        "Dimensions incorrectes"
    )

    assert result.mode == "RGB", (
        "Mode RGB incorrect"
    )

    print("Dimensions :", result.size)
    print("Mode :", result.mode)

    print("Center Crop : OK")


# ==========================================
# Test 2 - Padding
# ==========================================

def test_padding():

    print("\n=== Test Padding ===")

    image = create_test_image()

    result = resize_with_padding(
        image,
        size=224
    )

    # Vérification des dimensions
    assert result.size == (224, 224)

    # Vérification du mode
    assert result.mode == "RGB"

    # Vérification des marges blanches
    assert result.getpixel((0, 0)) == (
        255, 255, 255
    )

    assert result.getpixel((223, 223)) == (
        255, 255, 255
    )

    # Vérification de la présence de l'objet
    center_pixel = result.getpixel((112, 112))

    assert center_pixel[0] > center_pixel[1], (
        "L'objet rouge n'est pas détecté"
    )

    # Vérification que les extrémités
    # de l'objet restent visibles
    left_pixel = result.getpixel((15, 112))
    right_pixel = result.getpixel((208, 112))

    assert left_pixel[0] > left_pixel[1], (
        "Extrémité gauche perdue"
    )

    assert right_pixel[0] > right_pixel[1], (
        "Extrémité droite perdue"
    )

    print("Dimensions :", result.size)
    print("Mode :", result.mode)
    print("Marges blanches : OK")
    print("Objet conservé : OK")

    print("Padding : OK")


# ==========================================
# Test 3 - Image originale
# ==========================================

def test_original_image():

    print("\n=== Test Image Originale ===")

    image = create_test_image()

    original = image.copy()

    resize_center_crop(image)

    resize_with_padding(image)

    difference = ImageChops.difference(
        image,
        original
    )

    assert difference.getbbox() is None, (
        "L'image originale a été modifiée"
    )

    print("Image originale conservée : OK")


# ==========================================
# Test 4 - Différentes résolutions
# ==========================================

def test_multiple_resolutions():

    print("\n=== Test Résolutions ===")

    resolutions = [
        (3000, 4000),
        (4000, 3000),
        (1920, 1080),
        (1080, 1920),
        (500, 500)
    ]

    for width, height in resolutions:

        image = Image.new(
            "RGB",
            (width, height),
            color="white"
        )

        crop = resize_center_crop(image)

        padding = resize_with_padding(image)

        assert crop.size == (224, 224)
        assert padding.size == (224, 224)

        print(
            f"{width}x{height} "
            "-> 224x224 : OK"
        )

    print("Toutes les résolutions : OK")


# ==========================================
# Programme principal
# ==========================================

def main():

    print("================================")
    print("ACS - Tests de prétraitement")
    print("================================")

    test_center_crop()

    test_padding()

    test_original_image()

    test_multiple_resolutions()

    print("\n================================")
    print("Tous les tests ont réussi !")
    print("================================")


if __name__ == "__main__":
    main()