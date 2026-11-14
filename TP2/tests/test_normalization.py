from pathlib import Path
import sys

import torch
from PIL import Image


# ==========================================
# Configuration
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from src.data.preprocessing import (
    get_preprocessing_transform,
    IMAGENET_MEAN,
    IMAGENET_STD
)


# ==========================================
# Test de normalisation
# ==========================================

def test_normalization():

    print("=== ACS - Test de normalisation ===")

    # Image artificielle rouge
    image = Image.new(
        "RGB",
        (400, 300),
        color=(255, 0, 0)
    )

    # Valeurs théoriques après normalisation
    expected_red = torch.tensor([
        (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0],
        (0.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1],
        (0.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
    ])

    for method in ["center_crop", "padding"]:

        print(f"\nMéthode : {method}")

        transform = get_preprocessing_transform(method)

        tensor = transform(image)

        print("Dimensions :", tuple(tensor.shape))
        print("Type :", tensor.dtype)

        print("Minimum :", tensor.min().item())
        print("Maximum :", tensor.max().item())

        # Vérification des dimensions
        assert tensor.shape == (3, 224, 224)

        # Vérification du type
        assert tensor.dtype == torch.float32

        # Vérification des valeurs
        assert torch.isfinite(tensor).all()

        # Vérification du pixel central
        center_pixel = tensor[:, 112, 112]

        assert torch.allclose(
            center_pixel,
            expected_red,
            atol=1e-5
        )

        print("Test : OK")

    print("\nTous les tests ont réussi !")


if __name__ == "__main__":
    test_normalization()