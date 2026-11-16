from pathlib import Path
import sys

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from src.data.augmentation import (
    get_training_transform,
    get_evaluation_transform
)


# ==========================================
# Test des augmentations
# ==========================================

def test_augmentation():

    print("=== ACS - Test Data Augmentation ===")

    image = Image.new(
        "RGB",
        (400, 300),
        color=(100, 180, 70)
    )

    for method in ["center_crop", "padding"]:

        print(f"\nMéthode : {method}")

        train_transform = get_training_transform(
            method=method
        )

        eval_transform = get_evaluation_transform(
            method=method
        )

        # Test des transformations aléatoires
        torch.manual_seed(42)

        train_tensor = train_transform(image)

        assert train_tensor.shape == (3, 224, 224)

        assert train_tensor.dtype == torch.float32

        assert torch.isfinite(train_tensor).all()

        # Test de reproductibilité avec seed fixe
        torch.manual_seed(42)

        repeated_tensor = train_transform(image)

        assert torch.equal(
            train_tensor,
            repeated_tensor
        )

        # Test de déterminisme en évaluation
        eval_tensor_1 = eval_transform(image)

        eval_tensor_2 = eval_transform(image)

        assert torch.equal(
            eval_tensor_1,
            eval_tensor_2
        )

        print("Dimensions : OK")
        print("Valeurs finies : OK")
        print("Reproductibilité : OK")
        print("Validation déterministe : OK")

    print("\nTous les tests ont réussi !")


if __name__ == "__main__":
    test_augmentation()