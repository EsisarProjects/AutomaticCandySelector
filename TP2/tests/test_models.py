from pathlib import Path
import sys

import torch
import yaml


# ==========================================
# ACS - Configuration
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from src.classification.models import create_model


# ==========================================
# Test de l'architecture
# ==========================================

def test_model():

    print("=== ACS - Test MobileNetV3 ===")

    with open(
        ROOT / "configs/classes.yaml",
        "r",
        encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)

    num_classes = len(config["types"])

    model = create_model(
        num_classes=num_classes,
        pretrained=False,
        mode="linear_probe"
    )

    # Vérification des sorties
    assert model.classifier[-1].out_features == num_classes

    # Vérification du gel des paramètres
    trainable_parameters = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
    ]

    assert trainable_parameters == [
        "classifier.3.weight",
        "classifier.3.bias"
    ]

    print("Paramètres entraînables :")
    print(trainable_parameters)

    # Vérification de l'inférence
    model.eval()

    images = torch.randn(2, 3, 224, 224)

    with torch.inference_mode():
        outputs = model(images)

    assert outputs.shape == (2, num_classes)

    assert torch.isfinite(outputs).all()

    print("Dimensions de sortie :", tuple(outputs.shape))

    print("Architecture : OK")
    print("Feature Extraction : OK")
    print("Inférence : OK")


if __name__ == "__main__":
    test_model()