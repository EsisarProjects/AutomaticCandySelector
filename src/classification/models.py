from pathlib import Path
import argparse

import torch
import torch.nn as nn
import yaml

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)


# ==========================================
# ACS - Configuration
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "configs/classes.yaml"


# ==========================================
# Chargement des classes
# ==========================================

def load_classes():

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    classes = config["types"]

    return classes


# ==========================================
# Construction du modèle
# ==========================================

def create_model(
    num_classes,
    pretrained=True,
    mode="linear_probe"
):
    """
    Construit un classificateur MobileNetV3 Small.

    Paramètres
    ----------
    num_classes : int
        Nombre de classes de classification.

    pretrained : bool
        Utilisation des poids ImageNet.

    mode : str
        linear_probe : seule la dernière couche est entraînée.
        fine_tune    : tous les paramètres sont entraînables.

    Retour
    ------
    model : nn.Module
        MobileNetV3 adaptée au projet ACS.
    """

    if num_classes < 2:
        raise ValueError(
            "Le nombre de classes doit être supérieur à 1."
        )

    if mode not in ["linear_probe", "fine_tune"]:
        raise ValueError(
            f"Mode d'entraînement inconnu : {mode}"
        )

    # ======================================
    # Chargement des poids
    # ======================================

    if pretrained:

        weights = (
            MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

    else:

        weights = None

    model = mobilenet_v3_small(
        weights=weights
    )

    # ======================================
    # Feature Extraction
    # ======================================

    if mode == "linear_probe":

        # Gel de tous les paramètres existants
        for parameter in model.parameters():
            parameter.requires_grad = False

    # ======================================
    # Remplacement de la dernière couche
    # ======================================

    in_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        in_features,
        num_classes
    )

    # La nouvelle couche est entraînable
    # par défaut.

    return model


# ==========================================
# Statistiques du modèle
# ==========================================

def model_statistics(model):

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    frozen_params = total_params - trainable_params

    print("\n=== Statistiques du modèle ===")

    print(f"Paramètres totaux : {total_params:,}")

    print(f"Paramètres entraînables : {trainable_params:,}")

    print(f"Paramètres gelés : {frozen_params:,}")

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params
    }


# ==========================================
# Test du modèle
# ==========================================

def test_model(model, device="cpu"):

    print("\n=== Test du modèle ===")

    model = model.to(device)

    model.eval()

    # Batch artificiel de deux images RGB
    images = torch.randn(
        2, 3, 224, 224,
        device=device
    )

    with torch.inference_mode():
        outputs = model(images)

    print("Entrée :", tuple(images.shape))

    print("Sortie :", tuple(outputs.shape))

    print("Type :", outputs.dtype)

    print(
        "Valeurs finies :",
        torch.isfinite(outputs).all().item()
    )

    assert outputs.shape == (
        2,
        model.classifier[-1].out_features
    )

    assert torch.isfinite(outputs).all()

    print("\nTest du modèle : OK")


# ==========================================
# Programme principal
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Charger les poids pré-entraînés ImageNet"
    )

    parser.add_argument(
        "--mode",
        choices=["linear_probe", "fine_tune"],
        default="linear_probe"
    )

    args = parser.parse_args()

    print("===================================")
    print("ACS - MobileNetV3 Small")
    print("===================================")

    classes = load_classes()

    print("\nClasses :", classes)

    print("Nombre de classes :", len(classes))

    print("Poids pré-entraînés :", args.pretrained)

    print("Mode :", args.mode)

    if not args.pretrained:
        print(
            "\nATTENTION : modèle sans poids pré-entraînés. "
            "Utilisation réservée aux tests d'architecture."
        )

    # Création du modèle
    model = create_model(
        num_classes=len(classes),
        pretrained=args.pretrained,
        mode=args.mode
    )

    # Statistiques
    model_statistics(model)

    # Test sur CPU
    test_model(
        model,
        device="cpu"
    )

    print("\nModèle créé avec succès !")


if __name__ == "__main__":
    main()