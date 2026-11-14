from pathlib import Path
import sys

import pandas as pd
import torch
import yaml


# ==========================================
# ACS - Configuration
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders


# ==========================================
# Test du pipeline complet
# ==========================================

def test_data_pipeline():

    print("=== ACS - Test du pipeline ===")

    with open(
        ROOT / "configs/classes.yaml",
        "r",
        encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)

    expected_classes = config["types"]

    # ======================================
    # Création des DataLoaders
    # ======================================

    train_loader, val_loader, test_loader = (
        create_dataloaders(
            batch_size=4,
            num_workers=0,
            method="center_crop",
            seed=42
        )
    )

    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

    # ======================================
    # Vérification des datasets
    # ======================================

    for split_name, loader in loaders.items():

        dataset = loader.dataset

        print(f"\n=== {split_name.upper()} ===")

        assert len(dataset) > 0

        assert dataset.classes == expected_classes

        # Comparaison avec le fichier CSV
        csv_path = (
            ROOT
            / "datasets/metadata/splits"
            / f"classification_{split_name}.csv"
        )

        df = pd.read_csv(csv_path)

        assert len(dataset) == len(df)

        # Vérification du premier échantillon
        image, label = dataset[0]

        assert image.shape == (3, 224, 224)

        assert image.dtype == torch.float32

        assert label.dtype == torch.long

        assert torch.isfinite(image).all()

        assert 0 <= label.item() < len(expected_classes)

        print("Images :", len(dataset))
        print("Batches :", len(loader))
        print("Dimensions :", tuple(image.shape))
        print("Étiquette :", label.item())

        # ==================================
        # Vérification d'un batch
        # ==================================

        images, labels = next(iter(loader))

        assert images.ndim == 4

        assert images.shape[1:] == (3, 224, 224)

        assert len(images) <= 4

        assert labels.shape == (len(images),)

        assert labels.dtype == torch.long

        assert torch.isfinite(images).all()

        print("Batch :", tuple(images.shape))
        print("Test : OK")

    # ======================================
    # Vérification du déterminisme
    # ======================================

    val_dataset = val_loader.dataset

    image_1, label_1 = val_dataset[0]
    image_2, label_2 = val_dataset[0]

    assert torch.equal(image_1, image_2)

    assert torch.equal(label_1, label_2)

    print("\nValidation déterministe : OK")

    # ======================================
    # Vérification des sessions
    # ======================================

    expected_sessions = {
        "train": "S01",
        "val": "S02",
        "test": "S03"
    }

    for split_name, loader in loaders.items():

        sessions = set(
            loader.dataset.df["session_id"]
        )

        assert sessions == {
            expected_sessions[split_name]
        }

    print("Séparation des sessions : OK")

    # ======================================
    # Vérification des classes entraînées
    # ======================================

    train_distribution = (
        train_loader.dataset.get_class_distribution()
    )

    for class_name, count in train_distribution.items():

        assert count > 0, (
            f"Classe absente de l'entraînement : {class_name}"
        )

    print("Toutes les classes sont représentées : OK")

    print("\n================================")
    print("Tous les tests ont réussi !")
    print("================================")


if __name__ == "__main__":
    test_data_pipeline()