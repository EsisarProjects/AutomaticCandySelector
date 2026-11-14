from pathlib import Path
from collections import Counter

import pandas as pd
import torch
import yaml

from torch.utils.data import Dataset, DataLoader

from src.data.preprocessing import load_image

from src.data.augmentation import (
    get_training_transform,
    get_evaluation_transform
)


# ==========================================
# ACS - Configuration
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "configs/classes.yaml"

TRAINING_CONFIG_PATH = ROOT / "configs/classification.yaml"

SPLITS_DIR = ROOT / "datasets/metadata/splits"

SPLITS = ("train", "val", "test")


# ==========================================
# Chargement de la configuration
# ==========================================

def load_yaml(path):

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# ==========================================
# ACS - Dataset personnalisé
# ==========================================

class ACSDataset(Dataset):
    """
    Dataset de classification des bonbons ACS.

    Chaque échantillon contient :
        - Une image RGB transformée
        - Un indice numérique correspondant au type

    Les couleurs restent disponibles dans
    les métadonnées pour les futures expériences.
    """

    def __init__(
        self,
        csv_path,
        transform=None,
        classes_path=CONFIG_PATH
    ):

        self.csv_path = Path(csv_path)

        self.transform = transform

        # ==================================
        # Chargement des classes
        # ==================================

        config = load_yaml(classes_path)

        self.classes = config["types"]

        self.class_to_idx = {
            name: idx
            for idx, name in enumerate(self.classes)
        }

        self.idx_to_class = {
            idx: name
            for name, idx in self.class_to_idx.items()
        }

        # ==================================
        # Chargement des métadonnées
        # ==================================

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Fichier CSV introuvable : {self.csv_path}"
            )

        self.df = pd.read_csv(
            self.csv_path,
            dtype=str
        )

        if self.df.empty:
            raise ValueError(
                f"Dataset vide : {self.csv_path}"
            )

        required_columns = [
            "image_id",
            "image_path",
            "type",
            "couleur",
            "session_id",
            "sha256"
        ]

        missing = [
            column for column in required_columns
            if column not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Colonnes manquantes : {missing}"
            )

        if self.df[required_columns].isna().any().any():
            raise ValueError(
                "Certaines métadonnées sont absentes."
            )

        # ==================================
        # Validation des étiquettes
        # ==================================

        valid_variants = {
            (bonbon_type, couleur)
            for bonbon_type, couleurs
            in config["variantes"].items()
            for couleur in couleurs
        }

        invalid_variants = [
            (row["type"], row["couleur"])
            for _, row in self.df.iterrows()
            if (row["type"], row["couleur"])
            not in valid_variants
        ]

        if invalid_variants:
            raise ValueError(
                f"Variantes invalides : {invalid_variants[:5]}"
            )

        # ==================================
        # Vérification des fichiers
        # ==================================

        self.image_paths = []

        for relative_path in self.df["image_path"]:

            path = Path(relative_path)

            if not path.is_absolute():
                path = ROOT / path

            if not path.is_file():
                raise FileNotFoundError(
                    f"Image introuvable : {path}"
                )

            self.image_paths.append(path)

        # ==================================
        # Conversion des étiquettes
        # ==================================

        self.labels = [
            self.class_to_idx[bonbon_type]
            for bonbon_type in self.df["type"]
        ]

    # ======================================
    # Nombre d'échantillons
    # ======================================

    def __len__(self):

        return len(self.df)

    # ======================================
    # Chargement d'un échantillon
    # ======================================

    def __getitem__(self, index):

        # Récupération du chemin
        image_path = self.image_paths[index]

        # Chargement de l'image RGB
        image = load_image(image_path)

        # Application du prétraitement
        if self.transform is not None:
            image = self.transform(image)

        # Étiquette numérique
        label = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return image, label

    # ======================================
    # Statistiques des classes
    # ======================================

    def get_class_distribution(self):

        counts = Counter(self.df["type"])

        return {
            class_name: counts.get(class_name, 0)
            for class_name in self.classes
        }


# ==========================================
# Création des DataLoaders
# ==========================================

def create_dataloaders(
    batch_size=4,
    num_workers=0,
    method="center_crop",
    enable_photometric=False,
    seed=42
):
    """
    Crée trois DataLoaders :

        train_loader
        val_loader
        test_loader

    Le Data Augmentation est appliqué
    uniquement au jeu d'entraînement.
    """

    # ======================================
    # Transformations
    # ======================================

    train_transform = get_training_transform(
        method=method,
        enable_photometric=enable_photometric
    )

    eval_transform = get_evaluation_transform(
        method=method
    )

    # ======================================
    # Chargement des datasets
    # ======================================

    train_dataset = ACSDataset(
        csv_path=SPLITS_DIR / "classification_train.csv",
        transform=train_transform
    )

    val_dataset = ACSDataset(
        csv_path=SPLITS_DIR / "classification_val.csv",
        transform=eval_transform
    )

    test_dataset = ACSDataset(
        csv_path=SPLITS_DIR / "classification_test.csv",
        transform=eval_transform
    )

    # ======================================
    # Vérification des classes
    # ======================================

    datasets = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }

    for name, dataset in datasets.items():

        if dataset.class_to_idx != train_dataset.class_to_idx:
            raise ValueError(
                f"Mapping des classes incohérent : {name}"
            )

    # ======================================
    # Vérification des chevauchements
    # ======================================

    for i, name_a in enumerate(SPLITS):

        dataset_a = datasets[name_a]

        ids_a = set(dataset_a.df["image_id"])
        hashes_a = set(dataset_a.df["sha256"])

        for name_b in SPLITS[i + 1:]:

            dataset_b = datasets[name_b]

            ids_b = set(dataset_b.df["image_id"])
            hashes_b = set(dataset_b.df["sha256"])

            if not ids_a.isdisjoint(ids_b):
                raise ValueError(
                    f"Images communes : {name_a}/{name_b}"
                )

            if not hashes_a.isdisjoint(hashes_b):
                raise ValueError(
                    f"Doublons exacts : {name_a}/{name_b}"
                )

    # ======================================
    # Générateur reproductible
    # ======================================

    generator = torch.Generator()

    generator.manual_seed(seed)

    # ======================================
    # DataLoaders
    # ======================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )

    return train_loader, val_loader, test_loader


# ==========================================
# Programme principal
# ==========================================

def main():

    print("===================================")
    print("ACS - Dataset et DataLoader")
    print("===================================")

    config = load_yaml(TRAINING_CONFIG_PATH) or {}

    training_config = config.get("training", {})

    batch_size = training_config.get(
        "batch_size", 4
    )

    num_workers = training_config.get(
        "num_workers", 0
    )

    seed = training_config.get(
        "seed", 42
    )

    method = training_config.get(
        "preprocessing_method", "center_crop"
    )

    train_loader, val_loader, test_loader = (
        create_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers,
            method=method,
            seed=seed
        )
    )

    loaders = {
        "Train": train_loader,
        "Validation": val_loader,
        "Test": test_loader
    }

    # ======================================
    # Affichage des statistiques
    # ======================================

    for name, loader in loaders.items():

        dataset = loader.dataset

        print(f"\n=== {name} ===")

        print("Images :", len(dataset))

        print("Batches :", len(loader))

        print("Distribution :")

        for class_name, count in (
            dataset.get_class_distribution().items()
        ):
            print(f"  {class_name}: {count}")

    # ======================================
    # Vérification d'un batch réel
    # ======================================

    images, labels = next(iter(train_loader))

    print("\n=== Premier batch ===")

    print("Images :", tuple(images.shape))

    print("Labels :", tuple(labels.shape))

    print("Type images :", images.dtype)

    print("Type labels :", labels.dtype)

    print("Labels numériques :", labels.tolist())

    classes = train_loader.dataset.idx_to_class

    print(
        "Labels textuels :",
        [classes[label.item()] for label in labels]
    )

    print("\nDataLoaders créés avec succès !")


if __name__ == "__main__":
    main()