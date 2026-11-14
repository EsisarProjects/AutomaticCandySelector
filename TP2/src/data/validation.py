from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from src.data.dataset import create_dataloaders

from src.data.preprocessing import load_image

from src.data.augmentation import denormalize_image


# ==========================================
# ACS - Configuration
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "configs/classification.yaml"

FIGURES_DIR = ROOT / "results/figures"

TABLES_DIR = ROOT / "results/tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# Chargement de la configuration
# ==========================================

def load_config():

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


# ==========================================
# Validation des ensembles
# ==========================================

def validate_datasets(loaders, config):

    print("\n=== Validation des datasets ===")

    report = []

    # Vérification des ensembles
    # et de leurs sessions d'acquisition

    for split_name, loader in loaders.items():

        dataset = loader.dataset

        expected_sessions = set(
            config["split"][split_name]
        )

        actual_sessions = set(
            dataset.df["session_id"]
        )

        assert actual_sessions.issubset(expected_sessions), (
            f"Sessions incorrectes dans {split_name}"
        )

        # Vérification des images
        for image_path in dataset.image_paths:

            assert image_path.is_file(), (
                f"Image introuvable : {image_path}"
            )

        # Vérification des classes
        for label in dataset.labels:

            assert 0 <= label < len(dataset.classes)

        # Distribution des classes
        distribution = dataset.get_class_distribution()

        print(f"\n{split_name.upper()}")

        print(f"Images : {len(dataset)}")
        print(f"Batches : {len(loader)}")
        print(f"Sessions : {sorted(actual_sessions)}")

        print("Distribution des classes :")

        for class_name, count in distribution.items():

            print(f"  {class_name}: {count}")

        if split_name == "train":

            for class_name, count in distribution.items():

                assert count > 0, (
                    f"Classe absente du train : {class_name}"
                )

        report.append(
            f"{split_name}: {len(dataset)} images, "
            f"{len(loader)} batches"
        )

    # ======================================
    # Vérification des chevauchements
    # ======================================

    split_names = list(loaders.keys())

    for i, name_a in enumerate(split_names):

        dataset_a = loaders[name_a].dataset

        ids_a = set(dataset_a.df["image_id"])
        hashes_a = set(dataset_a.df["sha256"])
        sessions_a = set(dataset_a.df["session_id"])

        for name_b in split_names[i + 1:]:

            dataset_b = loaders[name_b].dataset

            ids_b = set(dataset_b.df["image_id"])
            hashes_b = set(dataset_b.df["sha256"])
            sessions_b = set(dataset_b.df["session_id"])

            assert ids_a.isdisjoint(ids_b), (
                f"Images communes : {name_a}/{name_b}"
            )

            assert hashes_a.isdisjoint(hashes_b), (
                f"Doublons exacts : {name_a}/{name_b}"
            )

            assert sessions_a.isdisjoint(sessions_b), (
                f"Sessions communes : {name_a}/{name_b}"
            )

    print("\nAucun chevauchement détecté : OK")

    return report


# ==========================================
# Validation des batches
# ==========================================

def validate_batches(loaders):

    print("\n=== Validation des batches ===")

    for split_name, loader in loaders.items():

        images, labels = next(iter(loader))

        print(f"\n{split_name.upper()}")

        print("Images :", tuple(images.shape))
        print("Labels :", tuple(labels.shape))

        # Vérification des dimensions
        assert images.ndim == 4

        assert images.shape[1:] == (
            3, 224, 224
        )

        assert labels.ndim == 1

        assert len(images) == len(labels)

        # Vérification des types
        assert images.dtype == torch.float32

        assert labels.dtype == torch.long

        # Vérification numérique
        assert torch.isfinite(images).all()

        assert (labels >= 0).all()

        assert (
            labels < len(loader.dataset.classes)
        ).all()

        print("Dimensions : OK")
        print("Types : OK")
        print("Valeurs numériques : OK")

    # ======================================
    # Déterminisme de la validation
    # ======================================

    for split_name in ["val", "test"]:

        dataset = loaders[split_name].dataset

        image_1, label_1 = dataset[0]
        image_2, label_2 = dataset[0]

        assert torch.equal(image_1, image_2)

        assert torch.equal(label_1, label_2)

    print("\nValidation et test déterministes : OK")


# ==========================================
# Visualisation des batches
# ==========================================

def visualize_batches(loaders):

    print("\n=== Visualisation des batches ===")

    split_names = ["train", "val", "test"]

    n_examples = 4

    fig, axes = plt.subplots(
        3,
        n_examples,
        figsize=(13, 10)
    )

    for row, split_name in enumerate(split_names):

        loader = loaders[split_name]

        images, labels = next(iter(loader))

        dataset = loader.dataset

        for col in range(n_examples):

            ax = axes[row, col]

            ax.axis("off")

            if col >= len(images):
                continue

            image = denormalize_image(
                images[col]
            )

            label_idx = labels[col].item()

            label_name = dataset.idx_to_class[
                label_idx
            ]

            ax.imshow(image)

            ax.set_title(
                f"{split_name.upper()}\n{label_name}",
                fontsize=10
            )

    fig.suptitle(
        "ACS - Visualisation des batches",
        fontsize=16
    )

    fig.tight_layout()

    output_path = (
        FIGURES_DIR / "pipeline_batches.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Figure enregistrée : {output_path}")


# ==========================================
# Comparaison des images originales
# et augmentées
# ==========================================

def compare_augmentations(train_loader):

    print("\n=== Inspection des augmentations ===")

    dataset = train_loader.dataset

    n_examples = min(4, len(dataset))

    fig, axes = plt.subplots(
        n_examples,
        3,
        figsize=(12, 3.5 * n_examples),
        squeeze=False
    )

    for i in range(n_examples):

        image_path = dataset.image_paths[i]

        original = load_image(image_path)

        # Deux augmentations indépendantes
        augmented_1, label = dataset[i]

        augmented_2, _ = dataset[i]

        label_name = dataset.idx_to_class[
            label.item()
        ]

        axes[i, 0].imshow(original)

        axes[i, 0].set_title(
            f"Originale - {label_name}"
        )

        axes[i, 1].imshow(
            denormalize_image(augmented_1)
        )

        axes[i, 1].set_title(
            "Augmentation 1"
        )

        axes[i, 2].imshow(
            denormalize_image(augmented_2)
        )

        axes[i, 2].set_title(
            "Augmentation 2"
        )

        for ax in axes[i]:
            ax.axis("off")

    fig.suptitle(
        "ACS - Comparaison des augmentations",
        fontsize=16
    )

    fig.tight_layout()

    output_path = (
        FIGURES_DIR / "pipeline_augmentations.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Figure enregistrée : {output_path}")


# ==========================================
# Sauvegarde du rapport
# ==========================================

def save_report(report):

    output_path = (
        TABLES_DIR / "pipeline_validation.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "ACS - Rapport de validation\n"
        )

        file.write("=" * 35 + "\n")

        for line in report:
            file.write(line + "\n")

        file.write(
            "\nContrôles automatiques : OK\n"
        )

    print(f"\nRapport : {output_path}")


# ==========================================
# Programme principal
# ==========================================

def main():

    print("===================================")
    print("ACS - Validation du pipeline")
    print("===================================")

    config = load_config()

    training_config = config.get("training", {})

    batch_size = training_config.get(
        "batch_size", 4
    )

    num_workers = training_config.get(
        "num_workers", 0
    )

    method = training_config.get(
        "preprocessing_method", "center_crop"
    )

    seed = training_config.get(
        "seed", 42
    )

    # Reproductibilité
    torch.manual_seed(seed)

    # Création des DataLoaders
    train_loader, val_loader, test_loader = (
        create_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers,
            method=method,
            seed=seed
        )
    )

    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

    # Vérifications automatiques
    report = validate_datasets(
        loaders,
        config
    )

    validate_batches(loaders)

    # Inspection visuelle
    visualize_batches(loaders)

    compare_augmentations(train_loader)

    # Rapport
    save_report(report)

    print("\n===================================")
    print("Validation automatique réussie !")
    print("Vérifiez également les figures.")
    print("===================================")


if __name__ == "__main__":
    main()