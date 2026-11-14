from pathlib import Path
import pandas as pd
import yaml

# ACS - Configuration

ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "configs/classification.yaml"

CLASSES_PATH = ROOT / "configs/classes.yaml"

DATASET_PATH = ROOT / "datasets/metadata/classification.csv"

OUTPUT_DIR = ROOT / "datasets/metadata/splits"

SPLITS = ["train", "val", "test"]


# Chargement des configurations
def load_config():

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    with open(CLASSES_PATH, "r", encoding="utf-8") as file:
        classes = yaml.safe_load(file)

    return config, classes


# Chargement du dataset
def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Fichier classification.csv introuvable. "
            "Exécutez d'abord preparation.py."
        )

    df = pd.read_csv(DATASET_PATH, dtype=str)

    if df.empty:
        raise ValueError(
            "Le dataset est vide. "
            "Ajoutez des photographies avant la séparation."
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
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Colonnes manquantes : {missing}"
        )

    if df[required_columns].isna().any().any():
        raise ValueError(
            "Certaines métadonnées obligatoires sont absentes."
        )

    if df["image_id"].duplicated().any():
        raise ValueError(
            "Des identifiants d'images sont dupliqués."
        )

    missing_files = [
        path
        for path in df["image_path"]
        if not (ROOT / path).is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Images introuvables : {missing_files[:5]}"
        )

    return df


# Vérification des classes
def validate_classes(df, classes):

    valid_pairs = {
        (bonbon_type, couleur)
        for bonbon_type, couleurs in classes["variantes"].items()
        for couleur in couleurs
    }

    invalid_pairs = {
        (bonbon_type, couleur)
        for bonbon_type, couleur
        in zip(df["type"], df["couleur"])
        if (bonbon_type, couleur) not in valid_pairs
    }

    if invalid_pairs:
        raise ValueError(
            f"Variantes inconnues : {invalid_pairs}"
        )

# Séparation par session
def split_dataset(df, config):

    split_config = config["split"]

    session_to_split = {}

    for split_name in SPLITS:

        sessions = split_config[split_name]

        for session in sessions:

            if session in session_to_split:
                raise ValueError(
                    f"Session attribuée plusieurs fois : {session}"
                )

            session_to_split[session] = split_name

    # Attribution du split
    df = df.copy()

    df["split"] = df["session_id"].map(session_to_split)

    # Vérification des sessions
    if df["split"].isna().any():

        unknown = df.loc[
            df["split"].isna(), "session_id"
        ].unique()

        raise ValueError(
            f"Sessions non configurées : {unknown}"
        )

    # Vérification des doublons entre ensembles
    duplicate_groups = (
        df.groupby("sha256")["split"]
        .nunique()
    )

    leakage = duplicate_groups[duplicate_groups > 1]

    if not leakage.empty:
        raise ValueError(
            "Data leakage détecté : "
            f"{len(leakage)} groupes d'images identiques "
            "sont présents dans plusieurs ensembles."
        )

    return df

# Sauvegarde des fichiers
def save_splits(df):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Dataset complet avec colonne split
    df.to_csv(
        OUTPUT_DIR / "classification_split.csv",
        index=False
    )

    # Création des fichiers indépendants
    for split_name in SPLITS:

        subset = df[df["split"] == split_name].copy()

        output_path = (
            OUTPUT_DIR / f"classification_{split_name}.csv"
        )

        subset.to_csv(output_path, index=False)

        print(
            f"{split_name.upper()} : "
            f"{len(subset)} images"
        )


# Statistiques
def print_statistics(df, classes):

    print("\n=== Distribution du dataset ===")

    print(f"Nombre total d'images : {len(df)}")

    for split_name in SPLITS:

        subset = df[df["split"] == split_name]

        percentage = (
            len(subset) / len(df)
        ) * 100

        print(
            f"{split_name}: "
            f"{len(subset)} images "
            f"({percentage:.2f} %)"
        )

    # Distribution par variante
    expected_variants = [
        (bonbon_type, couleur)
        for bonbon_type, couleurs in classes["variantes"].items()
        for couleur in couleurs
    ]

    index = pd.MultiIndex.from_tuples(
        expected_variants,
        names=["type", "couleur"]
    )

    distribution = (
        df.groupby(["type", "couleur", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            index=index,
            columns=SPLITS,
            fill_value=0
        )
    )

    distribution["total"] = distribution.sum(axis=1)

    print("\n=== Distribution par variante ===")
    print(distribution.to_string())

    missing_train = distribution[
        distribution["train"] == 0
    ]

    if not missing_train.empty:
        print(
            "\nATTENTION : certaines variantes "
            "ne sont pas représentées dans l'entraînement."
        )

    if (distribution[SPLITS] == 0).any().any():
        print(
            "\nATTENTION : le dataset n'est pas encore "
            "complet pour une évaluation par variante."
        )

    if df["sha256"].duplicated().any():
        print(
            "\nNote : des doublons exacts existent "
            "à l'intérieur d'un même ensemble."
        )


# ==========================================
# Programme principal
# ==========================================

def main():

    print("=== ACS - Séparation du dataset ===")

    config, classes = load_config()

    df = load_dataset()

    validate_classes(df, classes)

    df = split_dataset(df, config)

    save_splits(df)

    print_statistics(df, classes)

    print("\nSéparation terminée avec succès !")


if __name__ == "__main__":
    main()