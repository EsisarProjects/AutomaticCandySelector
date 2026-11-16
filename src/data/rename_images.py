from pathlib import Path
import argparse
import uuid
import yaml


ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = ROOT / "datasets/classification/raw"

CONFIG_PATH = ROOT / "configs/classes.yaml"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def rename_images(apply=False):

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    total = 0

    for bonbon_type, couleurs in config["variantes"].items():
        for couleur in couleurs:

            directory = DATASET_PATH / bonbon_type / couleur
            if not directory.exists():
                continue

            for session_dir in sorted(directory.iterdir()):

                if not session_dir.is_dir():
                    continue

                session = session_dir.name

                images = sorted(
                    path for path in session_dir.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in EXTENSIONS
                )

                if not images:
                    continue

                # Déterminer les nouveaux noms
                operations = []

                for index, image in enumerate(images, start=1):

                    new_name = (
                        f"{bonbon_type}_{couleur}_"
                        f"{session}_{index:03d}"
                        f"{image.suffix.lower()}"
                    )

                    destination = session_dir / new_name

                    operations.append((image, destination))

                print(f"\n{bonbon_type}/{couleur}/{session}")

                for source, destination in operations:

                    print(f"  {source.name} -> {destination.name}")

                if apply:

                    # Première étape : noms temporaires
                    temporary_files = []

                    for source, destination in operations:

                        temporary = session_dir / (
                            f".acs_tmp_{uuid.uuid4().hex}"
                            f"{source.suffix.lower()}"
                        )

                        source.rename(temporary)

                        temporary_files.append(
                            (temporary, destination)
                        )

                    # Deuxième étape : noms définitifs
                    for temporary, destination in temporary_files:

                        temporary.rename(destination)

                total += len(operations)

    print("\n=== ACS - Renommage des images ===")

    print(f"Images trouvées : {total}")

    if apply:
        print("Renommage terminé !")
    else:
        print("Simulation terminée.")
        print("Ajoutez --apply pour renommer les fichiers.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Appliquer le renommage"
    )

    args = parser.parse_args()

    rename_images(apply=args.apply)