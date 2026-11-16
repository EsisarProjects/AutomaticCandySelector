from pathlib import Path
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

# ==========================================
# ACS - Conversion HEIC vers JPG
# ==========================================

register_heif_opener()

ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = ROOT / "datasets/classification/raw"

def convert_heic():

    images = [
        path for path in DATASET_PATH.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".heic", ".heif"}
    ]

    print("=== ACS - Conversion HEIC ===")
    print(f"Images trouvées : {len(images)}")

    converted = 0
    skipped = 0

    for image_path in images:

        output_path = image_path.with_suffix(".jpg")

        # Ne pas écraser les fichiers existants
        if output_path.exists():
            print(f"Déjà existant : {output_path.name}")
            skipped += 1
            continue

        try:

            with Image.open(image_path) as img:

                # Corriger l'orientation
                image = ImageOps.exif_transpose(img)

                # Conversion RGB
                image = image.convert("RGB")

                # Enregistrement JPG
                image.save(
                    output_path,
                    format="JPEG",
                    quality=95,
                    subsampling=0
                )

            converted += 1

            print(
                f"{image_path.name} -> "
                f"{output_path.name}"
            )

        except Exception as error:

            print(f"Erreur : {image_path}")
            print(error)

    print("\n=== Résultats ===")
    print(f"Images converties : {converted}")
    print(f"Images ignorées : {skipped}")
    print("Conversion terminée !")


if __name__ == "__main__":
    convert_heic()