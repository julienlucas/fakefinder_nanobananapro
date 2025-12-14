import subprocess
import shutil
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def download_and_extract_images(repo_id, output_dir):
    print(f"📦 Téléchargement et extraction du dataset: {repo_id}")
    print(f"📁 Destination: {output_dir}\n")

    local_path = Path(output_dir)
    parquet_files = {
        "train": sorted((local_path / "data").glob("train-*.parquet")),
        "test": sorted((local_path / "data").glob("test-*.parquet"))
    }

    if not parquet_files["train"] and not parquet_files["test"]:
        print(f"📥 Téléchargement des fichiers Parquet depuis Hugging Face...")
        print(f"⏳ Cela peut prendre quelques minutes...\n")

        try:
            subprocess.run([
                "huggingface-cli", "download", repo_id,
                "--repo-type", "dataset",
                "--local-dir", str(output_dir)
            ], check=True)

            parquet_files = {
                "train": sorted((local_path / "data").glob("train-*.parquet")),
                "test": sorted((local_path / "data").glob("test-*.parquet"))
            }

            if not parquet_files["train"] and not parquet_files["test"]:
                print(f"❌ Aucun fichier Parquet trouvé après téléchargement")
                return

            print(f"✅ Fichiers Parquet téléchargés\n")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur lors du téléchargement: {e}")
            return
        except FileNotFoundError:
            print(f"❌ huggingface-cli non trouvé. Installez-le avec: pip install huggingface_hub")
            return

    print(f"📁 Chargement depuis les fichiers Parquet locaux...")
    dataset = load_dataset("parquet", data_files={
        "train": [str(f) for f in parquet_files["train"]],
        "test": [str(f) for f in parquet_files["test"]]
    })

    print(f"✅ Dataset chargé: {len(dataset['train'])} train, {len(dataset['test'])} test")

    output_path = Path(output_dir)

    for split in ["train", "test"]:
        print(f"\n📁 Extraction {split}...")
        split_data = dataset[split]

        for label_name, label_idx in [("fake", 0), ("real", 1)]:
            label_dir = output_path / split / label_name
            label_dir.mkdir(parents=True, exist_ok=True)

            count = 0
            for item in tqdm(split_data, desc=f"  {label_name}", leave=False):
                if item["label"] == label_idx:
                    image = item["image"]
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    filename = f"{split}_{label_name}_{count:06d}.jpg"
                    image.save(label_dir / filename, "JPEG", quality=95)
                    count += 1

            print(f"  ✅ {label_name}: {count} images")

    print(f"\n✅ Images extraites dans: {output_path}")

    print(f"\n🧹 Nettoyage des fichiers temporaires...")
    files_to_remove = [
        local_path / ".cache",
        local_path / "data",
        local_path / ".gitattributes",
        local_path / "dataset_infos.json",
        local_path / "README.md"
    ]

    for item in files_to_remove:
        if item.exists():
            if item.is_dir():
                shutil.rmtree(item)
                print(f"  ✅ Supprimé: {item.name}/")
            else:
                item.unlink()
                print(f"  ✅ Supprimé: {item.name}")

    print(f"\n✨ Nettoyage terminé!")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python download_dataset_images.py <repo_id> <output_dir>")
        print("\nExemple:")
        print("  uv run python download_dataset_images.py julienlucas/midjourney-dalle-sd-dataset ./AIvsReal_midjourney_dalle_sd")
        print("\nLe script télécharge automatiquement les fichiers Parquet puis extrait les images.")
        sys.exit(1)

    download_and_extract_images(sys.argv[1], sys.argv[2])

