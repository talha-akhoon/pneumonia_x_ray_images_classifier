import shutil
from datetime import datetime
from pathlib import Path
import kagglehub
from pneumonia_x_ray_images_classifier.config import RAW_DATA_DIR

def get_latest_pneumonia_dataset():
    folders = [f for f in RAW_DATA_DIR.iterdir() if f.is_dir()]
    if not folders:
        raise FileNotFoundError(f"No dataset folders found in {RAW_DATA_DIR}")

    latest_folder = max(folders, key=lambda p: p.name)

    return latest_folder / "chest_xray"


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dest = Path("data/raw") / timestamp
    dest.mkdir(parents=True, exist_ok=True)

    cache_path = Path(
        kagglehub.dataset_download(
            "paultimothymooney/chest-xray-pneumonia",
            force_download=True,
        )
    )

    src = cache_path / "chest_xray"
    dst = dest / "chest_xray"

    shutil.copytree(src, dst, dirs_exist_ok=True)

    # If there is an accidental nested "chest_xray/chest_xray", remove it safely
    nested = dst / "chest_xray"
    if nested.exists() and nested.is_dir():
        shutil.rmtree(nested)

    print(f"Dataset downloaded to cache: {cache_path}")
    print(f"Copied dataset into: {dst}")


if __name__ == "__main__":
    main()
