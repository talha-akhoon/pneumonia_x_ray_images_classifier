import os
import shutil
from datetime import datetime
from pathlib import Path

import kagglehub

def main():
    today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dest = Path("data/raw/") / today
    dest.mkdir(parents=True, exist_ok=True)
    path = Path(kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia", force_download=True))
    print(os.listdir(path))

    shutil.move(str(path / "chest_xray"), dest)
    shutil.rmtree(dest/ "chest_xray" / "chest_xray")

    print(f"Dataset downloaded and moved to {dest}")

if __name__ == "__main__":
    main()