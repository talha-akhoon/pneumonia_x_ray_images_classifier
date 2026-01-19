import os
import shutil
from sklearn.model_selection import train_test_split
from pneumonia_x_ray_images_classifier.config import PROCESSED_DATA_DIR
from pneumonia_x_ray_images_classifier.data.make_dataset import get_latest_pneumonia_dataset


def main():
    latest_folder = get_latest_pneumonia_dataset(use_raw=True)
    shutil.rmtree(PROCESSED_DATA_DIR, ignore_errors=True)

    classes = ["NORMAL", "PNEUMONIA"]
    src_split_dir = latest_folder / "test"
    dst_split_dir = PROCESSED_DATA_DIR / "test"


    for cls in classes:
        src_cls = src_split_dir / cls
        dst_cls = dst_split_dir / cls
        dst_cls.mkdir(parents=True, exist_ok=True)

        for img_path in src_cls.glob("*.*"):
            if img_path.is_file():
                shutil.copy2(img_path, dst_cls)

    images = []
    labels = []
    for cls in classes:
        for img_path in (latest_folder / "train" / cls).iterdir():
            if img_path.is_file():
                images.append(img_path)
                labels.append(cls)

    train_paths, val_paths, train_labels, val_labels = train_test_split(images, labels, stratify=labels, train_size=0.8, test_size=0.2, random_state=42)

    for img_path, cls in zip(train_paths, train_labels):
        dst = PROCESSED_DATA_DIR / "train" / cls
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dst)

    for img_path, cls in zip(val_paths, val_labels):
        dst = PROCESSED_DATA_DIR / "val" / cls
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dst)