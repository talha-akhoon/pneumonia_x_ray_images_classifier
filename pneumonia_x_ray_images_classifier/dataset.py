from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

class PneumoniaDataset(Dataset):
    def __init__(self, root_folder: Path, split:str, transform=None):
        self.root_folder = root_folder / split
        self.transform = transform
        self.classes = ["NORMAL", "PNEUMONIA"]
        self.classes_id = dict(zip(self.classes, range(len(self.classes))))

        self.images = []
        self.labels = []

        for cls in self.classes:
            for img_path in (self.root_folder / cls).iterdir():
                self.images.append(img_path)
                self.labels.append(self.classes_id[cls])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert("L")
        label = self.labels[index]

        if (self.transform is not None):
            img = self.transform(img)

        return img, label