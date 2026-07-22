from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PhotoRestorationDataset(Dataset):
    # Initialize dataset paths and image transforms.
    def __init__(self, clean_dir, damaged_dir, image_size=128):
        self.clean_dir = Path(clean_dir)
        self.damaged_dir = Path(damaged_dir)

        self.image_names = sorted(
            [
                file.name
                for file in self.clean_dir.iterdir()
                if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                ),
            ]
        )

    # Return the number of image pairs.
    def __len__(self):
        return len(self.image_names)

    # Load and return one image pair.
    def __getitem__(self, index):
        image_name = self.image_names[index]

        clean_path = self.clean_dir / image_name
        damaged_path = self.damaged_dir / image_name

        clean = cv2.imread(str(clean_path))
        damaged = cv2.imread(str(damaged_path))

        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        damaged = cv2.cvtColor(damaged, cv2.COLOR_BGR2RGB)

        clean = self.transform(clean)
        damaged = self.transform(damaged)

        return damaged, clean
    