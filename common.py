import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# -------------------------------------------------
# ImageNet preprocessing (CRITICAL)
# -------------------------------------------------
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------
# Dataset
# -------------------------------------------------
class FashionDataset(Dataset):
    def __init__(self, csv_path="./dataset/styles.csv",
                 img_dir="./dataset/images",
                 label_col="articleType"):
        
        self.df = pd.read_csv(csv_path, on_bad_lines='skip')
        self.img_dir = img_dir
        self.label_col = label_col

        self.df["image"] = self.df["id"].astype(str) + ".jpg"
        self.df = self.df.reset_index(drop=True)

        # Encode labels numerically
        self.label_map = {
            label: idx for idx, label in
            enumerate(self.df[label_col].unique())
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_dir, row["image"])
        image = Image.open(img_path).convert("RGB")
        image = image_transform(image)

        label = self.label_map[row[self.label_col]]
        return image, label

# -------------------------------------------------
# Public API used by your other scripts
# -------------------------------------------------
def prepare_data():
    return FashionDataset()

# -------------------------------------------------
# Visualization
# -------------------------------------------------
def show_images(images, rows=3, cols=4, figsize=(12, 9),
                filename="output/recommendations.png"):
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, img in enumerate(images):
        img = img.permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())

        axes[i].imshow(img)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
