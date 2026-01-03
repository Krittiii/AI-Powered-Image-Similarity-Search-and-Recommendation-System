import os
import torch
import torchvision.models as models
import joblib
import numpy as np
import torch.nn.functional as F

from common import prepare_data

# ------------------- CONFIG -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------

def get_model():
    """
    ResNet-50 backbone without classification head
    """
    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V2
    )

    # Remove FC layer
    model.fc = torch.nn.Identity()

    model.to(device)
    model.eval()
    return model

def extract_embeddings():
    print("Using device:", device)

    dataset = prepare_data()
    model = get_model()

    embeddings = []
    images = []

    with torch.no_grad():
        for i in range(len(dataset)):
            img, _ = dataset[i]
            images.append(img)

            img = img.unsqueeze(0).to(device)
            feat = model(img)          # [1, 2048]

            feat = F.normalize(feat, p=2, dim=1)  # 🔑 L2 normalization
            embeddings.append(feat.cpu().numpy()[0])

            if i % 500 == 0:
                print(f"Processed {i}/{len(dataset)} images")

    embeddings = np.array(embeddings)

    os.makedirs("output", exist_ok=True)
    joblib.dump((embeddings, images), "output/embeddings.pkl")

    print("Embeddings saved successfully!")
    print("Embeddings shape:", embeddings.shape)

if __name__ == "__main__":
    extract_embeddings()
