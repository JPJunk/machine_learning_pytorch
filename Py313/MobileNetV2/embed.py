import os
import torch
from torchvision import models, transforms
from PIL import Image
import csv


DATA_DIR = r"X:/PATH/Pictures"   # folder with .jpg images
OUT_DIR = r"X:/PATH/Embeddings"  # where to save .pt files

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Load MobileNetV2 as embedding model
# -----------------------------
base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

embedding_model = torch.nn.Sequential(
    base.features,
    torch.nn.AdaptiveAvgPool2d((1, 1))
)

embedding_model.eval()

# -----------------------------
# Preprocessing
# -----------------------------
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# -----------------------------
# Extract embedding
# -----------------------------
def get_embedding(path):
    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        feat = embedding_model(x)          # [1, 1280, 1, 1]
        feat = feat.view(1280)             # flatten to [1280]

    return feat

# -----------------------------
# Process all images
# -----------------------------
index_path = os.path.join(OUT_DIR, "index.csv")

with open(index_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "embedding_file"])

    for file in os.listdir(DATA_DIR):
        if not file.lower().endswith(".jpg"):
            continue

        full = os.path.join(DATA_DIR, file)
        emb = get_embedding(full)

        out_file = file.replace(".jpg", ".pt")
        out_path = os.path.join(OUT_DIR, out_file)

        torch.save(emb, out_path)

        writer.writerow([file, out_file])
        print(f"Saved embedding for {file} → {out_file}")