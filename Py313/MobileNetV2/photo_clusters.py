import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from sklearn.cluster import KMeans
import shutil
import csv

ROOT_DIR = r"X:\PATH\Pictures"
OUT_DIR = r"X:\PATH\photo_clusters"
FAILED_DIR = os.path.join(OUT_DIR, "failed")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FAILED_DIR, exist_ok=True)

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
        feat = embedding_model(x).view(1280)
    return feat.numpy()

# -----------------------------
# Walk all subdirectories
# -----------------------------
image_paths = []
for root, dirs, files in os.walk(ROOT_DIR):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(root, f))

print(f"Found {len(image_paths)} images.")

# -----------------------------
# Extract embeddings (safe)
# -----------------------------
valid_paths = []
failed_paths = []
embeddings = []

for i, path in enumerate(image_paths):
    try:
        emb = get_embedding(path)
        embeddings.append(emb)
        valid_paths.append(path)
        print(f"[{i+1}/{len(image_paths)}] Embedded: {path}")
    except Exception as e:
        print(f"FAILED: {path} → {e}")
        failed_paths.append(path)

        # Copy failed file to failed/ folder
        try:
            shutil.copy(path, os.path.join(FAILED_DIR, os.path.basename(path)))
        except Exception as e2:
            print(f"Could not copy failed file {path}: {e2}")

# Save failed list
with open(os.path.join(OUT_DIR, "failed.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["path"])
    for p in failed_paths:
        w.writerow([p])

# Convert embeddings to array
embeddings = np.vstack(embeddings)
np.save(os.path.join(OUT_DIR, "embeddings.npy"), embeddings)

# Save index of valid images
with open(os.path.join(OUT_DIR, "index.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "path"])
    for i, p in enumerate(valid_paths):
        w.writerow([i, p])

# -----------------------------
# Cosine similarity matrix
# -----------------------------
def cosine_sim(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))

# Compute top‑5 neighbors
with open(os.path.join(OUT_DIR, "similarity_top5.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image", "neighbor1", "sim1", "neighbor2", "sim2",
                "neighbor3", "sim3", "neighbor4", "sim4", "neighbor5", "sim5"])

    for i in range(len(embeddings)):
        sims = []
        for j in range(len(embeddings)):
            if i != j:
                sims.append((j, cosine_sim(embeddings[i], embeddings[j])))

        sims.sort(key=lambda x: x[1], reverse=True)
        top5 = sims[:5]

        row = [valid_paths[i]]
        for idx, sim in top5:
            row.append(valid_paths[idx])
            row.append(sim)

        w.writerow(row)

print("Saved similarity_top5.csv")

# -----------------------------
# K-Means clustering
# -----------------------------
k = 106
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(embeddings)

# Create cluster folders
for c in range(k):
    os.makedirs(os.path.join(OUT_DIR, f"cluster_{c:02d}"), exist_ok=True)

# Copy valid images into clusters
for i, path in enumerate(valid_paths):
    dst = os.path.join(OUT_DIR, f"cluster_{labels[i]:02d}", os.path.basename(path))
    try:
        shutil.copy(path, dst)
    except Exception as e:
        print(f"Failed to copy {path}: {e}")

print("Clustering complete.")
print(f"Clusters saved in: {OUT_DIR}")
print(f"Failed images saved in: {FAILED_DIR}")
