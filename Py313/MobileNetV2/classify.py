import os
import torch
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# Load pretrained MobileNetV2
# -----------------------------
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.eval()

# ImageNet class labels
labels = models.MobileNet_V2_Weights.IMAGENET1K_V1.meta["categories"]

# -----------------------------
# Preprocessing pipeline
# -----------------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Classification function
# -----------------------------
def classify_image(path):
    img = Image.open(path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        _, predicted = output.max(1)
        class_id = predicted.item()
        return labels[class_id]

# -----------------------------
# Walk through dataset folders
# -----------------------------
ROOT = r"archive\dataset"   # relative to your project folder

for category in os.listdir(ROOT):
    category_path = os.path.join(ROOT, category)

    if not os.path.isdir(category_path):
        continue

    print(f"\n=== Category folder: {category} ===")

    for file in os.listdir(category_path):
        if file.lower().endswith(".jpg"):
            img_path = os.path.join(category_path, file)
            prediction = classify_image(img_path)
            print(f"{file:30} → {prediction}")

