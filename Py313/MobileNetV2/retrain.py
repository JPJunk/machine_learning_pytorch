# MobileNetV2(
#   (features): Sequential(
#     (0): ConvBNReLU(3 → 32, kernel=3, stride=2)
#     (1): InvertedResidual(32 → 16, stride=1, expand_ratio=1)
#     (2): InvertedResidual(16 → 24, stride=2, expand_ratio=6)
#     (3): InvertedResidual(24 → 24, stride=1, expand_ratio=6)
#     (4): InvertedResidual(24 → 32, stride=2, expand_ratio=6)
#     (5): InvertedResidual(32 → 32, stride=1, expand_ratio=6)
#     (6): InvertedResidual(32 → 32, stride=1, expand_ratio=6)
#     (7): InvertedResidual(32 → 64, stride=2, expand_ratio=6)
#     (8): InvertedResidual(64 → 64, stride=1, expand_ratio=6)
#     (9): InvertedResidual(64 → 64, stride=1, expand_ratio=6)
#     (10): InvertedResidual(64 → 64, stride=1, expand_ratio=6)
#     (11): InvertedResidual(64 → 96, stride=1, expand_ratio=6)
#     (12): InvertedResidual(96 → 96, stride=1, expand_ratio=6)
#     (13): InvertedResidual(96 → 96, stride=1, expand_ratio=6)
#     (14): InvertedResidual(96 → 160, stride=2, expand_ratio=6)
#     (15): InvertedResidual(160 → 160, stride=1, expand_ratio=6)
#     (16): InvertedResidual(160 → 160, stride=1, expand_ratio=6)
#     (17): InvertedResidual(160 → 320, stride=1, expand_ratio=6)
#     (18): ConvBNReLU(320 → 1280, kernel=1, stride=1)
#   )
#   (classifier): Sequential(
#     (0): Dropout(p=0.2)
#     (1): Linear(1280 → num_classes)
#   )
# )

# features[0]   ConvBNReLU (low-level edges)
# features[1-3] InvertedResidual (early textures)
# features[4-6] InvertedResidual (mid-level shapes)
# features[7-13] InvertedResidual (higher-level patterns)
# features[14-17] InvertedResidual (semantic features)
# features[18]  ConvBNReLU (final expansion to 1280-dim)
# classifier[0] Dropout
# classifier[1] Linear (your new head)

# 🔀 Rule of thumb for how many layers to unfreeze
# - Small dataset, very similar to ImageNet → freeze everything except classifier.
# - Medium dataset, moderately different (like your cards) → unfreeze the last 2–4 inverted residual blocks.
# - Large dataset, very different (medical, satellite) → fine‑tune the whole network


import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from datetime import datetime

# -----------------------------
# Helper: timestamped print
# -----------------------------
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# -----------------------------
# Settings
# -----------------------------
DATA_DIR = r"archive/train"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
MODEL_PATH = "cards_mobilenetv2.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Using device: {device}")

# -----------------------------
# Dataset & transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

num_classes = len(dataset.classes)
log(f"Classes: {dataset.classes}")

# -----------------------------
# Load MobileNetV2 + modify head
# -----------------------------
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Freeze most of the backbone
for param in model.features.parameters():
    param.requires_grad = False

# Unfreeze the last N layers (e.g., last 9 inverted residual blocks)
for param in model.features[-9:].parameters():
    param.requires_grad = True

model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

# -----------------------------
# Loss & optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()

# Optimizer only sees trainable params (last blocks + classifier)
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_params, lr=LR)

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    log(f"Starting epoch {epoch+1}/{EPOCHS}")

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataset)
    epoch_acc = 100 * correct / total

    log(f"Epoch {epoch+1}/{EPOCHS}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.2f}%")

# -----------------------------
# Save trained model
# -----------------------------
torch.save(model.state_dict(), MODEL_PATH)
log(f"Model saved to {MODEL_PATH}")
