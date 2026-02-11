import torch
from torchvision import models, transforms
from PIL import Image
import os
import re

MODEL_PATH = "1cards_mobilenetv2.pth"
DATA_DIR = r"valid_cards_with_names"

# Class names (must match training order)
classes = [
    "ace of clubs", "ace of diamonds", "ace of hearts", "ace of spades",
    "eight of clubs", "eight of diamonds", "eight of hearts", "eight of spades",
    "five of clubs", "five of diamonds", "five of hearts", "five of spades",
    "four of clubs", "four of diamonds", "four of hearts", "four of spades",
    "jack of clubs", "jack of diamonds", "jack of hearts", "jack of spades",
    "joker",
    "king of clubs", "king of diamonds", "king of hearts", "king of spades",
    "nine of clubs", "nine of diamonds", "nine of hearts", "nine of spades",
    "queen of clubs", "queen of diamonds", "queen of hearts", "queen of spades",
    "seven of clubs", "seven of diamonds", "seven of hearts", "seven of spades",
    "six of clubs", "six of diamonds", "six of hearts", "six of spades",
    "ten of clubs", "ten of diamonds", "ten of hearts", "ten of spades",
    "three of clubs", "three of diamonds", "three of hearts", "three of spades",
    "two of clubs", "two of diamonds", "two of hearts", "two of spades"
]

num_classes = len(classes)

# -----------------------------
# Load model
# -----------------------------
model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

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
# Classification (top‑1 only for renaming)
# -----------------------------
def classify_top1(path):
    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    top_prob, top_idx = torch.topk(probs, 1)
    label = classes[top_idx.item()]
    conf = float(top_prob.item()) * 100.0
    return label, conf

# -----------------------------
# Safe filename helper
# -----------------------------
def safe_name(s):
    return re.sub(r"[^a-zA-Z0-9_]+", "_", s)

# Track counts per card type
counters = {}

# -----------------------------
# Run classification + rename
# -----------------------------
for file in os.listdir(DATA_DIR):
    if not file.lower().endswith(".jpg"):
        continue

    full = os.path.join(DATA_DIR, file)
    label, conf = classify_top1(full)

    # Running counter
    if label not in counters:
        counters[label] = 1
    else:
        counters[label] += 1

    # Build new filename
    label_clean = safe_name(label)
    conf_str = f"{conf:.2f}".replace(".", "_")  # avoid dots in filenames
    num_str = f"{counters[label]:03d}"

    new_name = f"{label_clean}_{conf_str}_{num_str}.jpg"
    new_full = os.path.join(DATA_DIR, new_name)

    # Rename
    os.rename(full, new_full)

    print(f"{file} → {new_name}")
