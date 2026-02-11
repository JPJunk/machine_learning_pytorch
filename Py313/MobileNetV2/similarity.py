import os
import torch

EMB_DIR = "embeddings"

X = []
names = []

for file in os.listdir(EMB_DIR):
    if file.endswith(".pt"):
        emb = torch.load(os.path.join(EMB_DIR, file))
        X.append(emb.numpy())
        names.append(file)

import numpy as np
X = np.vstack(X)   # shape: [N, 1280]

###

from sklearn.cluster import KMeans

k = 20   # number of clusters (tune this)
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(X)

###

import shutil

OUT = "clusters"
os.makedirs(OUT, exist_ok=True)

for label, name in zip(labels, names):
    folder = os.path.join(OUT, f"cluster_{label:02d}")
    os.makedirs(folder, exist_ok=True)

    src = os.path.join("images_to_embed", name.replace(".pt", ".jpg"))
    dst = os.path.join(folder, name.replace(".pt", ".jpg"))
    shutil.copy(src, dst)