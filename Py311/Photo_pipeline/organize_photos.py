# 🛠️ Photo Organizer with Advanced Face Clustering
# ```
# output/
#    1987/
#       01/
#          no_persons/
#          person_001/
#          person_002/
#          person_001_002/
#          ...
#    1994/
#       07/
#          ...
#    failed/
# ```

# ---

# 🧩 **Architecture**

# # **1. Recursive scan**
# Walk every file under the root folder.

# # **2. Robust timestamp extraction**
# For each image, gather:

# - EXIF `DateTimeOriginal`
# - EXIF `DateTimeDigitized`
# - EXIF `DateTime`
# - filesystem creation time
# - filesystem modified time

# Then compute:

# ## ⭐ `oldest_timestamp = min(all_available_timestamps)`

# If none exist → `unknown_date/unknown_month`.

# This handles:
# - scanned photos  
# - film scans  
# - digital photos  
# - files copied between devices  
# - files restored from backups  
# - files with broken EXIF  

# Useful for large mixed photo library.

# ---

# # **3. Face detection**
# - Use MTCNN  
# - If no faces → copy to `year/month/no_persons/`

# ---

# # **4. Face embedding extraction**
# For each detected face:

# - FaceNet embedding (identity)
# - CLIP embedding (appearance)
# - Store both for clustering

# ---

# # **5. Global 3‑stage clustering**
# After processing *all* images:

###

# ## **Stage 1 — FaceNet + HDBSCAN**
# Identity clusters.

# ## **Stage 2 — CLIP + UMAP + HDBSCAN**
# Appearance clusters for Stage‑1 noise.

# ## **Stage 3 — CLIP + KMeans**
# Forced grouping for Stage‑2 noise.

# This produces a global identity ID for every face.

# ---

# # **6. Multi‑person grouping**
# For each image:

# - Collect all identity IDs detected in that image  
# - Sort them  
# - Folder name = `person_001_004_007`  
# - If only one → `person_001`  
# - If none → `no_persons`

# ---

# # **7. Copy files into final structure**
# ```
# output/YYYY/MM/person_combo/
# ```

# ---

# # **8. Copy unreadable files to**
# ```
# output/failed/
# ```

# ---

# - scans your library  
# - extracts metadata  
# - detects faces  
# - builds embeddings  
# - runs all 3 clustering stages  
# - assigns identity IDs  
# - organizes your entire photo library into year/month/person folders  
# - copies files safely  
# - handles Windows quirks  
# - handles dtype issues  
# - handles EXIF failures  
# - handles multi‑person photos  
# - handles corrupted files  

import os
import shutil
import math
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ExifTags

import torch
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import clip
import hdbscan
import umap
from sklearn.cluster import KMeans

# =============================
# CONFIGURATION
# =============================

INPUT_ROOT = r"X:\PATH\TO\YOUR\PHOTO\LIBRARY"   # TODO: set this
OUTPUT_ROOT = r"X:\PATH\TO\OUTPUT\FOLDER"       # output root folder

os.makedirs(OUTPUT_ROOT, exist_ok=True)
FAILED_DIR = os.path.join(OUTPUT_ROOT, "failed")
os.makedirs(FAILED_DIR, exist_ok=True)

# Face/embedding settings
MAX_FACES_PER_IMAGE = 10  # safety cap

# Stage toggles
RUN_STAGE1 = True   # FaceNet + HDBSCAN (identity)
RUN_STAGE2 = True   # CLIP + UMAP + HDBSCAN (appearance on Stage-1 noise)
RUN_STAGE3 = True   # CLIP + KMeans (forced grouping on Stage-2 noise)

# =============================
# DEVICE & MODELS
# =============================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading models...")
mtcnn = MTCNN(keep_all=True, device=device)  # detect multiple faces
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
to_pil_from_tensor = transforms.ToPILImage()

# =============================
# DATA STRUCTURES
# =============================

@dataclass
class ImageInfo:
    path: str
    year: str
    month: str
    face_indices: List[int]  # indices into global face list

@dataclass
class FaceInfo:
    image_index: int
    facenet_emb: np.ndarray  # (512,)
    clip_emb: np.ndarray     # (512,)

# Global containers
all_images: List[ImageInfo] = []
all_faces: List[FaceInfo] = []

# =============================
# UTILS
# =============================

def collect_image_paths(root_dir: str) -> List[str]:
    paths = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                paths.append(os.path.join(root, f))
    return paths

def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def safe_copy(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        shutil.copy2(src, dst)
    except Exception as e:
        print(f"Copy error: {src} -> {dst}: {e}")

def parse_exif_datetime(exif_value: str) -> Optional[datetime]:
    try:
        # Typical format: "YYYY:MM:DD HH:MM:SS"
        return datetime.strptime(exif_value, "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None

def get_exif_datetimes(pil_img: Image.Image) -> List[datetime]:
    datetimes = []
    try:
        exif = pil_img._getexif()
        if not exif:
            return datetimes

        # Build tag map
        tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}

        for key in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
            if key in tag_map:
                dt = parse_exif_datetime(str(tag_map[key]))
                if dt:
                    datetimes.append(dt)
    except Exception:
        pass
    return datetimes

def get_filesystem_datetimes(path: str) -> List[datetime]:
    times = []
    try:
        stat = os.stat(path)
        # On Windows: st_ctime = creation time, st_mtime = modified time
        for t in [stat.st_ctime, stat.st_mtime]:
            try:
                times.append(datetime.fromtimestamp(t))
            except Exception:
                pass
    except Exception:
        pass
    return times

def get_oldest_timestamp(path: str, pil_img: Optional[Image.Image]) -> Tuple[str, str]:
    """
    Returns (year_str, month_str) using the oldest available timestamp from:
    - EXIF DateTimeOriginal / DateTimeDigitized / DateTime
    - filesystem creation time
    - filesystem modified time
    If none available: ("unknown_year", "unknown_month")
    """
    candidates: List[datetime] = []

    if pil_img is not None:
        candidates.extend(get_exif_datetimes(pil_img))

    candidates.extend(get_filesystem_datetimes(path))

    if not candidates:
        return "unknown_year", "unknown_month"

    oldest = min(candidates)
    return oldest.strftime("%Y"), oldest.strftime("%m")

# =============================
# STEP 1: SCAN & FACE EXTRACTION
# =============================

print(f"Scanning images under: {INPUT_ROOT}")
image_paths = collect_image_paths(INPUT_ROOT)
print(f"Found {len(image_paths)} candidate image files.\n")

for idx, path in enumerate(image_paths):
    if (idx + 1) % 100 == 0 or (idx + 1) == len(image_paths):
        print(f"[Scan] {idx+1}/{len(image_paths)}: {path}")

    pil_img = None
    try:
        pil_img = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[Scan] Failed to open image, sending to failed: {path} ({e})")
        failed_dst = os.path.join(FAILED_DIR, os.path.basename(path))
        safe_copy(path, failed_dst)
        continue

    year, month = get_oldest_timestamp(path, pil_img)

    faces = None
    try:
        faces = mtcnn(pil_img)  # Tensor [N, 3, 160, 160] or None
    except Exception as e:
        print(f"[Scan] Face detection failed, sending to failed: {path} ({e})")
        failed_dst = os.path.join(FAILED_DIR, os.path.basename(path))
        safe_copy(path, failed_dst)
        continue

    face_indices: List[int] = []

    if faces is not None:
        if faces.ndimension() == 3:
            faces = faces.unsqueeze(0)
        if faces.size(0) > MAX_FACES_PER_IMAGE:
            faces = faces[:MAX_FACES_PER_IMAGE]

        faces = faces.to(device)

        with torch.no_grad():
            # FaceNet embeddings
            fn_embs = facenet(faces).cpu().numpy()  # (N, 512)

        # CLIP embeddings per face
        clip_embs_list = []
        for face_tensor in faces:
            face_pil = to_pil_from_tensor(face_tensor.cpu())
            clip_tensor = clip_preprocess(face_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                ce = clip_model.encode_image(clip_tensor).cpu().numpy()[0]
            clip_embs_list.append(ce)
        clip_embs = np.vstack(clip_embs_list)  # (N, 512)

        # Store faces
        for i in range(fn_embs.shape[0]):
            face_index = len(all_faces)
            all_faces.append(
                FaceInfo(
                    image_index=len(all_images),
                    facenet_emb=fn_embs[i],
                    clip_emb=clip_embs[i],
                )
            )
            face_indices.append(face_index)

    # Store image info
    all_images.append(
        ImageInfo(
            path=path,
            year=year,
            month=month,
            face_indices=face_indices
        )
    )

print(f"\nFinished scanning. Total images: {len(all_images)}, total faces: {len(all_faces)}")

if not all_faces:
    print("No faces detected in any image. Only 'no_persons' and 'failed' structure will be created.")
    # We can still organize no_persons by date, but faces logic stops here.


# =============================
# STEP 2: GLOBAL 3-STAGE CLUSTERING
# =============================

# Map each face to a final identity label (int)
# We'll later remap these to person_001, person_002, ...
face_identity_label = np.full(len(all_faces), -1, dtype=int)

if all_faces and (RUN_STAGE1 or RUN_STAGE2 or RUN_STAGE3):
    # Prepare embedding matrices
    facenet_embeddings = np.vstack([f.facenet_emb for f in all_faces]).astype(np.float32)
    clip_embeddings = np.vstack([f.clip_emb for f in all_faces]).astype(np.float32)

    facenet_embeddings = l2_normalize(facenet_embeddings)
    clip_embeddings = l2_normalize(clip_embeddings)

    next_label_offset = 0  # raw label space across stages

    # ---- Stage 1: FaceNet + HDBSCAN (identity) ----
    if RUN_STAGE1:
        print("\n[Stage 1] FaceNet + HDBSCAN (identity clustering)...")
        clusterer1 = hdbscan.HDBSCAN(
            min_cluster_size=3,
            min_samples=2,
            metric="euclidean",
            cluster_selection_method="eom"
        )
        labels1 = clusterer1.fit_predict(facenet_embeddings)
        print("Stage 1 cluster stats:")
        c1 = Counter(labels1)
        print(f"  Total faces: {len(labels1)}")
        print(f"  Clusters (excluding -1): {len([l for l in c1 if l != -1])}")
        print(f"  Noise: {c1.get(-1, 0)}")

        # Assign Stage 1 labels to faces
        face_identity_label = labels1.copy()
        # Track which faces are still noise
        stage1_noise_indices = np.where(labels1 == -1)[0]
        next_label_offset = max([l for l in labels1 if l != -1], default=-1) + 1
    else:
        stage1_noise_indices = np.arange(len(all_faces))

    # ---- Stage 2: CLIP + UMAP + HDBSCAN (appearance on Stage-1 noise) ----
    if RUN_STAGE2 and len(stage1_noise_indices) > 0:
        print(f"\n[Stage 2] CLIP + UMAP + HDBSCAN on {len(stage1_noise_indices)} Stage-1 noise faces...")

        clip_noise = clip_embeddings[stage1_noise_indices]

        print("[Stage 2] Running UMAP...")
        reducer = umap.UMAP(
            n_components=20,
            n_neighbors=15,
            min_dist=0.0,
            metric="cosine",
            random_state=42
        )
        umap_embs = reducer.fit_transform(clip_noise)
        umap_embs = umap_embs.astype(np.float64)  # Windows-safe

        print("[Stage 2] Clustering UMAP embeddings with HDBSCAN...")
        clusterer2 = hdbscan.HDBSCAN(
            min_cluster_size=3,
            min_samples=1,
            metric="euclidean",
            cluster_selection_epsilon=0.15,
            cluster_selection_method="eom",
            algorithm="generic"
        )
        labels2 = clusterer2.fit_predict(umap_embs)

        print("Stage 2 cluster stats:")
        c2 = Counter(labels2)
        print(f"  Total faces: {len(labels2)}")
        print(f"  Clusters (excluding -1): {len([l for l in c2 if l != -1])}")
        print(f"  Noise: {c2.get(-1, 0)}")

        # Map Stage 2 labels into global label space
        # Offset positive labels by next_label_offset
        mapped_labels2 = np.full_like(labels2, -1)
        for lbl in set(labels2):
            if lbl == -1:
                continue
            mapped_labels2[labels2 == lbl] = lbl + next_label_offset

        # Assign to face_identity_label where Stage1 was noise
        face_identity_label[stage1_noise_indices] = np.where(
            labels2 == -1,
            face_identity_label[stage1_noise_indices],
            mapped_labels2
        )

        # Update offset and Stage2 noise
        if any(l != -1 for l in mapped_labels2):
            next_label_offset = max(next_label_offset, mapped_labels2.max() + 1)
        stage2_noise_indices = stage1_noise_indices[labels2 == -1]
    else:
        stage2_noise_indices = stage1_noise_indices

    # ---- Stage 3: CLIP + KMeans (forced grouping on Stage-2 noise) ----
    if RUN_STAGE3 and len(stage2_noise_indices) > 0:
        print(f"\n[Stage 3] CLIP + KMeans on {len(stage2_noise_indices)} remaining noise faces...")

        clip_noise2 = clip_embeddings[stage2_noise_indices]
        n = len(clip_noise2)
        k = max(5, min(80, int(math.sqrt(n / 2))))
        print(f"[Stage 3] Using KMeans with k={k}")

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels3 = kmeans.fit_predict(clip_noise2)

        print("Stage 3 cluster stats:")
        c3 = Counter(labels3)
        print(f"  Total faces: {len(labels3)}")
        print(f"  Clusters: {len(c3)}")

        # Map Stage 3 labels into global label space
        mapped_labels3 = labels3 + next_label_offset
        face_identity_label[stage2_noise_indices] = mapped_labels3
        next_label_offset = mapped_labels3.max() + 1

    # At this point, every face has a non-negative identity label
    # Now remap them to consecutive person IDs: 0..(N_persons-1)
    unique_raw_labels = sorted(set(face_identity_label))
    raw_to_person_id = {raw: i for i, raw in enumerate(unique_raw_labels)}

    # Convenience: store final person ID per face
    face_person_id = np.array([raw_to_person_id[lbl] for lbl in face_identity_label], dtype=int)

    print(f"\nTotal unique persons detected: {len(unique_raw_labels)}")

else:
    face_person_id = np.array([], dtype=int)


# =============================
# STEP 3: BUILD FINAL STRUCTURE & COPY FILES
# =============================

print("\nBuilding final folder structure and copying files...")

for img_idx, img in enumerate(all_images):
    if (img_idx + 1) % 200 == 0 or (img_idx + 1) == len(all_images):
        print(f"[Organize] {img_idx+1}/{len(all_images)}: {img.path}")

    if not img.face_indices:
        # No persons detected in this image
        rel_dir = os.path.join(img.year, img.month, "no_persons")
        dst_dir = os.path.join(OUTPUT_ROOT, rel_dir)
        dst = os.path.join(dst_dir, os.path.basename(img.path))
        safe_copy(img.path, dst)
        continue

    # Collect person IDs for all faces in this image
    persons_in_image = sorted({int(face_person_id[f_idx]) for f_idx in img.face_indices})

    # Build folder name: person_001_004_007 etc.
    person_tokens = [f"person_{pid+1:03d}" for pid in persons_in_image]
    folder_name = "_".join(person_tokens)

    rel_dir = os.path.join(img.year, img.month, folder_name)
    dst_dir = os.path.join(OUTPUT_ROOT, rel_dir)
    dst = os.path.join(dst_dir, os.path.basename(img.path))
    safe_copy(img.path, dst)

print("\nDone.")
print(f"Organized library written under: {OUTPUT_ROOT}")
print("Structure:")
print("  YEAR / MONTH / person_XXX[_YYY...] / image.jpg")
print("  YEAR / MONTH / no_persons / image.jpg")
print("  failed / (images that failed to open or process)")