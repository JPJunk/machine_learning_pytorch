# ============================================================
# STANDARD LIBRARIES
# ============================================================

import os
import string
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# ============================================================
# THIRD-PARTY LIBRARIES
# ============================================================

import numpy as np
from PIL import Image, ImageDraw
import pdfplumber
import fitz  # PyMuPDF
from pdf2image import convert_from_path

# OCR engines
import onnxruntime as ort
import cv2

# Clustering (used later for region grouping)
from sklearn.cluster import KMeans

# OCR engines (high-level)
from paddleocr import PaddleOCR
import easyocr

from dataclasses import dataclass

from symspellpy.symspellpy import SymSpell

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class RegionBox:
    """
    Simple container for bounding box coordinates.
    Used when grouping OCR regions into zones.
    """
    idx: int
    x1: float
    y1: float
    x2: float
    y2: float

# ============================================================
# CHARACTER SET FOR CRNN DECODING
# ============================================================
# Character set used by the CRNN decoder and for filtering OCR output.
# Includes digits, ASCII, extended European letters, and Lakota-specific glyphs.

# Digits
NUMBERS = "0123456789"

# English letters
ENGLISH_LOWER = "abcdefghijklmnopqrstuvwxyz"
ENGLISH_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ENGLISH_LETTERS = ENGLISH_LOWER + ENGLISH_UPPER

# ASCII punctuation (includes space)
ASCII_PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "

# Extended European characters
ADDITIONAL_EUROPEAN_LOWER = (
    "áàâäãåāăąæ"
    "çćčĉċ"
    "ďđ"
    "éèêëēĕėęě"
    "ğġģ"
    "íìîïīĭįı"
    "ĵ"
    "ķ"
    "ĺļľł"
    "ñńņňŉ"
    "óòôöõøōŏőœ"
    "ŕŗř"
    "śŝşšť"
    "úùûüũūŭůűų"
    "ŵ"
    "ýÿŷ"
    "źżž"
)
ADDITIONAL_EUROPEAN_UPPER = (
    "ÁÀÂÄÃÅĀĂĄÆ"
    "ÇĆČĈĊ"
    "ĎĐ"
    "ÉÈÊËĒĔĖĘĚ"
    "ĞĠĢ"
    "ÍÌÎÏĪĬĮİ"
    "Ĵ"
    "Ķ"
    "ĹĻĽŁ"
    "ÑŃŅŇ"
    "ÓÒÔÖÕØŌŎŐŒ"
    "ŔŖŘ"
    "ŚŜŞŠŤ"
    "ÚÙÛÜŨŪŬŮŰŲ"
    "Ŵ"
    "ÝŸŶ"
    "ŹŻŽ"
)
ADDITIONAL_EUROPEAN_LETTERS = ADDITIONAL_EUROPEAN_LOWER + ADDITIONAL_EUROPEAN_UPPER

# Lakota-specific letters (example)
LAKOTA_LOWER = "áéíóúąįųčšžȟǧʼ"
LAKOTA_UPPER = "ÁÉÍÓÚĄĮŲČŠŽȞǦʼ"
LAKOTA_LETTERS = LAKOTA_LOWER + LAKOTA_UPPER

# Final character set used by CRNN decoder
CHARS = (
    NUMBERS +
    ENGLISH_LETTERS +
    ADDITIONAL_EUROPEAN_LETTERS +
    LAKOTA_LETTERS +
    ASCII_PUNCTUATION
)

# ============================================================
# PIPELINE CONFIGURATION
# ============================================================

# Input/output roots
INPUT_ROOT = r"ocr_test"
OUTPUT_ROOT = r"ocr_output"
DEBUG_ROOT = r"ocr_debug"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# OCR languages
EASYOCR_LANGS = ["en"]   # Add "fi" if needed
PADDLE_LANG = "en"

# Minimum text length to keep a region
MIN_TEXT_LEN = 1

# ============================================================
# OPTIONAL: FIRST RUN — DOWNLOAD ONNX MODELS FROM HUGGINGFACE
# ============================================================
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="marsena/paddleocr-onnx-models",
    local_dir="./paddleocr_onnx"
)

# ============================================================
# OCR ENGINE INITIALIZATION
# ============================================================

# EasyOCR (English)
# Used for recognition fallback or comparison
easyocr_instance = easyocr.Reader(
    ['en'],
    gpu=False
)

# ============================================================
# ONNX MODELS (DETECTION + RECOGNITION)
# ============================================================

# DBNet text detection model (PaddleOCR v5 server)
DBNET_MODEL_PATH = "paddleocr_onnx/PP-OCRv5_server_det_infer.onnx"
dbnet_session = ort.InferenceSession(
    DBNET_MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

# ============================================================
# LOAD SYMSPELL
# ============================================================

MAX_EDIT_DISTANCE = 1
PREFIX_LENGTH = 7

STEM_LIKE_SUFFIXES = (
    "ing", "tion", "sion", "ment", "able", "ible",
    "al", "ive", "ous", "less", "ness",
)

MIN_CONF_FOR_CORRECTION = 0.90  # only correct if conf < 0.90

def load_symspell():
    symspell = SymSpell(max_dictionary_edit_distance=MAX_EDIT_DISTANCE,
                        prefix_length=PREFIX_LENGTH)

    dict_path = Path(__file__).parent / "frequency_dictionary_en_82_765.txt"
    if not dict_path.exists():
        raise FileNotFoundError("Missing SymSpell dictionary file.")

    # Read file, strip BOM if present, rewrite to temp
    with open(dict_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    # Write cleaned version to memory or temp file
    cleaned_path = dict_path.with_suffix(".cleaned.txt")
    with open(cleaned_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)

    symspell.load_dictionary(str(cleaned_path), term_index=0, count_index=1, separator=" ")
    return symspell

symspell_instance = load_symspell()

# ============================================================
"""
SECTION 1 — FILESYSTEM & UTILITY HELPERS
----------------------------------------
These helpers provide simple, predictable I/O operations used throughout
the OCR pipeline. They intentionally contain no OCR logic.

Functions:
    safe_mkdir(path)
    save_text(text, out_dir)
    save_json(obj, out_dir, name)
    save_image(img, out_dir, name)
    to_python(obj)
"""

def safe_mkdir(path: Path):
    """
    Ensure that a directory exists.

    Used before writing:
      - text.txt
      - layout.json
      - debug overlays
      - region crops
      - extracted images
    """
    os.makedirs(path, exist_ok=True)


def save_text(text: str, out_dir: Path):
    """
    Save plain text output to <out_dir>/text.txt.

    Typical uses:
      - reconstructed text in natural reading order
      - debug text dumps
    """
    safe_mkdir(out_dir)
    out_path = out_dir / "text.txt"

    # Atomic write pattern (write then replace)
    tmp_path = out_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    tmp_path.replace(out_path)


def to_python(obj):
    """
    Convert numpy types to native Python types so they can be JSON‑serialized.

    Handles:
      - np.int32, np.int64 → int
      - np.float32, np.float64 → float
      - lists/tuples → recursively converted
      - dicts → recursively converted
    """
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [to_python(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    return obj


def save_json(obj: Any, out_dir: Path, name: str):
    """
    Save a Python object as JSON to <out_dir>/<name>.

    Typical uses:
      - layout.json
      - page metadata
      - paragraph structure dumps
    """
    safe_mkdir(out_dir)
    out_path = out_dir / name

    # Convert numpy types before writing
    obj = to_python(obj)

    tmp_path = out_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp_path.replace(out_path)


def save_image(img: Image.Image, out_dir: Path, name: str):
    """
    Save a PIL image to <out_dir>/<name>.

    Used for:
      - region crops
      - debug overlays
      - paragraph overlays
      - extracted embedded images
    """
    safe_mkdir(out_dir)
    out_path = out_dir / name

    # PIL save is atomic enough for our use case
    img.save(out_path)
    return out_path


# ============================================================
""" SECTION 2 — LOW‑LEVEL OCR MODEL HELPERS 
These functions prepare images for ONNX models and decode their outputs.
Functions:
    preprocess_dbnet(img) 
    postprocess_dbnet(prob_map) 
    pil_crop(img, bbox)
Purpose:
    - Preprocess image → DBNet input
    - Convert DBNet probability map → bounding boxes 
These are the “raw” detection helpers. 
"""

# ============================================================
# DBNet PREPROCESSING
# ============================================================
def preprocess_dbnet(
    img: np.ndarray,
    limit_side_len: int = 960
) -> Tuple[np.ndarray, float, float]:
    """
    Prepare an RGB image for DBNet inference.

    Steps:
      1. Compute scale factor so the longest side ≤ limit_side_len.
      2. Resize while preserving aspect ratio.
      3. Force dimensions to multiples of 32 (DBNet requirement).
      4. Normalize to [0,1], convert to CHW, add batch dimension.
      5. Return scale factors for mapping boxes back.

    Args:
        img: Input RGB image (H, W, 3).
        limit_side_len: Maximum allowed size for the longest side.

    Returns:
        norm: Preprocessed tensor (1, 3, H', W').
        sx:   Horizontal scale factor (orig_w / resized_w).
        sy:   Vertical scale factor   (orig_h / resized_h).
    """
    h, w = img.shape[:2]

    # 1. Compute scale factor
    max_side = max(h, w)
    scale = min(1.0, limit_side_len / max_side)

    new_h = int(h * scale)
    new_w = int(w * scale)

    # 2. Force multiples of 32
    new_h = max(32, (new_h // 32) * 32)
    new_w = max(32, (new_w // 32) * 32)

    # Resize
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 3. Scale factors for mapping back
    sx = w / new_w
    sy = h / new_h

    # 4. Normalize + CHW + batch
    norm = resized.astype(np.float32) / 255.0
    norm = norm.transpose(2, 0, 1)  # HWC → CHW
    norm = np.expand_dims(norm, 0)  # CHW → NCHW

    return norm, sx, sy


# ============================================================
# DBNet POSTPROCESSING
# ============================================================
def postprocess_dbnet(
    prob_map,
    box_thresh: float = 0.6,
    unclip_ratio: float = 1.5,
    min_area: float = 50.0,
    y_offset: float = 2.0
) -> List[np.ndarray]:
    """
    Robust DBNet postprocessing that accepts multiple tensor formats
    and guarantees OpenCV‑safe polygon expansion.
    """

    # ------------------------------------------------------------
    # 1. Convert to NumPy array safely
    # ------------------------------------------------------------
    if hasattr(prob_map, "numpy"):  # Paddle / PyTorch
        prob = prob_map.numpy()
    else:
        prob = np.array(prob_map)

    prob = prob.astype(np.float32)

    # ------------------------------------------------------------
    # 2. Normalize shape to (H, W)
    # ------------------------------------------------------------
    if prob.ndim == 4:
        prob = prob[0, 0]
    elif prob.ndim == 3:
        prob = prob[0]
    elif prob.ndim != 2:
        raise ValueError(f"Unexpected prob_map shape: {prob.shape}")

    # ------------------------------------------------------------
    # 3. Threshold → binary mask
    # ------------------------------------------------------------
    _, binary = cv2.threshold(prob, box_thresh, 1, cv2.THRESH_BINARY)
    binary = (binary * 255).astype(np.uint8)

    # ------------------------------------------------------------
    # 4. Find contours
    # ------------------------------------------------------------
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # 5. Fit min-area rectangle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.float32)

        poly = box.reshape(-1, 2)
        poly_area = cv2.contourArea(poly)
        if poly_area < 1:
            continue

        # --------------------------------------------------------
        # 6. Compute expansion distance safely
        # --------------------------------------------------------
        perimeter = cv2.arcLength(poly, True)
        if perimeter < 1e-6:
            continue

        distance = unclip_ratio * poly_area / perimeter

        # Clamp distance to avoid invalid epsilon
        if not np.isfinite(distance) or distance <= 0:
            distance = 1.0

        # epsilon must be > 0
        epsilon = max(1.0, float(distance))

        # --------------------------------------------------------
        # 7. Expand polygon safely
        # --------------------------------------------------------
        try:
            expanded = cv2.approxPolyDP(poly, epsilon=epsilon, closed=True)
        except Exception:
            expanded = None

        if expanded is None or len(expanded) < 4:
            expanded = poly  # fallback

        poly = expanded.reshape(-1, 2).astype(np.float32)

        # --------------------------------------------------------
        # 8. Apply vertical offset
        # --------------------------------------------------------
        poly[:, 1] += y_offset

        boxes.append(poly)

    return boxes


# ============================================================
# PIL CROP WITH PADDING FOR OCR
# ============================================================
def pil_crop(
    img,
    bbox,
    # proportional padding
    pad_ratio_x=0.05,
    pad_ratio_y=0.3,

    # fixed shrink (per axis)
    shrink_x=10.0,
    shrink_y=10.0,

    # clamps (currently unused, kept for future tuning)
    min_pad_x=6,
    min_pad_y=3,
    max_pad_x=32,
    max_pad_y=32,

    # adaptive boosts (currently unused)
    handwriting_boost=1.40,
    contrast_boost=1.35,
    small_font_boost=1.50,

    handwriting_threshold=0.65,
    contrast_threshold=22.0,
    small_font_threshold=22,

    confidence=None
):
    """
    Produce a padded crop for OCR engines.

    Current behavior (safe & minimal):
        1. Compute proportional padding based on bbox size.
        2. Apply fixed shrink to avoid over-padding.
        3. Apply padding equally on all sides.
        4. Return a single crop (EasyOCR-style).

    Notes:
        - Many advanced tuning blocks (handwriting detection,
          contrast analysis, aspect ratio boosts, confidence-aware
          padding, etc.) are intentionally kept in the codebase but
          commented out. They can be re-enabled later for fine-tuning.
        - This function is intentionally conservative to avoid
          over-expanding regions and merging adjacent text.

    Returns:
        easy_crop : PIL.Image
        easy_bbox : (x1, y1, x2, y2)
    """

    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    # ------------------------------------------------------------
    # 1. Base proportional padding
    # ------------------------------------------------------------
    pad_x = w * pad_ratio_x
    pad_y = h * pad_ratio_y

    # ------------------------------------------------------------
    # 2. Fixed shrink (prevents over-padding)
    # ------------------------------------------------------------
    pad_x = max(0, pad_x - shrink_x)
    pad_y = max(0, pad_y - shrink_y)

    # ------------------------------------------------------------
    # ADVANCED TUNING BLOCKS (kept for future use)
    # ------------------------------------------------------------
    # These blocks are intentionally preserved but disabled.
    # They can be re-enabled when fine-tuning the crop behavior.

    # # 3. Clamp padding
    # pad_x = max(min_pad_x, min(pad_x, max_pad_x))
    # pad_y = max(min_pad_y, min(pad_y, max_pad_y))

    # # 4. Confidence-aware padding
    # if confidence is not None and confidence < 0.90:
    #     factor = 1.0 + (0.90 - confidence) * 0.6
    #     pad_x = min(pad_x * factor, max_pad_x)
    #     pad_y = min(pad_y * factor, max_pad_y)

    # # 5. Extract crop for analysis
    # crop_gray = img.crop((x1, y1, x2, y2)).convert("L")
    # arr = np.array(crop_gray, dtype=np.float32)

    # # 6. Handwriting detection
    # local_var = np.var(arr)
    # gx = np.abs(np.gradient(arr, axis=1))
    # gy = np.abs(np.gradient(arr, axis=0))
    # stroke_strength = np.mean(gx + gy)
    # norm_var = local_var / (255.0**2)
    # norm_grad = stroke_strength / 255.0
    # handwriting_score = 0.7 * norm_var + 0.3 * norm_grad
    # if handwriting_score > handwriting_threshold:
    #     pad_x = min(pad_x * handwriting_boost, max_pad_x)
    #     pad_y = min(pad_y * handwriting_boost, max_pad_y)

    # # 7. Low contrast detection
    # contrast = float(arr.max() - arr.min())
    # if contrast < contrast_threshold:
    #     pad_x = min(pad_x * contrast_boost, max_pad_x)
    #     pad_y = min(pad_y * contrast_boost, max_pad_y)

    # # 8. Small font detection
    # if h < small_font_threshold:
    #     pad_y = min(pad_y * small_font_boost, max_pad_y)

    # # 9. Aspect ratio adjustments
    # aspect = h / w
    # if aspect > 1.6:  # tall
    #     pad_y = min(pad_y * (1.0 + (aspect - 1.6) * 0.25), max_pad_y)
    # if aspect < 0.4:  # wide
    #     pad_x = min(pad_x * (1.0 + (0.4 - aspect) * 0.25), max_pad_x)

    # ------------------------------------------------------------
    # 10. Final padded bbox
    # ------------------------------------------------------------
    def apply_pad(extra_x, extra_y):
        fx1 = max(0, int(x1 - pad_x * extra_x))
        fy1 = max(0, int(y1 - pad_y * extra_y))
        fx2 = min(img.width,  int(x2 + pad_x * extra_x))
        fy2 = min(img.height, int(y2 + pad_y * extra_y))
        return (fx1, fy1, fx2, fy2)

    easy_bbox = apply_pad(1.0, 1.0)
    easy_crop = img.crop(easy_bbox)

    return easy_crop, easy_bbox


# ============================================================
"""
SECTION 3 — REGION DETECTION (PaddleOCR / DBNet / EasyOCR)
This is where text regions (words or lines) are detected.

Functions:
    detect_regions_paddle(img)
    easy_ocr_region(easy_crop, easy_bbox, easy_ocr, symspell, hallucination_penalty)
    shrink_bboxes_centerline(regions, width_ratio, max_width)

Purpose:
    - Use PaddleOCR detection
    - Use EasyOCR for recognition
    - Produce a list of regions with:
    - bbox
    - text
    - confidence
    - engine
    - type (word/line/paragraph)
This is the foundation for everything else.
"""

# ============================================================
# REGION DETECTION WITH PADDLEOCR DBNet
# ============================================================
def detect_regions_paddle(
    img: Image.Image,
    debug_dir=DEBUG_ROOT,
    page_num=1
) -> Tuple[List[Dict[str, Any]], Optional[Image.Image]]:
    """
    Detect coarse text regions using DBNet (ONNX).

    Pipeline:
        1. Preprocess image → DBNet input tensor.
        2. Run DBNet ONNX model.
        3. Convert probability map → polygons → bounding boxes.
        4. Classify each region as text / table / figure.
        5. Optionally save a probability‑map debug overlay.

    Returns:
        regions: list of dicts with:
            {
                "type": "text" | "table" | "figure",
                "bbox": [x1, y1, x2, y2],
                "score": 1.0
            }
        overlay_img: PIL image with region boxes drawn (for debugging)
    """

    arr = np.array(img)
    h_img, w_img = arr.shape[:2]

    # ------------------------------------------------------------
    # 1. Preprocess for DBNet
    # ------------------------------------------------------------
    inp, sx, sy = preprocess_dbnet(arr)

    # ------------------------------------------------------------
    # 2. Run DBNet ONNX model
    # ------------------------------------------------------------
    outputs = dbnet_session.run(None, {"x": inp})
    prob_map = outputs[0]

    # Squeeze to 2D
    while prob_map.ndim > 2:
        prob_map = prob_map[0]

    # ------------------------------------------------------------
    # 3. Extract bounding boxes
    # ------------------------------------------------------------
    boxes = postprocess_dbnet(prob_map)
    print("DBNet boxes:", len(boxes))

    # ------------------------------------------------------------
    # 4. Optional: probability map debug overlay
    # ------------------------------------------------------------
    if debug_dir is not None:
        pm = prob_map

        if pm.ndim == 2 and pm.size > 0:
            pm_norm = (pm - pm.min()) / (pm.max() - pm.min() + 1e-6)
            pm_uint8 = (pm_norm * 255).astype(np.uint8)

            pm_resized = cv2.resize(pm_uint8, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
            pm_color = cv2.applyColorMap(pm_resized, cv2.COLORMAP_JET)

            base = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(base, 0.55, pm_color, 0.45, 0)

            # Draw DBNet boxes
            for box in boxes:
                xs = (box[:, 0] * sx).astype(int)
                ys = (box[:, 1] * sy).astype(int)
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

            fname = f"prob_map_page_{page_num}.png"
            cv2.imwrite(os.path.join(debug_dir, fname), overlay)

    # ------------------------------------------------------------
    # 5. Convert polygons → region dicts
    # ------------------------------------------------------------
    regions = []
    for box in boxes:
        xs = box[:, 0] * sx
        ys = box[:, 1] * sy
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

        # Small padding to avoid tight crops
        pad = int(min(max(6, min(w_img, h_img) * 0.005), 24))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w_img, x2 + pad)
        y2 = min(h_img, y2 + pad)

        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue

        aspect = w / max(h, 1)

        # Extract region crop
        region_crop = arr[y1:y2, x1:x2]

        # Convert to grayscale
        if region_crop.ndim == 3:
            gray = cv2.cvtColor(region_crop, cv2.COLOR_RGB2GRAY)
        else:
            gray = region_crop.astype(np.uint8)

        # Stroke strength = mean gradient magnitude
        gx = np.abs(np.gradient(gray.astype(np.float32), axis=1))
        gy = np.abs(np.gradient(gray.astype(np.float32), axis=0))
        stroke_strength = np.mean(gx + gy)
        edge_density = stroke_strength / 255.0

        # Text density = fraction of dark pixels
        text_density = np.mean(gray < 200)

        # --------------------------------------------------------
        # Region classification heuristics
        # --------------------------------------------------------
        is_large = (w > 350 and h > 250)
        is_wide_text = (aspect > 4 and h < 140)
        is_table_shape = (1.2 < aspect < 4 and h > 120 and w > 250)
        is_dense_edges = (edge_density > 0.12)
        is_sparse_text = (text_density < 0.15)

        if is_large and is_sparse_text:
            rtype = "figure"
        elif is_table_shape and is_dense_edges:
            rtype = "table"
        else:
            rtype = "text"

        regions.append({
            "type": rtype,
            "bbox": [x1, y1, x2, y2],
            "score": 1.0
        })

    print(f"Detected {len(regions)} regions")

    # ------------------------------------------------------------
    # 6. Optional: draw bounding boxes on original image
    # ------------------------------------------------------------
    overlay_img = img.copy()
    draw = ImageDraw.Draw(overlay_img)

    for idx, r in enumerate(regions):
        x1, y1, x2, y2 = r["bbox"]
        color = "blue" if r["type"] == "text" else "green"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 2, y1 + 2), str(idx + 1), fill="red")

    return regions, overlay_img


# ============================================================
# EASYOCR REGION RECOGNITION WITH SYMSPELL CORRECTION
# ============================================================
def easy_ocr_region(
    easy_crop,
    easy_bbox,
    easy_ocr,
    symspell=None,
    hallucination_penalty=0.15
):
    """
    Run EasyOCR on a padded crop and return page‑level word fragments.

    Current behavior:
        - Only EasyOCR is used (PaddleOCR pipeline preserved but disabled).
        - Each EasyOCR fragment is corrected with SymSpell (safe mode).
        - Fragment bounding boxes are shifted from crop‑local → page coords.
        - Returns a clean list of word fragments.

    Args:
        easy_crop: PIL crop for EasyOCR.
        easy_bbox: (x1, y1, x2, y2) page‑level bbox of the crop.
        easy_ocr:  EasyOCR Reader instance.
        symspell:  Optional SymSpell instance for safe correction.
        hallucination_penalty: Confidence penalty for corrected tokens.

    Returns:
        final_frags: list of dicts:
            {
                "bbox": (x1, y1, x2, y2),   # page coords
                "text": "...",
                "confidence": float,
                "engine": "easyocr",
                "type": "word"
            }
    """

    # ------------------------------------------------------------
    # 1. Run EasyOCR on the crop (crop‑local coordinates)
    # ------------------------------------------------------------
    easy_result = easy_ocr.readtext(np.array(easy_crop))
    easy_frags = []

    for (bbox, text, conf) in easy_result:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        easy_frags.append({
            "bbox": (float(x1), float(y1), float(x2), float(y2)),
            "text": text,
            "confidence": float(conf),
            "engine": "easyocr",
            "type": "word"
        })

    # ------------------------------------------------------------
    # 2. Apply SymSpell correction (safe mode)
    # ------------------------------------------------------------
    for f in easy_frags:
        new_text, new_conf = adjust_with_symspell(
            f["text"],
            f["confidence"],
            symspell,
            hallucination_penalty=hallucination_penalty
        )
        f["text"] = new_text
        f["confidence"] = new_conf

    # ------------------------------------------------------------
    # 3. Shift crop‑local boxes → page‑level coordinates
    # ------------------------------------------------------------
    bx1, by1, _, _ = easy_bbox
    final_frags = []

    for e in easy_frags:
        fx1, fy1, fx2, fy2 = e["bbox"]
        page_box = (
            fx1 + bx1,
            fy1 + by1,
            fx2 + bx1,
            fy2 + by1
        )

        final_frags.append({
            **e,
            "bbox": page_box
        })

    return final_frags


# ============================================================
"""
# SECTION 4 — REGION GEOMETRY NORMALIZATION
# ------------------------------------------------------------
# Purpose:
#   DBNet tends to produce bounding boxes that are horizontally
#   oversized. A single detected box may span:
#       • multiple columns
#       • multiple paragraphs
#       • entire page-width headings
#
#   Oversized boxes hide whitespace between logical text blocks.
#   Since XY‑cut relies on whitespace to segment the page, wide
#   boxes cause unrelated regions to merge into a single zone.
#
#   Geometry normalization fixes this by shrinking each region
#   horizontally around its centerline. Vertical coordinates are
#   preserved. The result is a set of narrow “centerline boxes”
#   that expose whitespace and dramatically improve:
#       • XY‑cut segmentation
#       • column detection
#       • paragraph grouping
#       • natural reading order
#
# Output of this section:
#   A list of RegionBox objects with normalized geometry,
#   ready for XY‑cut and column clustering.
"""

# ============================================================
# shrink bounding boxes around centerline
# ============================================================
def shrink_bboxes_centerline(
    regions: List[Dict[str, Any]],
    width_ratio: float = 0.25,
    max_width: float = 80.0
) -> List[RegionBox]:
    """
    Shrink bounding boxes horizontally around their center x.

    Purpose:
        DBNet often produces wide bounding boxes that span across
        columns or multiple logical text blocks. XY‑cut relies on
        whitespace gaps to split zones, so overly wide boxes hide
        whitespace and cause incorrect merging.

    Strategy:
        - Compute the horizontal center of each region.
        - Shrink width to (width * width_ratio), capped at max_width.
        - Keep vertical coordinates unchanged.
        - Return RegionBox objects for XY‑cut.

    Args:
        regions: List of region dicts with "bbox": [x1, y1, x2, y2].
        width_ratio: Fraction of original width to keep (0.25 = 25%).
        max_width: Maximum allowed shrunk width.

    Returns:
        List[RegionBox] with horizontally‑shrunk bounding boxes.
    """
    boxes: List[RegionBox] = []

    for i, r in enumerate(regions):
        x1, y1, x2, y2 = r["bbox"]

        # Center x-coordinate
        cx = (x1 + x2) / 2.0

        # Original width
        w = max(1.0, x2 - x1)

        # New width = shrink factor, capped
        new_w = min(w * width_ratio, max_width)
        half = new_w / 2.0

        # New shrunk box around center
        nx1 = cx - half
        nx2 = cx + half

        # Clamp to valid ordering
        if nx1 > nx2:
            nx1, nx2 = nx2, nx1

        boxes.append(
            RegionBox(
                idx=i,
                x1=nx1,
                y1=y1,
                x2=nx2,
                y2=y2
            )
        )

    return boxes


# ============================================================
"""
# SECTION 5 — XY‑CUT + COLUMN DETECTION
# ------------------------------------------------------------
# This section is the core of the layout engine.
#
# Goal:
#   Convert a flat list of detected regions into meaningful
#   reading-order zones such as:
#       • columns
#       • paragraphs
#       • horizontal bands
#       • multi-column article structures
#
# How it works:
#   1. Compute whitespace gaps between normalized region boxes.
#   2. Recursively apply XY‑cut:
#        - vertical cuts first (column detection)
#        - horizontal cuts second (band segmentation)
#   3. Merge zones that belong to the same horizontal band.
#   4. Produce a stable, hierarchical zone structure.
#
# Why this matters:
#   OCR text alone is not enough — documents must be reconstructed
#   in natural reading order. XY‑cut provides a deterministic,
#   geometry-driven way to recover layout structure without relying
#   on machine learning or heuristics tied to specific fonts.
#
# Output of this section:
#   A list of zones, where each zone is a list of region indices.
#   These zones form the backbone of:
#       • paragraph reconstruction
#       • reading order
#       • layout.json structure
"""

# ============================================================
# compute whitespace gaps along an axis
# ============================================================
def _compute_whitespace_gaps(
    boxes: List[RegionBox],
    axis: str,
    page_extent: float,
    min_gap: float
) -> List[Tuple[float, float]]:
    """
    Compute whitespace gaps along a given axis.

    Args:
        boxes: List of RegionBox objects (already horizontally shrunk).
        axis:  "x" or "y" — which axis to project onto.
        page_extent: Full width or height of the page.
        min_gap: Minimum gap size to consider meaningful.

    Returns:
        List of (gap_start, gap_end) intervals representing whitespace.

    Method:
        1. Project all boxes onto the chosen axis.
        2. Merge overlapping intervals → occupied spans.
        3. Any space between spans ≥ min_gap is a whitespace gap.
    """
    if not boxes:
        return [(0.0, page_extent)]

    # Project boxes onto chosen axis
    if axis == "x":
        coords = sorted((b.x1, b.x2) for b in boxes)
    else:
        coords = sorted((b.y1, b.y2) for b in boxes)

    # Merge overlapping intervals
    occupied = []
    for start, end in coords:
        if not occupied:
            occupied.append([start, end])
        else:
            last = occupied[-1]
            if start <= last[1]:
                last[1] = max(last[1], end)
            else:
                occupied.append([start, end])

    # Compute gaps between occupied intervals
    gaps = []
    cur = 0.0
    for start, end in occupied:
        if start - cur >= min_gap:
            gaps.append((cur, start))
        cur = max(cur, end)

    # Gap after last interval
    if page_extent - cur >= min_gap:
        gaps.append((cur, page_extent))

    return gaps


# ============================================================
# recursive XY‑cut implementation
# ============================================================
def _xy_cut_recursive(
    boxes: List[RegionBox],
    page_width: float,
    page_height: float,
    min_gap_ratio: float,
    min_box_count: int,
    column_gap_override: float,
) -> List[List[RegionBox]]:
    """
    Recursively split boxes into zones using whitespace gaps.

    Strategy:
        1. Try vertical cuts first (column detection).
        2. If no vertical cut is strong enough → try horizontal cuts.
        3. If no cuts apply → return boxes as a single zone.

    Notes:
        - Uses centerline‑shrunk boxes to expose whitespace.
        - min_gap_ratio controls sensitivity.
        - column_gap_override forces strong vertical cuts.
    """
    if not boxes:
        return []

    # Stop recursion if too few boxes
    if len(boxes) <= min_box_count:
        return [boxes]

    # Bounding box of all regions
    min_x = min(b.x1 for b in boxes)
    max_x = max(b.x2 for b in boxes)
    min_y = min(b.y1 for b in boxes)
    max_y = max(b.y2 for b in boxes)

    width = max_x - min_x
    height = max_y - min_y

    # Adaptive minimum gap sizes
    min_gap_x = max(width * min_gap_ratio, 5.0)
    min_gap_y = max(height * min_gap_ratio, 5.0)

    # --------------------------------------------------------
    # 1. Vertical cuts (column detection)
    # --------------------------------------------------------
    v_gaps = _compute_whitespace_gaps(boxes, "x", page_width, min_gap_x)

    best_v_gap = None
    best_v_width = 0.0

    for gx1, gx2 in v_gaps:
        if gx2 <= min_x or gx1 >= max_x:
            continue

        gap_width = min(gx2, max_x) - max(gx1, min_x)
        if gap_width > best_v_width:
            best_v_width = gap_width
            best_v_gap = (gx1, gx2)

    # Strong vertical cut → multi‑column layout
    if best_v_gap and best_v_width >= column_gap_override:
        cut_x = (best_v_gap[0] + best_v_gap[1]) / 2.0

        left = [b for b in boxes if b.x2 <= cut_x]
        right = [b for b in boxes if b.x1 >= cut_x]

        if left and right:
            return (
                _xy_cut_recursive(left, page_width, page_height, min_gap_ratio, min_box_count, column_gap_override)
                + _xy_cut_recursive(right, page_width, page_height, min_gap_ratio, min_box_count, column_gap_override)
            )

    # --------------------------------------------------------
    # 2. Horizontal cuts (bands)
    # --------------------------------------------------------
    h_gaps = _compute_whitespace_gaps(boxes, "y", page_height, min_gap_y)

    best_h_gap = None
    best_h_height = 0.0

    for gy1, gy2 in h_gaps:
        if gy2 <= min_y or gy1 >= max_y:
            continue

        gap_height = min(gy2, max_y) - max(gy1, min_y)
        if gap_height > best_h_height:
            best_h_height = gap_height
            best_h_gap = (gy1, gy2)

    if best_h_gap:
        cut_y = (best_h_gap[0] + best_h_gap[1]) / 2.0

        top = [b for b in boxes if b.y2 <= cut_y]
        bottom = [b for b in boxes if b.y1 >= cut_y]

        if top and bottom:
            return (
                _xy_cut_recursive(top, page_width, page_height, min_gap_ratio, min_box_count, column_gap_override)
                + _xy_cut_recursive(bottom, page_width, page_height, min_gap_ratio, min_box_count, column_gap_override)
            )

    # --------------------------------------------------------
    # 3. No cuts → return as single zone
    # --------------------------------------------------------
    return [boxes]


# ============================================================
# public XY‑cut entry point
# ============================================================
def detect_zones_xycut(
    regions: List[Dict[str, Any]],
    page_width: float,
    page_height: float,
    width_ratio: float = 0.25,
    max_width: float = 80.0,
    min_gap_ratio: float = 0.04,
    min_box_count: int = 3,
    column_gap_override: float = 150.0,
) -> List[List[int]]:
    """
    Public entry point for XY‑cut zone detection.

    Steps:
        1. Shrink bounding boxes horizontally (centerline shrink).
        2. Run recursive XY‑cut segmentation.
        3. Sort resulting zones top‑to‑bottom, left‑to‑right.
        4. Convert RegionBox objects → region indices.

    Returns:
        List of zones, where each zone is a list of region indices.
    """
    if not regions:
        return []

    # 1. Shrink boxes to expose whitespace
    boxes = shrink_bboxes_centerline(
        regions,
        width_ratio=width_ratio,
        max_width=max_width
    )

    # 2. Recursive XY‑cut segmentation
    zones = _xy_cut_recursive(
        boxes,
        page_width=page_width,
        page_height=page_height,
        min_gap_ratio=min_gap_ratio,
        min_box_count=min_box_count,
        column_gap_override=column_gap_override,
    )

    # 3. Sort zones in natural reading order
    def zone_key(zone: List[RegionBox]):
        return (min(b.y1 for b in zone), min(b.x1 for b in zone))

    zones_sorted = sorted(zones, key=zone_key)

    # 4. Convert RegionBox → region indices
    return [[b.idx for b in zone] for zone in zones_sorted]


# ============================================================
# merge zones by vertical band
# ============================================================
def merge_zones_by_vertical_band(
    zones: List[List[int]],
    regions: List[Dict[str, Any]],
    band_threshold: float = 12.0
) -> List[List[int]]:
    """
    Merge XY‑cut zones that belong to the same horizontal band.

    Why this is needed:
        XY‑cut is sensitive to small vertical whitespace variations.
        A single logical band (e.g., a paragraph row or a multi‑column
        header) may be split into multiple adjacent zones.

    Strategy:
        - Compute the top coordinate of each zone.
        - Sort zones by their top coordinate.
        - If two zones have similar top values (within band_threshold),
          they belong to the same horizontal band → merge them.

    Args:
        zones:    List of zones, each zone = list of region indices.
        regions:  Original region dicts (to read bbox coordinates).
        band_threshold: Maximum vertical distance (in px) between
                        zone tops to consider them part of the same band.

    Returns:
        List of merged zones, each zone = list of region indices.
    """
    if not zones:
        return []

    # ------------------------------------------------------------
    # Compute top/bottom for each zone
    # ------------------------------------------------------------
    zone_info = []
    for z_id, zone in enumerate(zones):
        if not zone:
            continue

        tops = [regions[i]["bbox"][1] for i in zone]
        bottoms = [regions[i]["bbox"][3] for i in zone]

        zone_info.append({
            "id": z_id,
            "zone": zone,
            "top": min(tops),
            "bottom": max(bottoms),
        })

    if not zone_info:
        return []

    # ------------------------------------------------------------
    # Sort zones by vertical position
    # ------------------------------------------------------------
    zone_info.sort(key=lambda z: z["top"])

    # ------------------------------------------------------------
    # Merge zones with similar top coordinates
    # ------------------------------------------------------------
    merged = []
    current_band = [zone_info[0]]

    for z in zone_info[1:]:
        prev = current_band[-1]

        # Same horizontal band?
        if abs(z["top"] - prev["top"]) <= band_threshold:
            current_band.append(z)
        else:
            # Flush previous band
            merged.append([i for zz in current_band for i in zz["zone"]])
            current_band = [z]

    # Flush last band
    merged.append([i for zz in current_band for i in zz["zone"]])

    return merged


# ============================================================
"""
# SECTION 5.4 — COLUMN DETECTION WITHIN HORIZONTAL BANDS
# ------------------------------------------------------------
# Purpose:
#   XY‑cut splits the page into horizontal bands, but each band
#   may contain multiple columns. This section determines how
#   many columns exist inside each band and assigns each region
#   to a column.
#
# Why this matters:
#   • Multi‑column documents (articles, reports, textbooks)
#     require correct column grouping to reconstruct reading order.
#   • XY‑cut alone cannot detect columns — it only splits by
#     whitespace. Column detection must be done *within* each band.
#
# How it works:
#   1. detect_columns_gap_based_for_band():
#        - Looks at x‑centers of regions inside a band.
#        - Finds large horizontal gaps → column boundaries.
#        - Assigns column IDs locally.
#        - If the band is very wide and clustering collapses,
#          falls back to global clustering.
#
#   2. cluster_columns_by_x():
#        - Global x‑center clustering across the entire page.
#        - Used as a fallback when local clustering fails.
#
# Output:
#   A mapping: region_id → column_id
#   Used later for:
#       • paragraph grouping
#       • reading order
#       • layout.json column structure
"""

# ============================================================
# column detection within a horizontal band
# ============================================================
def detect_columns_gap_based_for_band(
    regions: List[Dict[str, Any]],
    region_indices: List[int],
    page_width: Optional[float] = None,
    min_gap_ratio: float = 0.06
) -> Dict[int, int]:
    """
    Detect columns within a horizontal band using x‑center gaps.

    Method:
        1. Compute x‑centers of all regions in the band.
        2. Sort them left→right.
        3. Find large gaps between consecutive centers.
        4. Each large gap defines a new column.
        5. If the band spans most of the page width and only one
           column is detected, fall back to global clustering.

    Args:
        regions: Full list of region dicts.
        region_indices: Indices of regions belonging to this band.
        page_width: Full page width (for adaptive gap threshold).
        min_gap_ratio: Fraction of page width required to count as a gap.

    Returns:
        Dict mapping region_id → column_id.
    """
    if not region_indices:
        return {}

    # ------------------------------------------------------------
    # 1. Extract x-centers for regions in this band
    # ------------------------------------------------------------
    xs = [(regions[i]["bbox"][0] + regions[i]["bbox"][2]) / 2.0 for i in region_indices]
    sorted_idx = np.argsort(xs)
    sorted_xs = [xs[i] for i in sorted_idx]

    # ------------------------------------------------------------
    # 2. Find large gaps between consecutive x-centers
    # ------------------------------------------------------------
    gaps = np.diff(sorted_xs)
    gap_thresh = page_width * min_gap_ratio if page_width else 50.0
    split_indices = np.where(gaps > gap_thresh)[0]

    # ------------------------------------------------------------
    # 3. Assign column IDs
    # ------------------------------------------------------------
    col_ids = np.zeros(len(sorted_xs), dtype=int)
    col = 0
    for si in split_indices:
        col += 1
        col_ids[si + 1:] = col

    # Map back to region IDs
    column_for_region = {}
    for rank, idx in enumerate(sorted_idx):
        region_id = region_indices[idx]
        column_for_region[region_id] = int(col_ids[rank])

    # ------------------------------------------------------------
    # 4. Optional override: if the band is wide but clustering collapsed
    # ------------------------------------------------------------
    if page_width:
        x_min = min(regions[i]["bbox"][0] for i in region_indices)
        x_max = max(regions[i]["bbox"][2] for i in region_indices)
        zone_width = x_max - x_min
        unique_cols = set(column_for_region.values())

        if zone_width > 0.8 * page_width and len(unique_cols) <= 1:
            # Fallback to global clustering
            try:
                global_cluster = cluster_columns_by_x(regions, page_width)

                raw_cols = [int(global_cluster[i]) for i in region_indices]

                # Normalize to 0..k-1
                sorted_unique = sorted(set(raw_cols))
                remap = {v: j for j, v in enumerate(sorted_unique)}

                for i, rid in enumerate(region_indices):
                    column_for_region[rid] = remap.get(raw_cols[i], 0)

                print(f"[COL DETECT] Overriding collapsed zone with global x-cluster: {len(sorted_unique)} columns")

            except Exception:
                pass

    return column_for_region


# ============================================================
# global column clustering by x‑centers
# ============================================================
def cluster_columns_by_x(
    regions: List[Dict[str, Any]],
    page_width: float,
    min_gap_ratio: float = 0.06
) -> np.ndarray:
    """
    Global column clustering using x‑center gaps.

    Method:
        1. Compute x‑centers for all regions.
        2. Sort them left→right.
        3. Find large gaps between consecutive centers.
        4. Each gap defines a new column.

    Args:
        regions: List of region dicts.
        page_width: Full page width.
        min_gap_ratio: Fraction of page width required to count as a gap.

    Returns:
        NumPy array of column IDs, aligned with regions list.
    """
    # 1. Collect x-centers
    xs = np.array([(r["bbox"][0] + r["bbox"][2]) / 2.0 for r in regions])
    xs_sorted_idx = np.argsort(xs)
    xs_sorted = xs[xs_sorted_idx]

    # 2. Find large gaps
    gaps = np.diff(xs_sorted)
    gap_thresh = page_width * min_gap_ratio
    split_indices = np.where(gaps > gap_thresh)[0]

    # 3. Assign column IDs
    col_ids = np.zeros(len(xs_sorted), dtype=int)
    col = 0
    for si in split_indices:
        col += 1
        col_ids[si + 1:] += 1

    # 4. Map back to original order
    column_for_region = np.zeros(len(regions), dtype=int)
    for rank, idx in enumerate(xs_sorted_idx):
        column_for_region[idx] = col_ids[rank]

    return column_for_region


# ============================================================
"""
# SECTION 6 — LINE RECONSTRUCTION & PARAGRAPH MERGING
# ------------------------------------------------------------
# Purpose:
#   Convert low‑level OCR word fragments into coherent text:
#       fragments → visual lines → logical paragraphs
#
# Why this matters:
#   OCR engines output words in arbitrary order. Without a
#   reconstruction layer, the text is unreadable. This subsystem
#   restores the document’s *reading order* using geometry alone.
#
# Core components:
#   1. Baseline clustering (group_words_into_lines_baseline)
#        - Words that share a similar vertical baseline are grouped
#          into a visual line.
#        - Horizontal gaps determine word boundaries.
#
#   2. Line merging (merge_lines_to_paragraphs)
#        - Lines are merged into paragraphs using vertical spacing,
#          indentation cues, and XY‑cut zone boundaries.
#
#   3. Strict reconstruction (reconstruct_lines_zones_strict)
#        - Applies glitch fixes (mis‑detections, tiny fragments).
#        - Respects column assignments and XY‑cut zones.
#        - Produces final reading order and paragraph metadata.
#
# Inputs:
#   - OCR fragments (from EasyOCR/PaddleOCR)
#   - XY‑cut zones (layout segmentation)
#   - Column assignments (multi‑column pages)
#
# Outputs:
#   - paragraphs: list[str]
#   - paragraph_boxes: geometric metadata for layout.json
#   - zone_columns: region → column mapping
#   - line_assignments: region → line id
#   - zones: list of region-index lists
#
# This subsystem is the "reading order engine" of the pipeline.
"""

# ============================================================
# group words into visual lines using baseline similarity
# ============================================================
def group_words_into_lines_baseline(
    regions: List[Dict[str, Any]],
    baseline_tol_factor: float = 0.40,
    horiz_gap_factor: float = 1.8
) -> List[Tuple[List[int], str]]:
    """
    Group OCR word fragments into visual text lines using baseline similarity.

    Method:
        1. Compute baselines (bbox bottom y).
        2. Cluster words whose baselines fall within a tolerance window.
        3. Sort each cluster left→right.
        4. Merge horizontally adjacent words into full logical lines.

    Args:
        regions: List of OCR word fragments with "bbox" and "text".
        baseline_tol_factor: Fraction of median height allowed between baselines.
        horiz_gap_factor: Multiplier of median word width allowed between words.

    Returns:
        List of (line_indices, line_text):
            line_indices: indices into `regions`
            line_text: merged text for that line
    """
    if not regions:
        return []

    # ------------------------------------------------------------
    # Extract geometry for all word fragments
    # ------------------------------------------------------------
    baselines = np.array([r["bbox"][3] for r in regions])  # bottom y
    tops      = np.array([r["bbox"][1] for r in regions])
    heights   = np.array([r["bbox"][3] - r["bbox"][1] for r in regions])
    centers_x = np.array([(r["bbox"][0] + r["bbox"][2]) / 2 for r in regions])
    widths    = np.array([r["bbox"][2] - r["bbox"][0] for r in regions])

    median_height = np.median(heights)
    baseline_tol  = median_height * baseline_tol_factor

    # ------------------------------------------------------------
    # 1. Baseline clustering
    # ------------------------------------------------------------
    sorted_idx = np.argsort(baselines).tolist()

    line_clusters = []
    current_cluster = [sorted_idx[0]]

    for i in range(1, len(sorted_idx)):
        prev = sorted_idx[i - 1]
        curr = sorted_idx[i]

        if abs(baselines[curr] - baselines[prev]) <= baseline_tol:
            current_cluster.append(curr)
        else:
            line_clusters.append(current_cluster)
            current_cluster = [curr]

    line_clusters.append(current_cluster)

    # ------------------------------------------------------------
    # 2–3. Sort each cluster left→right and merge horizontally
    # ------------------------------------------------------------
    final_lines = []

    for cluster in line_clusters:
        cluster_sorted = sorted(cluster, key=lambda i: centers_x[i])

        median_width = np.median([widths[i] for i in cluster_sorted])
        horiz_gap = median_width * horiz_gap_factor

        merged = []
        current_line = [cluster_sorted[0]]

        for idx in cluster_sorted[1:]:
            prev = current_line[-1]
            prev_x2 = regions[prev]["bbox"][2]
            curr_x1 = regions[idx]["bbox"][0]

            if curr_x1 - prev_x2 <= horiz_gap:
                current_line.append(idx)
            else:
                merged.append(current_line)
                current_line = [idx]

        merged.append(current_line)

        # --------------------------------------------------------
        # 4. Build text lines
        # --------------------------------------------------------
        for line in merged:
            words = [
                regions[i]["text"].strip()
                for i in line
                if regions[i]["text"].strip()
            ]
            if not words:
                continue

            line_text = " ".join(words)
            final_lines.append((line, line_text))

    return final_lines


# ============================================================
# merge visual lines into paragraphs
# ============================================================
def merge_lines_to_paragraphs(
    raw_lines: List[Dict[str, Any]],
    gap_factor: float = 1.5,
    debug: bool = False
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Merge visual lines into paragraphs using column identity and vertical spacing.

    Assumptions:
        - raw_lines are already sorted by (col, top).
        - Each entry contains:
              "col", "top", "bottom", "text"
          and optionally:
              "left", "right"

    Method:
        1. Iterate through lines in reading order.
        2. Start a new paragraph when:
              • column changes, or
              • vertical gap > gap_factor * line_height.
        3. Otherwise merge into the current paragraph.

    Args:
        raw_lines: List of line dicts.
        gap_factor: Controls paragraph separation sensitivity.
        debug: Print merge/split decisions.

    Returns:
        paragraphs: List of paragraph text strings.
        paragraph_boxes: List of geometric metadata dicts.
    """
    if not raw_lines:
        return [], []

    # Sort by reading order: column → top
    lines = sorted(raw_lines, key=lambda l: (l["col"], l["top"]))

    paragraphs = []
    paragraph_boxes = []

    # Initialize first paragraph
    first = lines[0]
    current_col = first["col"]
    current_top = first["top"]
    current_bottom = first["bottom"]
    current_left = first.get("left")
    current_right = first.get("right")
    current_text_parts = [first["text"]]
    prev_line = first

    # ------------------------------------------------------------
    # Iterate through lines and decide paragraph boundaries
    # ------------------------------------------------------------
    for line in lines[1:]:
        same_column = (line["col"] == current_col)

        vertical_gap = line["top"] - prev_line["bottom"]
        line_height = max(prev_line["bottom"] - prev_line["top"], 1.0)
        allowed_gap = gap_factor * line_height

        if not same_column:
            reason = "NEW paragraph: column change"
            split = True
        elif vertical_gap > allowed_gap:
            reason = f"NEW paragraph: gap {vertical_gap:.1f} > {allowed_gap:.1f}"
            split = True
        else:
            reason = f"MERGE: gap {vertical_gap:.1f} <= {allowed_gap:.1f}"
            split = False

        if debug:
            print(f"[PARA] prev='{prev_line['text']}'  curr='{line['text']}'  → {reason}")

        # --------------------------------------------------------
        # Split paragraph
        # --------------------------------------------------------
        if split:
            para_text = " ".join(current_text_parts).strip()
            if para_text:
                paragraphs.append(para_text)
                paragraph_boxes.append({
                    "col": current_col,
                    "top": current_top,
                    "bottom": current_bottom,
                    "left": current_left,
                    "right": current_right,
                })

            # Start new paragraph
            current_col = line["col"]
            current_top = line["top"]
            current_bottom = line["bottom"]
            current_left = line.get("left")
            current_right = line.get("right")
            current_text_parts = [line["text"]]

        else:
            # Continue same paragraph
            current_text_parts.append(line["text"])
            current_bottom = max(current_bottom, line["bottom"])

            if "left" in line:
                current_left = (
                    min(current_left, line["left"])
                    if current_left is not None else line["left"]
                )
                current_right = (
                    max(current_right, line["right"])
                    if current_right is not None else line["right"]
                )

        prev_line = line

    # ------------------------------------------------------------
    # Final paragraph
    # ------------------------------------------------------------
    para_text = " ".join(current_text_parts).strip()
    if para_text:
        paragraphs.append(para_text)
        paragraph_boxes.append({
            "col": current_col,
            "top": current_top,
            "bottom": current_bottom,
            "left": current_left,
            "right": current_right,
        })

    return paragraphs, paragraph_boxes


# ============================================================
# full strict reconstruction pipeline
# ============================================================
def reconstruct_lines_zones_strict(
    regions: List[Dict[str, Any]],
    page_width: float,
    page_height: float,
    eps_scale: float = 0.55,
    min_confidence: float = 0.80,
    glitch_fixes: bool = False,
    debug_paragraphs: bool = False,
) -> Tuple[
    List[str],                 # paragraphs
    List[List[int]],           # zones
    Dict[int, int],            # zone_columns
    Dict[int, int],            # line_assignments
    List[Dict[str, Any]]       # paragraph_boxes
]:
    """
    Full strict reconstruction pipeline.

    Stages:
        1. XY‑cut zone detection
        2. Merge zones that belong to the same horizontal band
        3. Column detection inside each zone
        4. Baseline‑aware line grouping per column
        5. Optional glitch fixes (OCR cleanup)
        6. Paragraph merging using zone + column + vertical gaps

    Returns:
        paragraphs: list[str]
        zones: list of region-index lists
        zone_columns: region_id → column_id
        line_assignments: region_id → line_id
        paragraph_boxes: geometric metadata for layout.json
    """
    if not regions:
        return [], [], {}, {}, []

    # Confidence lookup for optional filtering
    confidences = {i: r.get("confidence", 1.0) for i, r in enumerate(regions)}

    # ------------------------------------------------------------
    # 1. XY‑cut zones
    # ------------------------------------------------------------
    zones = detect_zones_xycut(regions, page_width, page_height)

    # Merge zones that belong to the same horizontal band
    zones = merge_zones_by_vertical_band(zones, regions, band_threshold=12)

    # Sort zones top→bottom
    zone_tops = []
    for z_id, zone in enumerate(zones):
        if zone:
            top_y = min(regions[i]["bbox"][1] for i in zone)
            zone_tops.append((z_id, top_y))

    sorted_zone_ids = [z_id for z_id, _ in sorted(zone_tops, key=lambda x: x[1])]

    zone_columns: Dict[int, int] = {}
    line_assignments: Dict[int, int] = {}
    next_line_id = 0
    raw_lines: List[Dict[str, Any]] = []

    # ------------------------------------------------------------
    # Optional: global column clustering (fallback)
    # ------------------------------------------------------------
    column_for_region = None
    try:
        col_res = cluster_columns_by_x(regions, page_width)
        if isinstance(col_res, dict):
            column_for_region = {int(k): int(v) for k, v in col_res.items()}
        else:
            column_for_region = {i: int(col_res[i]) for i in range(len(col_res))}
    except Exception:
        column_for_region = None

    # ------------------------------------------------------------
    # 2. Process each zone
    # ------------------------------------------------------------
    for z_id in sorted_zone_ids:
        zone = zones[z_id]
        if not zone:
            continue

        # Local column detection
        zone_cols = detect_columns_gap_based_for_band(regions, zone)

        # --------------------------------------------------------
        # Reconcile with global clustering if local collapsed
        # --------------------------------------------------------
        if column_for_region is not None:
            local_cols = {zone_cols.get(i, 0) for i in zone}
            global_cols = {column_for_region.get(i, 0) for i in zone}

            if len(local_cols) == 1 and len(global_cols) > 1:
                sorted_global = sorted(global_cols)
                remap = {g: j for j, g in enumerate(sorted_global)}
                for idx in zone:
                    gcol = column_for_region.get(idx, 0)
                    zone_cols[idx] = remap.get(gcol, 0)

        # Map region → column
        for idx in zone:
            zone_columns[idx] = zone_cols.get(idx, 0)

        # Group regions by column
        col_groups: Dict[int, List[int]] = {}
        for idx in zone:
            col = zone_cols.get(idx, 0)
            col_groups.setdefault(col, []).append(idx)

        # --------------------------------------------------------
        # 3. Process each column
        # --------------------------------------------------------
        for col in sorted(col_groups.keys()):
            group = col_groups[col]
            if not group:
                continue

            subregions = [regions[i] for i in group]

            # Baseline-aware line grouping (local indices)
            line_groups = group_words_into_lines_baseline(subregions)

            # ----------------------------------------------------
            # 4. Convert local → global indices
            # ----------------------------------------------------
            for local_indices, line_text in line_groups:
                global_indices = [group[i] for i in local_indices]

                # Compute bounding box of the line
                tops = [regions[i]["bbox"][1] for i in global_indices]
                bottoms = [regions[i]["bbox"][3] for i in global_indices]
                lefts = [regions[i]["bbox"][0] for i in global_indices]
                rights = [regions[i]["bbox"][2] for i in global_indices]

                top = float(min(tops))
                bottom = float(max(bottoms))
                left = float(min(lefts))
                right = float(max(rights))

                # Optional glitch cleanup
                if glitch_fixes:
                    line_text = clean_ocr_glitches(line_text)

                # Store raw line
                raw_lines.append({
                    "text": line_text,
                    "top": top,
                    "bottom": bottom,
                    "left": left,
                    "right": right,
                    "zone": z_id,
                    "col": col,
                })

                # Assign line id
                for rid in global_indices:
                    line_assignments[rid] = next_line_id
                next_line_id += 1

    # ------------------------------------------------------------
    # Debug: raw lines before paragraph merging
    # ------------------------------------------------------------
    if debug_paragraphs:
        print("\n--- Raw lines before paragraph merge ---")
        for l in sorted(raw_lines, key=lambda x: (x["zone"], x["col"], x["top"])):
            print(f"[Z{l['zone']} C{l['col']}] y=({l['top']:.1f},{l['bottom']:.1f}) :: {l['text']}")

    # ------------------------------------------------------------
    # 5. Merge lines into paragraphs
    # ------------------------------------------------------------
    paragraphs, paragraph_boxes = merge_lines_to_paragraphs(
        raw_lines,
        gap_factor=1.5,
        debug=debug_paragraphs,
    )

    return paragraphs, zones, zone_columns, line_assignments, paragraph_boxes


# ============================================================
"""
# SECTION 7 — TEXT CLEANUP & REGION CLASSIFICATION
# ------------------------------------------------------------
# Purpose:
#   Enrich OCR output with semantic labels and safe text cleanup.
#
# Why this matters:
#   After line/paragraph reconstruction, some regions still need
#   lightweight semantic classification (e.g., captions, list items)
#   and conservative text cleanup to remove recurring OCR artifacts.
#
# Components:
#   1. clean_ocr_glitches()
#        - Applies conservative, hand‑tuned string replacements.
#        - Targets recurring OCR errors observed in real documents.
#        - Avoids semantic drift (not a spellchecker).
#
#   2. detect_captions()
#        - Identifies short text lines located directly under
#          figures/tables.
#        - Used to tag caption regions in layout.json.
#
#   3. detect_list_items()
#        - Detects bullets, numbered items, and indentation‑based
#          list markers.
#        - Helps downstream consumers (e.g., LLM cleanup) preserve
#          list structure.
#
#   4. adjust_with_symspell()
#        - Optional, conservative SymSpell correction.
#        - Avoids hallucination traps, numbers, names, and short words.
#        - Applies a confidence penalty to discourage over‑correction.
#
# Output:
#   - caption flags
#   - list‑item flags
#   - cleaned text
#   - optional SymSpell‑adjusted text
#
# This subsystem enriches the semantic layer of the OCR pipeline
# without altering the underlying geometry or reading order.
"""

# ============================================================
# detect captions under figures/tables
# ============================================================
def detect_captions(regions: List[Dict[str, Any]]) -> Dict[int, bool]:
    """
    Detect potential captions under figures or tables.

    Heuristics:
        - Only consider text regions.
        - Caption lines are short (height < 80 px).
        - Must be horizontally aligned with a figure/table.
        - Must appear directly below the object (0 < dy < 150 px).

    Returns:
        Dict[int, bool]: region_index → is_caption
    """
    caps = {i: False for i in range(len(regions))}

    for i, r in enumerate(regions):
        if r["type"] != "text":
            continue

        x1, y1, x2, y2 = r["bbox"]
        h = y2 - y1

        if h > 80:
            continue

        for j, other in enumerate(regions):
            if other["type"] not in ("figure", "table"):
                continue

            ox1, oy1, ox2, oy2 = other["bbox"]

            # Horizontal alignment
            if x1 >= ox1 - 50 and x2 <= ox2 + 50:
                # Must be below the object
                if 0 < y1 - oy2 < 150:
                    caps[i] = True

    return caps


# ============================================================
# detect list-like items
# ============================================================
def detect_list_items(regions: List[Dict[str, Any]]) -> Dict[int, bool]:
    """
    Detect list-like items based on text content and indentation.

    Heuristics:
        - Bullet markers: "•", "-", "*", "·"
        - Numbered lists: "1.", "2)", "(3)", "4 -"
        - Indentation: left margin significantly to the right

    Returns:
        Dict[int, bool]: region_index → is_list_item
    """
    items = {i: False for i in range(len(regions))}
    bullets = ("•", "-", "*", "·")

    for i, r in enumerate(regions):
        text = r.get("text", "").strip()
        if not text:
            continue

        # Bullet markers
        if text.startswith(bullets):
            items[i] = True
            continue

        # Numbered lists
        if len(text) >= 2 and text[0].isdigit() and text[1] in (".", ")", "-"):
            items[i] = True
            continue

        # Indentation heuristic
        x1, _, _, _ = r["bbox"]
        if x1 > 80:
            items[i] = True

    return items


# ============================================================
# clean recurring OCR glitches
# ============================================================
def clean_ocr_glitches(text: str) -> str:
    """
    Fix recurring OCR glitches using conservative substring replacements.

    Notes:
        - Not a spellchecker.
        - Avoids semantic drift.
        - Applied line-by-line during reconstruction.
    """

    # THESE ARE JUST EXAMPLES
    fixes = {
        "smell part": "small part",
        "then any": "than any",
    }

    for wrong, right in fixes.items():
        text = text.replace(wrong, right)

    return text


# ============================================================
# SymSpell-based conservative correction
# ============================================================
def adjust_with_symspell(
    text: str,
    conf: float,
    symspell,
    hallucination_penalty: float = 0.25,
) -> Tuple[str, float]:
    """
    Conservative SymSpell-based correction.

    Rules:
        - Skip high-confidence text.
        - Skip names, hyphenated words, numbers, mixed alphanumerics.
        - Skip short alphabetic words (<= 2 chars).
        - Skip stem-like endings (likely split words).
        - Apply SymSpell only to alphabetic tokens.
        - Apply confidence penalty to discourage over-correction.

    Returns:
        (corrected_text, new_confidence)
    """
    if symspell is None:
        return text, conf

    original = text.strip()
    if not original:
        return text, conf

    # High confidence → no correction
    if conf >= MIN_CONF_FOR_CORRECTION:
        return text, conf

    leading = ""
    trailing = ""
    core = original

    # Strip punctuation
    while core and core[0] in string.punctuation:
        leading += core[0]
        core = core[1:]

    while core and core[-1] in string.punctuation:
        trailing = core[-1] + trailing
        core = core[:-1]

    if not core:
        return original, conf

    # Names / hyphenated words
    if core[0].isupper() or "-" in core:
        return original, conf

    # Numbers / mixed alphanumerics
    if any(c.isdigit() for c in core):
        return original, conf

    # Stem-like endings
    for suf in STEM_LIKE_SUFFIXES:
        if core.endswith(suf) and len(core) <= len(suf) + 3:
            return original, conf

    # Short alphabetic words
    if len(core) <= 2 and core.isalpha():
        return leading + core + trailing, conf

    if not core.isalpha():
        return leading + core + trailing, conf

    # SymSpell lookup
    lookup_token = core.lower()
    suggestions = symspell.lookup(lookup_token, verbosity=0)
    if not suggestions:
        return leading + core + trailing, conf

    # Exact match → no change
    for s in suggestions:
        if s.distance == 0 and s.term == lookup_token:
            return leading + core + trailing, conf

    # Best suggestion
    best = min(suggestions, key=lambda s: (s.distance, -s.count))
    dist = best.distance
    freq = best.count
    suggestion = best.term

    if dist > 1 or freq < 50:
        return leading + core + trailing, conf

    # Reapply casing
    if core.isupper():
        suggestion = suggestion.upper()
    elif core[0].isupper():
        suggestion = suggestion.capitalize()

    # Confidence penalty
    h = dist / max(1, len(core))
    new_conf = max(0.0, conf * (1.0 - hallucination_penalty * h))

    corrected = f"{leading}{suggestion}{trailing}"
    print(
        f"SymSpell correction: '{original}' → '{corrected}'  "
        f"dist={dist} freq={freq} conf: {conf:.3f} → {new_conf:.3f}"
    )

    return corrected, new_conf


# ============================================================
"""
# SECTION 8 — VISUALIZATION & DEBUG OVERLAYS
# ------------------------------------------------------------
# Purpose:
#   Provide visual diagnostics for every structural stage of the
#   OCR pipeline. These overlays make the geometric reasoning
#   behind the engine *visible*, allowing rapid debugging and
#   validation on real documents.
#
# Why this matters:
#   The pipeline relies heavily on geometry:
#       • DBNet region detection
#       • XY‑cut zone segmentation
#       • column detection
#       • baseline‑based line grouping
#       • paragraph merging
#
#   Visual overlays allow you to inspect each of these layers
#   directly on the page image, making it easy to diagnose:
#       - missing or merged zones
#       - incorrect column assignments
#       - mis‑grouped lines
#       - paragraph boundary errors
#
# Components:
#   1. draw_paragraph_overlay()
#        - Draws paragraph bounding boxes with labels P1, P2, …
#        - Used to validate paragraph merging and reading order.
#
#   2. draw_debug_overlay()
#        - Full structural overlay:
#             • region bounding boxes
#             • zone ID
#             • column ID
#             • line ID
#        - This is the most important visualization tool for
#          verifying the entire reconstruction pipeline.
#
# Notes:
#   - These functions perform *no OCR logic*.
#   - They operate purely on geometry and metadata.
#   - They are safe to call at any stage of development.
"""

# ============================================================
# draw paragraph bounding boxes
# ============================================================
def draw_paragraph_overlay(
    image_path: Path,
    paragraph_boxes: List[Dict[str, Any]],
    out_path: Path
):
    """
    Draw paragraph bounding boxes on top of the page image.

    Args:
        image_path: Path to the original page image.
        paragraph_boxes: List of dicts with keys:
            "top", "bottom", "left", "right", "zone", "col".
        out_path: Path where the overlay image will be saved.

    Notes:
        - Each paragraph receives a colored rectangle.
        - Labels "P1", "P2", … are drawn in the top-left corner.
        - Colors cycle through a small palette.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Simple repeating color palette
    colors = ["red", "blue", "green", "orange", "magenta", "cyan"]

    for i, box in enumerate(paragraph_boxes):
        left = box.get("left", 0)
        right = box.get("right", img.width)
        top = box["top"]
        bottom = box["bottom"]

        color = colors[i % len(colors)]

        # Draw rectangle
        draw.rectangle([left, top, right, bottom], outline=color, width=3)

        # Draw paragraph label
        draw.text((left + 3, top + 3), f"P{i+1}", fill=color)

    img.save(out_path)


# ============================================================
# draw full debug overlay
# ============================================================
def draw_debug_overlay(
    image_path: Path,
    regions: List[Dict[str, Any]],
    zones: List[List[int]],
    zone_columns: Dict[int, int],
    line_assignments: Dict[int, int],
    out_path: Path,
):
    """
    Draw a debug overlay showing:
        - region bounding boxes colored by zone
        - zone ID
        - column ID
        - line ID

    Args:
        image_path: Path to the original page image.
        regions: List of region dicts (each with bbox + text).
        zones: List of zones (each zone = list of region indices).
        zone_columns: Mapping region_idx → column_id.
        line_assignments: Mapping region_idx → line_id.
        out_path: Path where the overlay image will be saved.

    Notes:
        - This is the primary visualization tool for verifying
          XY‑cut, column detection, and line grouping.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    # Distinct colors for zones
    colors = [
        (255, 0, 0),     # red
        (0, 255, 0),     # green
        (0, 0, 255),     # blue
        (255, 255, 0),   # yellow
        (255, 0, 255),   # magenta
        (0, 255, 255),   # cyan
        (128, 128, 255),
        (255, 128, 128),
        (128, 255, 128),
    ]

    # Map region index → zone ID
    zone_id_by_region: Dict[int, int] = {}
    for zid, zone in enumerate(zones):
        for idx in zone:
            zone_id_by_region[idx] = zid

    # ------------------------------------------------------------
    # Draw each region with zone/column/line labels
    # ------------------------------------------------------------
    for r_idx, r in enumerate(regions):
        x1, y1, x2, y2 = map(int, r["bbox"])

        zid = zone_id_by_region.get(r_idx, -1)
        col_id = zone_columns.get(r_idx, -1)
        line_id = line_assignments.get(r_idx, -1)

        # Choose color based on zone
        color = colors[zid % len(colors)] if zid >= 0 else (200, 200, 200)

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label above the box
        label = f"Z{zid} C{col_id} L{line_id}"
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), img)


# ============================================================
# save debug overlays
# ============================================================
def save_debug_overlays(
    img_path: Path,
    paragraph_boxes: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    zones: List[List[int]],
    zone_columns: Dict[int, int],
    line_assignments: Dict[int, int],
    out_dir: Path,
    page_num: int
):
    """
    Save paragraph and full debug overlays for a page.

    Ensures:
        - img_path is a valid Path
        - the image exists before drawing overlays
        - consistent return values
    """

    # --------------------------------------------------------
    # Validate image path
    # --------------------------------------------------------
    if img_path is None:
        raise ValueError("save_debug_overlays: img_path is None (save_image returned None).")

    img_path = Path(img_path)

    if not img_path.exists():
        raise FileNotFoundError(
            f"save_debug_overlays: image file does not exist: {img_path}"
        )

    # --------------------------------------------------------
    # Paragraph overlay
    # --------------------------------------------------------
    para_overlay = out_dir / f"page_{page_num}_paragraphs.png"
    draw_paragraph_overlay(img_path, paragraph_boxes, para_overlay)

    # --------------------------------------------------------
    # Full debug overlay
    # --------------------------------------------------------
    debug_overlay = out_dir / f"page_{page_num}_overlay.png"
    draw_debug_overlay(
        image_path=img_path,
        regions=regions,
        zones=zones,
        zone_columns=zone_columns,
        line_assignments=line_assignments,
        out_path=debug_overlay,
    )

    return para_overlay, debug_overlay


# ============================================================
"""
# SECTION 9 — PDF / IMAGE INPUT PROCESSING
# ------------------------------------------------------------
# Purpose:
#   Provide the high‑level I/O layer for the OCR engine.
#   This subsystem handles all external document formats and
#   orchestrates the full page‑level processing pipeline.
#
# Responsibilities:
#   1. PDF extraction
#        - Extract embedded raster images (figures, diagrams).
#        - Extract tables using pdfplumber.
#        - Render each PDF page to a high‑resolution image.
#
#   2. Image processing
#        - Accept standalone image files (PNG/JPG/TIFF).
#        - Normalize orientation, DPI, and color mode.
#
#   3. Page‑level OCR pipeline
#        For each rendered page:
#           • DBNet region detection
#           • Hybrid OCR (EasyOCR + PaddleOCR)
#           • Caption detection
#           • List‑item detection
#           • XY‑cut zone segmentation
#           • Column detection
#           • Baseline‑based line grouping
#           • Paragraph merging
#           • Text cleanup (glitches, SymSpell)
#
#   4. Output generation
#        - paragraph text
#        - layout metadata (zones, columns, lines)
#        - extracted images and tables
#        - debug overlays for visual inspection
#
# Why this matters:
#   This section is the “page‑level orchestrator” of the entire
#   OCR engine. It connects raw documents to the geometric and
#   linguistic reconstruction layers, producing structured,
#   machine‑readable output for downstream consumers.
#
# Notes:
#   - No OCR logic is implemented here; this layer only coordinates
#     the pipeline and handles file I/O.
#   - All heavy lifting (zones, columns, lines, paragraphs) is
#     delegated to earlier sections.
"""

# ============================================================
# extract embedded images from PDF
# ============================================================
def extract_embedded_images(pdf_path: Path, out_dir: Path) -> int:
    """
    Extract all embedded raster images from a PDF using PyMuPDF (fitz).

    Args:
        pdf_path: Path to the PDF file.
        out_dir: Directory where extracted images will be saved.

    Returns:
        int: Number of extracted images.

    Notes:
        - Each embedded image is saved as page_{p}_img_{i}.{ext}.
        - This extracts *embedded* images, not rendered page snapshots.
    """
    safe_mkdir(out_dir)
    doc = fitz.open(str(pdf_path))

    count = 0
    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base = doc.extract_image(xref)
            img_bytes = base["image"]
            ext = base["ext"]

            out_path = out_dir / f"page_{page_index+1}_img_{img_index+1}.{ext}"
            with open(out_path, "wb") as f:
                f.write(img_bytes)

            count += 1

    doc.close()
    return count


# ============================================================
# extract tables from PDF
# ============================================================
def extract_pdf_tables(pdf_path: Path, out_dir: Path) -> List[Dict[str, Any]]:
    """
    Extract tables from a PDF using pdfplumber.

    Each table is saved as a TSV-like text file.

    Args:
        pdf_path: Path to the PDF file.
        out_dir: Directory where extracted tables will be saved.

    Returns:
        List[Dict[str, Any]]: Metadata describing extracted tables.
            Each entry contains:
                - "page": page number
                - "index": table index on that page
                - "file": output filename
    """
    safe_mkdir(out_dir)
    tables_meta = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables()
            except Exception:
                continue

            for t_idx, table in enumerate(tables):
                # Convert table rows to TSV
                lines = []
                for row in table:
                    if row:
                        row_clean = [cell if cell else "" for cell in row]
                        lines.append("\t".join(row_clean))

                text = "\n".join(lines)
                filename = f"page_{page_num}_table_{t_idx+1}.txt"
                out_path = out_dir / filename

                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)

                tables_meta.append({
                    "page": page_num,
                    "index": t_idx + 1,
                    "file": filename,
                })

    return tables_meta


# ============================================================
"""
# SECTION 10 — PDF / IMAGE INPUT PROCESSING
# ------------------------------------------------------------
# Purpose:
#   Provide the high‑level I/O and orchestration layer for the
#   entire OCR engine. This subsystem bridges external documents
#   (PDFs, images) with the geometric + linguistic reconstruction
#   pipeline defined in earlier sections.
#
# Responsibilities:
#
#   1. PDF extraction
#        - Extract embedded raster images (figures, diagrams).
#        - Extract tables using pdfplumber.
#        - Render each PDF page to a high‑resolution RGB image.
#
#   2. Image ingestion
#        - Accept standalone image files (PNG/JPG/TIFF).
#        - Normalize orientation, DPI, and color mode.
#
#   3. Page‑level OCR pipeline
#        For each rendered page:
#           • DBNet region detection (coarse blocks)
#           • Hybrid OCR (EasyOCR + PaddleOCR) → word fragments
#           • Caption detection
#           • List‑item detection
#           • XY‑cut zone segmentation
#           • Column detection
#           • Baseline‑based line grouping
#           • Paragraph merging
#           • Text cleanup (glitches, SymSpell)
#
#   4. Debug visualization
#        - Paragraph overlays
#        - Full structural overlays (zones, columns, lines)
#        - Region crops for inspection
#
#   5. Output generation
#        - Word‑level region metadata
#        - Paragraph‑level metadata
#        - Zone/column/line assignments
#        - Debug images and extracted assets
#
# Why this matters:
#   This section is the “page‑level orchestrator” of the OCR engine.
#   It coordinates all subsystems — geometry, text recognition,
#   cleanup, and visualization — and produces the final structured
#   output consumed by downstream tools.
#
# Notes:
#   - No OCR logic is implemented here; this layer only coordinates
#     the pipeline and handles file I/O.
#   - All heavy lifting (zones, columns, lines, paragraphs) is
#     delegated to earlier sections.
"""

# ============================================================
# render a PDF page to a high-resolution RGB image
# ============================================================
def render_pdf_page(pdf_path: Path, page_num: int, dpi: int = 300) -> Image.Image:
    POPPLER_BIN = r"X:\PATH\poppler-25.12.0\Library\bin"

    pages = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=page_num,
        last_page=page_num,
        poppler_path=POPPLER_BIN
    )
    if not pages:
        raise RuntimeError(f"PDF page {page_num} rendered empty")

    return pages[0].convert("RGB")


# ============================================================
# detect DBNet regions in an image
# ============================================================
def detect_dbnet_regions(img: Image.Image, out_dir: Path, page_num: int):
    raw_regions, overlay = detect_regions_paddle(img, debug_dir=out_dir, page_num=page_num)
    overlay_path = out_dir / f"page_{page_num}_dbnet_boxes.png"
    overlay.save(overlay_path)
    return raw_regions, overlay_path


# ============================================================
# extract base text regions from raw DBNet output
# ============================================================
def extract_text_regions(raw_regions):
    base = []
    for r in raw_regions:
        if r.get("type") == "text":
            x1, y1, x2, y2 = r["bbox"]
            base.append({
                "index": len(base),
                "type": "text",
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "text": "",
                "confidence": 0.0,
                "engine": ""
            })
    return base


# ============================================================
# run EasyOCR on base regions
# ============================================================
def run_easy_ocr(img: Image.Image, base_regions):
    expanded = []
    for r in base_regions:
        raw_bbox = r["bbox"]
        crop, easy_bbox = pil_crop(img, raw_bbox)

        r["raw_bbox"] = raw_bbox
        r["easy_bbox"] = easy_bbox
        r["final_bbox"] = easy_bbox
        r["bbox"] = easy_bbox

        fragments = easy_ocr_region(
            easy_crop=crop,
            easy_bbox=easy_bbox,
            easy_ocr=easyocr_instance,
            symspell=symspell_instance,
        )
        expanded.extend(fragments)

    return expanded


# ============================================================
# apply caption detection
# ============================================================
def apply_caption_detection(regions):
    caps = detect_captions(regions)
    for i, flag in caps.items():
        regions[i]["is_caption"] = flag


# ============================================================
# apply list item detection
# ============================================================
def apply_list_detection(regions):
    items = detect_list_items(regions)
    for i, flag in items.items():
        regions[i]["is_list_item"] = flag


# ============================================================
# reconstruct full page structure
# ============================================================
def reconstruct_page_structure(regions, page_width, page_height):
    return reconstruct_lines_zones_strict(
        regions,
        page_width=page_width,
        page_height=page_height,
        eps_scale=0.55,
        min_confidence=0.80,
        glitch_fixes=False
    )


# ============================================================
# build region metadata
# ============================================================
def build_region_metadata(regions, paragraphs):
    meta = []

    for r in regions:
        meta.append({
            "index": r.get("index", 0) + 1,
            "type": r.get("type", "word"),
            "bbox": r["bbox"],
            "text": r.get("text", ""),
            "confidence": r.get("confidence", 0.0),
            "engine": r.get("engine", ""),
            "is_caption": r.get("is_caption", False),
            "is_list_item": r.get("is_list_item", False),
        })

    for p_idx, p_text in enumerate(paragraphs):
        meta.append({
            "index": len(meta) + 1,
            "type": "paragraph",
            "text": p_text,
            "paragraph_index": p_idx + 1
        })

    return meta


# ------------------------------------------------------------
# Process a single image file (PNG/JPG/etc.)
# ------------------------------------------------------------
def process_image_file(image_path: Path, out_dir: Path) -> Dict[str, Any]:
    """
    Process a standalone image using the same OCR pipeline as a PDF page.

    Steps:
        1. Load image
        2. Detect DBNet regions
        3. Extract text blocks
        4. Hybrid OCR → word fragments
        5. Caption + list detection
        6. XY-cut → columns → lines → paragraphs
        7. Draw overlays
        8. Build metadata

    Returns:
        dict with:
            - page number (always 1)
            - regions metadata
            - zones
            - overlay path
    """
    # --------------------------------------------------------
    # 1. Load image
    # --------------------------------------------------------
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {
            "page": 1,
            "error": f"Failed to open image: {e}",
            "regions": []
        }

    img_path = save_image(img, out_dir, "page.png")
    print(f"Loaded image: {image_path}")

    # --------------------------------------------------------
    # 2. Detect DBNet regions
    # --------------------------------------------------------
    raw_regions, dbnet_overlay = detect_regions_paddle(
        img, debug_dir=out_dir, page_num=1
    )
    dbnet_overlay_path = out_dir / "page_dbnet_boxes.png"
    dbnet_overlay.save(dbnet_overlay_path)

    # --------------------------------------------------------
    # 3. Extract text blocks
    # --------------------------------------------------------
    base_regions = []
    for r in raw_regions:
        if r.get("type") == "text":
            x1, y1, x2, y2 = r["bbox"]
            base_regions.append({
                "index": len(base_regions),
                "type": "text",
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "text": "",
                "confidence": 0.0,
                "engine": ""
            })

    # --------------------------------------------------------
    # 4. Hybrid OCR → word fragments
    # --------------------------------------------------------
    expanded = []
    for r in base_regions:
        raw_bbox = r["bbox"]
        crop, easy_bbox = pil_crop(img, raw_bbox)

        r["raw_bbox"] = raw_bbox
        r["easy_bbox"] = easy_bbox
        r["final_bbox"] = easy_bbox
        r["bbox"] = easy_bbox

        fragments = easy_ocr_region(
            easy_crop=crop,
            easy_bbox=easy_bbox,
            easy_ocr=easyocr_instance,
            symspell=symspell_instance,
        )
        expanded.extend(fragments)

    regions = expanded

    # --------------------------------------------------------
    # 5. Caption + list detection
    # --------------------------------------------------------
    apply_caption_detection(regions)
    apply_list_detection(regions)

    # --------------------------------------------------------
    # 6. Full reconstruction
    # --------------------------------------------------------
    page_width, page_height = img.size

    print("\n=== START RECONSTRUCTION ===")
    print(f"Total fragments: {len(regions)}")

    paragraphs, zones, zone_columns, line_assignments, paragraph_boxes = (
        reconstruct_page_structure(regions, page_width, page_height)
    )

    # --------------------------------------------------------
    # 7. Overlays
    # --------------------------------------------------------
    para_overlay_path = out_dir / "page_paragraphs.png"
    draw_paragraph_overlay(img_path, paragraph_boxes, para_overlay_path)

    debug_overlay_path = out_dir / "debug_overlay.png"
    draw_debug_overlay(
        image_path=img_path,
        regions=regions,
        zones=zones,
        zone_columns=zone_columns,
        line_assignments=line_assignments,
        out_path=debug_overlay_path
    )

    # --------------------------------------------------------
    # 8. Metadata
    # --------------------------------------------------------
    region_meta = build_region_metadata(regions, paragraphs)

    return {
        "page": 1,
        "regions": region_meta,
        "zones": zones,
        "overlay": str(debug_overlay_path),
    }

# ------------------------------------------------------------
# Process a single PDF page
# ------------------------------------------------------------
def process_pdf_page(pdf_path: Path, page_num: int, out_dir: Path) -> Dict[str, Any]:
    """
    High‑level page orchestrator:
        - render PDF page
        - detect DBNet regions
        - hybrid OCR
        - caption/list detection
        - XY‑cut → columns → lines → paragraphs
        - overlays + metadata
    """
    # 1. Render page
    try:
        img = render_pdf_page(pdf_path, page_num)
    except Exception as e:
        return {"page": page_num, "error": str(e), "regions": []}

    img_path = save_image(img, out_dir, f"page_{page_num}.png")

    # 2. DBNet region detection
    raw_regions, dbnet_overlay_path = detect_dbnet_regions(img, out_dir, page_num)

    # 3. Extract text blocks
    base_regions = extract_text_regions(raw_regions)

    # 4. Easy OCR → word fragments
    regions = run_easy_ocr(img, base_regions)

    # 5. Classification
    apply_caption_detection(regions)
    apply_list_detection(regions)

    # 6. Reconstruction
    page_width, page_height = img.size
    paragraphs, zones, zone_columns, line_assignments, paragraph_boxes = (
        reconstruct_page_structure(regions, page_width, page_height)
    )

    # 7. Debug overlays
    para_overlay, debug_overlay = save_debug_overlays(
        img_path, paragraph_boxes, regions, zones, zone_columns, line_assignments,
        out_dir, page_num
    )

    # 8. Metadata
    region_meta = build_region_metadata(regions, paragraphs)

    return {
        "page": page_num,
        "regions": region_meta,
        "zones": zones,
        "overlay": str(debug_overlay),
    }


# ============================================================
"""
# SECTION 11 — FILE DISPATCHER & ENTRY POINT
# ------------------------------------------------------------
# Purpose:
#   Provide the top‑level interface for running the OCR engine on
#   arbitrary input files. This subsystem decides how to process
#   each file, orchestrates the page‑level pipeline, and writes
#   all final outputs to disk.
#
# Responsibilities:
#
#   1. File‑type dispatch
#        - Detect whether the input is a PDF or an image.
#        - Route to the appropriate processing pipeline.
#
#   2. Output directory management
#        - Create a clean output directory for each input file.
#        - Organize extracted assets (images, tables, overlays).
#
#   3. Document‑level processing
#        For PDFs:
#            • extract embedded images
#            • extract tables
#            • render each page
#            • run the full OCR pipeline per page
#
#        For images:
#            • normalize and process as a single page
#
#   4. Output generation
#        - text.txt (raw text in reading order)
#        - natural_reading_order/text.txt (paragraph‑level)
#        - layout.json (full structural metadata)
#        - debug overlays for each page
#
# Why this matters:
#   This section is the “front door” of the OCR engine. It is the
#   only part of the system that interacts with the filesystem and
#   external tools. Everything else in the pipeline is pure logic.
#
# Notes:
#   - No OCR logic is implemented here; this layer only coordinates
#     the pipeline and handles I/O.
#   - All heavy lifting (zones, columns, lines, paragraphs) is
#     delegated to earlier sections.
"""

# ------------------------------------------------------------
# Process a full PDF document
# ------------------------------------------------------------
def process_pdf_file(pdf_path: Path, out_dir: Path) -> Dict[str, Any]:
    """
    Process an entire PDF document.

    Steps:
        1. Extract embedded images
        2. Extract tables
        3. Determine page count
        4. Process each page with the full OCR pipeline
        5. Return document-level metadata

    Returns:
        dict with:
            - file: input PDF path
            - embedded_images_count
            - tables: extracted table metadata
            - pages: list of page-level metadata dicts
    """
    safe_mkdir(out_dir)

    # --------------------------------------------------------
    # 1. Extract embedded images
    # --------------------------------------------------------
    embedded_dir = out_dir / "embedded_images"
    embedded_count = extract_embedded_images(pdf_path, embedded_dir)

    # --------------------------------------------------------
    # 2. Extract tables
    # --------------------------------------------------------
    tables_dir = out_dir / "tables"
    tables_meta = extract_pdf_tables(pdf_path, tables_dir)

    # --------------------------------------------------------
    # 3. Determine number of pages
    # --------------------------------------------------------
    pages_meta = []
    try:
        with fitz.open(str(pdf_path)) as doc:
            page_count = len(doc)
    except Exception:
        page_count = 0

    # --------------------------------------------------------
    # 4. Process each page
    # --------------------------------------------------------
    for page_num in range(1, page_count + 1):
        page_meta = process_pdf_page(pdf_path, page_num, out_dir)
        pages_meta.append(page_meta)

    # --------------------------------------------------------
    # 5. Return full document metadata
    # --------------------------------------------------------
    return {
        "file": str(pdf_path),
        "embedded_images_count": embedded_count,
        "tables": tables_meta,
        "pages": pages_meta,
    }


# ============================================================
"""
# SECTION 10 — TEXT EXTRACTION (FULL DOCUMENT)
# ------------------------------------------------------------
# Purpose:
#   Convert the full OCR metadata (pages_meta) into a clean,
#   linear text output in natural reading order.
#
# What this layer does:
#   - Collects paragraph-level regions produced by the
#     reconstruction pipeline (zones → columns → lines → paragraphs)
#   - Preserves the reading order established earlier
#   - Produces:
#         • full document text (text.txt)
#         • natural_reading_order/text.txt
#
# What this layer does NOT do:
#   - No OCR
#   - No geometry
#   - No layout reasoning
#
# This is a pure export layer that serializes the already
# reconstructed paragraphs into a final, human-readable form.
"""

# ------------------------------------------------------------
# Extract full document text
# ------------------------------------------------------------
def extract_full_text(pages_meta: List[Dict[str, Any]]) -> str:
    """
    Extract the full document text in natural reading order.

    Strategy:
        - Paragraphs are stored as regions with type="paragraph"
        - extract_natural_lines() returns them in correct order
        - Join them with newline separators

    Returns:
        A single string containing all paragraphs separated by "\n".
    """
    paragraphs = extract_natural_lines(pages_meta)
    return "\n".join(paragraphs)

# ------------------------------------------------------------
# Extract paragraphs in natural reading order
# ------------------------------------------------------------
def extract_natural_lines(pages_meta: List[Dict[str, Any]]) -> List[str]:
    """
    Extract paragraph-level text from pages_meta in natural reading order.

    Strategy:
        - Iterate through pages in order
        - For each page, iterate through regions
        - Select only regions where type == "paragraph"
        - Append their text to the output list

    Notes:
        - Paragraph order is already correct because:
              • process_pdf_page() sorts zones top→bottom
              • reconstruct_lines_zones_strict() sorts lines
              • merge_lines_to_paragraphs() preserves order
        - This function simply collects them.

    Returns:
        List[str] of paragraph texts.
    """
    lines: List[str] = []

    for page in pages_meta:
        regions = page.get("regions", [])
        for r in regions:
            if r.get("type") == "paragraph":
                text = (r.get("text") or "").strip()
                if text:
                    lines.append(text)

    return lines


# ============================================================
"""
# SECTION 11 — FILE DISPATCHER & ENTRY POINT
# ------------------------------------------------------------
# Purpose:
#   Provide the top‑level interface for running the OCR engine on
#   arbitrary input files. This layer is responsible for deciding
#   how each file should be processed, orchestrating the correct
#   pipeline, and writing all final outputs to disk.
#
# Responsibilities:
#
#   1. File‑type dispatch
#        - Detect whether the input is a PDF or an image.
#        - Route to the appropriate processing pipeline:
#              • process_pdf_file()
#              • process_image_file()
#
#   2. Output directory management
#        - Create a dedicated output directory for each input file.
#        - Organize extracted assets (images, tables, overlays).
#
#   3. Document‑level processing
#        - PDFs:
#              • extract embedded images
#              • extract tables
#              • render pages
#              • run the full OCR pipeline per page
#        - Images:
#              • treat as a single‑page document
#
#   4. Text extraction
#        - extract_full_text(): paragraph‑level text in reading order
#        - extract_natural_lines(): raw paragraph list
#        - Save text outputs to:
#              • text.txt
#              • natural_reading_order/text.txt
#
#   5. Metadata export
#        - Save layout.json containing:
#              • word‑level regions
#              • paragraph‑level regions
#              • zones, columns, lines
#              • captions, list items
#              • overlays and extracted assets
#
# Why this matters:
#   This section is the “front door” of the OCR engine. It is the
#   only layer that interacts with the filesystem and external
#   documents. All geometric and linguistic reasoning happens in
#   earlier sections; this layer simply coordinates the pipeline
#   and serializes the results.
#
# Notes:
#   - No OCR logic is implemented here.
#   - All heavy lifting (zones, columns, lines, paragraphs) is
#     delegated to the reconstruction pipeline.
#   - This layer is intentionally thin and predictable.
"""

# ------------------------------------------------------------
# Process a single file (PDF or image)
# ------------------------------------------------------------
def process_file(path: Path):
    """
    Dispatch a file to the correct processing pipeline.

    Steps:
        1. Create output directory for this file
        2. If PDF → process_pdf_file()
        3. Else → treat as image → process_image_file()
        4. Extract full text and natural reading order
        5. Save layout.json

    Notes:
        - Output directory = OUTPUT_ROOT / <filename_without_ext>
        - Both pipelines produce pages_meta
        - Text extraction operates purely on pages_meta
    """
    print(f"Processing: {path}")

    out_dir = Path(OUTPUT_ROOT) / path.stem
    safe_mkdir(out_dir)

    ext = path.suffix.lower()

    # --------------------------------------------------------
    # 1. PDF vs image dispatch
    # --------------------------------------------------------
    if ext == ".pdf":
        meta = process_pdf_file(path, out_dir)
        pages_meta = meta.get("pages", [])
    else:
        # Treat all non-PDF files as images
        page_meta = process_image_file(path, out_dir)
        meta = {
            "file": str(path),
            "embedded_images_count": 0,
            "tables": [],
            "pages": [page_meta],
        }
        pages_meta = meta["pages"]

    # --------------------------------------------------------
    # 2. Save full text (paragraphs only)
    # --------------------------------------------------------
    full_text = extract_full_text(pages_meta)
    save_text(full_text, out_dir)

    # --------------------------------------------------------
    # 3. Save natural reading order text
    # --------------------------------------------------------
    natural_lines = extract_natural_lines(pages_meta)
    natural_text = "\n".join(natural_lines)
    save_text(natural_text, out_dir / "natural_reading_order")

    # --------------------------------------------------------
    # 4. Save layout.json (full metadata)
    # --------------------------------------------------------
    meta = to_python(meta)
    save_json(meta, out_dir, "layout.json")


# ============================================================
# ENTRY POINT
# ============================================================
def main():
    """
    Entry point for batch OCR processing.

    Steps:
        1. Scan INPUT_ROOT recursively for all files
        2. Process each file
        3. Print progress every 10 files
        4. Write all outputs to OUTPUT_ROOT
    """
    input_root = Path(INPUT_ROOT)
    print(f"Scanning input directory: {input_root}")

    all_files = [p for p in input_root.rglob("*") if p.is_file()]
    total = len(all_files)

    print(f"Found {total} files.\n")

    for idx, path in enumerate(all_files, start=1):
        # Progress indicator
        if idx % 10 == 0 or idx == total:
            print(f"[{idx}/{total}]")

        process_file(path)

    print("\nDone.")
    print(f"OCR output written to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()