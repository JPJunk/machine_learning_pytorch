"""
photo_repair_pipeline.py

Run this in the same environment as your organizer script.

It will:
- Read all files from FAILED_DIR
- Try to salvage them using tolerant decoding
- Optionally apply very light cleanup
- Save any successfully decoded images into RECOVERED_DIR

No human interaction. If it recovers something, good. If not, it skips.
"""

import os
import shutil
from pathlib import Path

from typing import Optional

from PIL import Image, ImageFile, ImageFilter, ImageEnhance

# =============================
# CONFIGURATION
# =============================

# Folder with corrupted/unreadable images from the organizer script
FAILED_DIR = r"organized_photos\failed"   # TODO: adjust if needed

# Where recovered images will be written
RECOVERED_DIR = r"organized_photos\recovered"

os.makedirs(RECOVERED_DIR, exist_ok=True)

# If True, applies very light sharpening/contrast to recovered images
APPLY_LIGHT_ENHANCEMENT = True

# Extensions we consider as images
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# =============================
# UTILS
# =============================

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def safe_open_image(path: Path) -> Optional[Image.Image]:
    """
    Try different strategies to open a possibly corrupted image.
    Returns a PIL Image or None if completely unreadable.
    """

    # 1) Normal open
    try:
        img = Image.open(path)
        img.load()
        return img
    except Exception:
        pass

    # 2) Allow truncated images
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(path)
        img.load()
        return img
    except Exception:
        pass
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False

    # 3) Try to copy raw bytes into a new file and reopen
    #    (sometimes helps with weird filesystem/metadata issues)
    try:
        tmp_path = path.with_suffix(path.suffix + ".tmp_repair")
        with open(path, "rb") as f_in, open(tmp_path, "wb") as f_out:
            data = f_in.read()
            f_out.write(data)

        try:
            img = Image.open(tmp_path)
            img.load()
            return img
        except Exception:
            pass
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass
    except Exception:
        pass

    # 4) Give up
    return None


def lightly_enhance(img: Image.Image) -> Image.Image:
    """
    Apply very mild sharpening and contrast enhancement.
    This is intentionally conservative.
    """
    # Convert to RGB, just to be safe with modes
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")

    # Light sharpen
    img = img.filter(ImageFilter.SHARPEN)

    # Mild contrast boost
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.05)

    # Mild color boost (only for RGB/RGBA)
    if img.mode in ("RGB", "RGBA"):
        color_enhancer = ImageEnhance.Color(img)
        img = color_enhancer.enhance(1.03)

    return img


def recover_image(src_path: Path, dst_root: Path):
    """
    Attempt to recover a single image file.
    If successful, writes a new file into dst_root with the same name (or .png).
    """
    img = safe_open_image(src_path)
    if img is None:
        print(f"[FAILED] Could not recover: {src_path}")
        return

    if APPLY_LIGHT_ENHANCEMENT:
        try:
            img = lightly_enhance(img)
        except Exception as e:
            print(f"[WARN] Enhancement failed for {src_path}: {e}")

    # Always convert to RGB to avoid weird modes
    try:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
    except Exception:
        pass

    # Decide output format and filename
    # To avoid re-triggering decoder bugs, default to PNG
    dst_ext = ".png"
    dst_name = src_path.stem + dst_ext
    dst_path = dst_root / dst_name

    # If a file with that name exists, add suffix
    counter = 1
    while dst_path.exists():
        dst_path = dst_root / f"{src_path.stem}_{counter}{dst_ext}"
        counter += 1

    try:
        os.makedirs(dst_root, exist_ok=True)
        img.save(dst_path, format="PNG")
        print(f"[OK] Recovered: {src_path} -> {dst_path}")
    except Exception as e:
        print(f"[FAILED] Could not save recovered image for {src_path}: {e}")


# =============================
# MAIN
# =============================

def main():
    failed_dir = Path(FAILED_DIR)
    recovered_dir = Path(RECOVERED_DIR)

    if not failed_dir.exists():
        print(f"FAILED_DIR does not exist: {failed_dir}")
        return

    print(f"Recovering images from: {failed_dir}")
    print(f"Writing recovered images to: {recovered_dir}")
    print()

    all_files = [p for p in failed_dir.iterdir() if p.is_file()]
    img_files = [p for p in all_files if is_image_file(p)]

    print(f"Found {len(all_files)} files, {len(img_files)} image-like files in failed folder.\n")

    for idx, path in enumerate(img_files):
        if (idx + 1) % 50 == 0 or (idx + 1) == len(img_files):
            print(f"[Progress] {idx+1}/{len(img_files)}")

        recover_image(path, recovered_dir)

    print("\nRecovery pass complete.")
    print(f"Recovered images (if any) are under: {recovered_dir}")


if __name__ == "__main__":
    main()