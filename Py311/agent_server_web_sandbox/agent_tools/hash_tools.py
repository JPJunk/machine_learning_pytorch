# agent_tools/hash_tools.py - Hashing tools for files and text, with error handling and algorithm support.
import os
import logging

# Configure logging to write to app.log in the project root directory
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_FILE = os.path.join(_BASE_DIR, "app.log")

logging.basicConfig(
    filename=_LOG_FILE,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

import hashlib
from .common import sanitize_edge_metadata

def hash_file(path: str, algo: str = "sha256") -> str:
    """Return hex digest of file safely."""
    logger.info(f"Calculating hash for file: {path} using algorithm: {algo}")
    clean_path = sanitize_edge_metadata(path).strip('"').strip("'").replace("\\", "/")
    if not os.path.exists(clean_path):
        logger.warning(f"File not found: {clean_path}")
        return f"[ERROR] File not found: {clean_path}"
        
    algo_clean = sanitize_edge_metadata(algo).lower().strip()
    if algo_clean not in hashlib.algorithms_available:
        logger.warning(f"Unsupported hash algorithm requested: {algo}")
        return f"[ERROR] Unsupported hash algorithm requested: {algo}. Try 'sha256' or 'md5'."

    try:
        h = hashlib.new(algo_clean)
        with open(clean_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        result = h.hexdigest()
        logger.info(f"Successfully calculated hash for {clean_path}")
        return result
    except Exception as e:
        logger.error(f"Failed calculating file hash for {clean_path}: {e}")
        return f"[ERROR] Failed calculating file hash: {str(e)}"

def hash_text(text: str, algo: str = "sha256") -> str:
    """Return hex digest of text safely."""
    logger.info(f"Calculating hash for text (length: {len(text)}) using algorithm: {algo}")
    if not isinstance(text, str):
        logger.warning("Non-string input received for hash_text")
        return "[ERROR] Hash input must be a text string."
        
    clean_text = sanitize_edge_metadata(text)
    algo_clean = sanitize_edge_metadata(algo).lower().strip()
    if algo_clean not in hashlib.algorithms_available:
        logger.warning(f"Unsupported hash algorithm requested: {algo}")
        return f"[ERROR] Unsupported hash algorithm requested: {algo}."

    try:
        h = hashlib.new(algo_clean)
        h.update(clean_text.encode("utf-8"))
        result = h.hexdigest()
        logger.info(f"Successfully calculated hash for text")
        return result
    except Exception as e:
        logger.error(f"Failed calculating text hash: {e}")
        return f"[ERROR] Failed calculating text hash: {str(e)}"
