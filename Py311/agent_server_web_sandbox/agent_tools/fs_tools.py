# agent_tools/fs_tools.py - File system tools for the agent, including reading/writing files and handling PDFs.
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

import shutil
from pypdf import PdfReader

from .common import sanitize_edge_metadata

# Maximum allowed file size for reading (10 MB) to prevent memory exhaustion
MAX_FILE_SIZE = 10 * 1024 * 1024

def read_file(path: str) -> str:
    """Read a text file or extract raw text from a PDF file from disk safely."""
    logger.info(f"Reading file: {path}")
    # Siivotaan polku mahdollisista ylimääräisistä lainausmerkeistä ja Edge placeholder tag-geista
    clean_path = sanitize_edge_metadata(path).strip('"').strip("'").replace("\\", "/")
    
    if not os.path.exists(clean_path):
        logger.warning(f"File does not exist: {clean_path}")
        return f"[ERROR] File does not exist at: {clean_path}"
    if os.path.isdir(clean_path):
        logger.warning(f"Path is a directory, not a file: {clean_path}")
        return f"[ERROR] Path is a directory, not a file: {clean_path}"
        
    # Check for symlink to prevent following unexpected links
    if os.path.islink(clean_path):
        real_path = os.path.realpath(clean_path)
        if not os.path.exists(real_path):
            logger.warning(f"Symlink points to non-existent target: {clean_path}")
            return f"[ERROR] Symlink points to non-existent target: {clean_path}"

    # --- PDF Tiedostojen automaattinen tunnistus ja purku ---
    if clean_path.lower().endswith(".pdf"):
        logger.info(f"Detected PDF file, attempting extraction: {clean_path}")
        try:
            reader = PdfReader(clean_path)
            extracted_text = []
            
            # Käydään sivut läpi ja eristetään teksti muistiin
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    extracted_text.append(f"--- PAGE {i+1} ---")
                    extracted_text.append(page_text)
            
            if not extracted_text:
                logger.warning(f"PDF file '{clean_path}' was read, but no digital text layer could be extracted")
                return f"[WARNING] PDF file '{clean_path}' was read, but no digital text layer could be extracted (it might be a scanned image or restricted)."
                
            logger.info(f"Successfully extracted text from PDF: {clean_path}")
            return "\n".join(extracted_text)
        except ImportError:
            logger.error("pypdf package not found for PDF extraction")
            return "[ERROR] Cannot read PDF. Please install 'pypdf' package in your venv: pip install pypdf"
        except Exception as e:
            logger.error(f"Failed to parse local PDF file {clean_path}: {e}")
            return f"[ERROR] Failed to parse local PDF file: {str(e)}"

    # --- Normaalit tekstitiedostot (.txt, .md, .json jne.) ---
    try:
        # Check file size before reading to prevent memory exhaustion
        file_size = os.path.getsize(clean_path)
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"File too large ({file_size} bytes) for reading: {clean_path}")
            return f"[ERROR] File is too large ({file_size} bytes). Max allowed is {MAX_FILE_SIZE} bytes."
            
        logger.info(f"Reading text file: {clean_path} ({file_size} bytes)")
        with open(clean_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except UnicodeDecodeError:
        # Varamekanismi jos tiedosto on jossain muussa kuin UTF-8 muodossa (esim. Windows ANSI)
        logger.info(f"UTF-8 decode failed for {clean_path}, trying cp1252")
        try:
            with open(clean_path, "r", encoding="cp1252") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Binary or layout character decoding failure for {clean_path}: {e}")
            return f"[ERROR] Binary or layout character decoding failure: {str(e)}"
    except Exception as e:
        logger.error(f"Failed to read file {clean_path}: {e}")
        return f"[ERROR] Failed to read file: {str(e)}"

def write_file(path: str, content: str) -> str:
    """Write text content to a file, auto-creating directories if missing."""
    logger.info(f"Writing to file: {path} ({len(content)} bytes)")
    try:
        clean_path = sanitize_edge_metadata(path).strip('"').strip("'").replace("\\", "/")
        dirname = os.path.dirname(clean_path)
        if dirname and not os.path.exists(dirname):
            logger.info(f"Creating directory: {dirname}")
            os.makedirs(dirname, exist_ok=True)
            
        with open(clean_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Successfully wrote to file: {clean_path}")
        return f"Success: Written to {clean_path}"
    except Exception as e:
        logger.error(f"Failed to write file {path}: {e}")
        return f"[ERROR] Failed to write file: {str(e)}"

def append_file(path: str, content: str) -> str:
    """Append text content to a file, auto-creating directories if missing."""
    logger.info(f"Appending to file: {path} ({len(content)} bytes)")
    try:
        clean_path = sanitize_edge_metadata(path).strip('"').strip("'").replace("\\", "/")
        dirname = os.path.dirname(clean_path)
        if dirname and not os.path.exists(dirname):
            logger.info(f"Creating directory: {dirname}")
            os.makedirs(dirname, exist_ok=True)
            
        with open(clean_path, "a", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Successfully appended to file: {clean_path}")
        return f"Success: Appended to {clean_path}"
    except Exception as e:
        logger.error(f"Failed to append to file {path}: {e}")
        return f"[ERROR] Failed to append to file: {str(e)}"

def copy_file(src: str, dst: str) -> str:
    """Copy a file safely."""
    logger.info(f"Copying file: {src} -> {dst}")
    clean_src = sanitize_edge_metadata(src).strip('"').strip("'").replace("\\", "/")
    clean_dst = sanitize_edge_metadata(dst).strip('"').strip("'").replace("\\", "/")
    if not os.path.exists(clean_src):
        logger.warning(f"Source file does not exist: {clean_src}")
        return f"[ERROR] Source file does not exist: {clean_src}"
    try:
        shutil.copy2(clean_src, clean_dst)
        logger.info(f"Successfully copied {clean_src} to {clean_dst}")
        return f"Success: Copied {clean_src} to {clean_dst}"
    except Exception as e:
        logger.error(f"Failed to copy file from {clean_src} to {clean_dst}: {e}")
        return f"[ERROR] Failed to copy file: {str(e)}"

def move_file(src: str, dst: str) -> str:
    """Move or rename a file safely."""
    logger.info(f"Moving file: {src} -> {dst}")
    clean_src = sanitize_edge_metadata(src).strip('"').strip("'").replace("\\", "/")
    clean_dst = sanitize_edge_metadata(dst).strip('"').strip("'").replace("\\", "/")
    if not os.path.exists(clean_src):
        logger.warning(f"Source file does not exist: {clean_src}")
        return f"[ERROR] Source file does not exist: {clean_src}"
    try:
        shutil.move(clean_src, clean_dst)
        logger.info(f"Successfully moved {clean_src} to {clean_dst}")
        return f"Success: Moved {clean_src} to {clean_dst}"
    except Exception as e:
        logger.error(f"Failed to move file from {clean_src} to {clean_dst}: {e}")
        return f"[ERROR] Failed to move file: {str(e)}"

def delete_file(path: str) -> str:
    """Delete a file safely."""
    logger.info(f"Deleting file: {path}")
    clean_path = sanitize_edge_metadata(path).strip('"').strip("'").replace("\\", "/")
    if not os.path.exists(clean_path):
        logger.warning(f"Target file does not exist: {clean_path}")
        return f"[ERROR] Target file does not exist: {clean_path}"
    if os.path.isdir(clean_path):
        logger.warning(f"Path is a directory. Delete tool only processes files: {clean_path}")
        return f"[ERROR] Path is a directory. Delete tool only processes files: {clean_path}"
    try:
        os.remove(clean_path)
        logger.info(f"Successfully deleted file: {clean_path}")
        return f"Success: Deleted file {clean_path}"
    except Exception as e:
        logger.error(f"Failed to delete file {clean_path}: {e}")
        return f"[ERROR] Failed to delete file: {str(e)}"

def list_files(path: str) -> list:
    """List files in a directory safely."""
    logger.info(f"Listing files in directory: {path}")
    clean_path = sanitize_edge_metadata(path).strip('"').strip("'").replace("\\", "/")
    if not os.path.exists(clean_path):
        logger.warning(f"Directory does not exist: {clean_path}")
        return [f"[ERROR] Directory does not exist: {clean_path}"]
    if not os.path.isdir(clean_path):
        logger.warning(f"Path is a file, not a directory: {clean_path}")
        return [f"[ERROR] Path is a file, not a directory: {clean_path}"]
    try:
        files = os.listdir(clean_path)
        logger.info(f"Listed {len(files)} items in {clean_path}")
        return files
    except Exception as e:
        logger.error(f"Failed to list directory contents for {clean_path}: {e}")
        return [f"[ERROR] Failed to list directory contents: {str(e)}"]
