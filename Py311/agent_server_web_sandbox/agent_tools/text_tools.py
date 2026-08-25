# agent_tools/text_tools.py - Text processing tools for the agent, including JSON handling and text cleaning.
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

import json
import re
from typing import Any
from .common import sanitize_edge_metadata

def json_load(text: str) -> Any:
    """Parse a JSON text block safely, stripping markdown block ticks if added by the model."""
    logger.info(f"Loading JSON from text (length: {len(text)})")
    if not isinstance(text, str):
        logger.warning("Non-string input received for json_load")
        return {"error": "Input must be a valid text string representation."}
        
    clean_input = sanitize_edge_metadata(text).strip()
    # Heal model formatting artifacts like ```json { ... } ``` wrappers
    if clean_input.startswith("```"):
        logger.debug("Stripping markdown code block from JSON input")
        clean_input = re.sub(r"^```(?:json)?", "", clean_input)
        clean_input = re.sub(r"```$", "", clean_input).strip()
        
    try:
        result = json.loads(clean_input)
        logger.info("JSON loaded successfully")
        return result
    except Exception as e:
        logger.error(f"JSON decoding interface parsing violation: {e}")
        return f"[ERROR] JSON decoding interface parsing violation: {str(e)}"

def json_dump(obj: Any) -> str:
    """Serialize target dictionary arrays or structural data arrays to readable indent formats."""
    logger.info("Serializing object to JSON")
    try:
        # If the input was passed accidentally as a raw string representation of a dict, fix it
        if isinstance(obj, str):
            logger.debug("Input is a string, attempting to parse as JSON first")
            try:
                obj = json.loads(obj)
            except Exception:
                pass
        result = json.dumps(obj, ensure_ascii=False, indent=2)
        logger.info("JSON serialization successful")
        return result
    except Exception as e:
        logger.error(f"JSON serialization formatting process failure: {e}")
        return f"[ERROR] JSON serialization formatting process failure: {str(e)}"

def clean_text(text: str) -> str:
    """Compresses duplicate line tabs, newline breaks, and structural blank spaces into clean text arrays."""
    logger.info(f"Cleaning text (length: {len(text)})")
    if not isinstance(text, str):
        logger.warning("Non-string input received for clean_text")
        return ""
    result = re.sub(r"\s+", " ", sanitize_edge_metadata(text)).strip()
    logger.debug(f"Cleaned text length: {len(result)}")
    return result

def split_lines(text: str) -> list:
    """Splits block strings into arrays of clean line strings while automatically omitting blanks."""
    logger.info(f"Splitting text into lines (length: {len(text)})")
    if not isinstance(text, str):
        logger.warning("Non-string input received for split_lines")
        return []
    result = [line.strip() for line in sanitize_edge_metadata(text).splitlines() if line.strip()]
    logger.info(f"Split into {len(result)} lines")
    return result
