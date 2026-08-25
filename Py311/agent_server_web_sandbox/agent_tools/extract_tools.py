# agent_tools/extract_tools.py - Text extraction tools for the agent, including number, date, and URL extraction from text.
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

import re
from typing import List

from .common import sanitize_edge_metadata


def extract_numbers(text: str) -> List[float]:
    """Return all integers and floats found in text."""
    logger.info(f"Extracting numbers from text (length: {len(text)})")
    if not isinstance(text, str):
        logger.warning("Non-string input received for extract_numbers")
        return []
    
    clean_text = sanitize_edge_metadata(text)
    # Match optional negative sign, digits, optional decimal part
    nums = re.findall(r"-?\d+(?:\.\d+)?", clean_text)
    result = [float(n) if "." in n else int(n) for n in nums]
    logger.info(f"Extracted {len(result)} numbers")
    return result


def extract_dates(text: str) -> List[str]:
    """Extract common date formats (YYYY-MM-DD, DD.MM.YYYY, DD/MM/YYYY)."""
    logger.info(f"Extracting dates from text (length: {len(text)})")
    if not isinstance(text, str):
        logger.warning("Non-string input received for extract_dates")
        return []
    
    clean_text = sanitize_edge_metadata(text)
    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",       # YYYY-MM-DD (ISO 8601)
        r"\b\d{2}\.\d{2}\.\d{4}\b",     # DD.MM.YYYY (European style)
        r"\b\d{2}/\d{2}/\d{4}\b",       # DD/MM/YYYY (US/Other style)
    ]
    dates = []
    for p in patterns:
        dates.extend(re.findall(p, clean_text))
    unique_dates = list(set(dates))
    logger.info(f"Extracted {len(unique_dates)} unique dates")
    return unique_dates  # Returns unique dates only


def extract_urls(text: str) -> List[str]:
    """Extract all clean HTTP/HTTPS URLs from text."""
    logger.info(f"Extracting URLs from text (length: {len(text)})")
    if not isinstance(text, str):
        logger.warning("Non-string input received for extract_urls")
        return []
    
    clean_text = sanitize_edge_metadata(text)
    # Improved regex to strip trailing punctuation common in chat logs
    urls = re.findall(r"https?://[^\s()<>]+", clean_text)
    # Clean up markdown punctuation tails
    cleaned_urls = [url.rstrip('.,;:)]}') for url in urls]
    logger.info(f"Extracted {len(cleaned_urls)} URLs")
    return cleaned_urls
