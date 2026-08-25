# agent_tools/embed_tools.py
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
from openai import OpenAI
import numpy as np
from .common import sanitize_edge_metadata, EMBED_MODEL_NAME
from .fs_tools import read_file, list_files
from .database_tools import db_init, db_insert, db_query


# Directly targets Ollama's local high-speed embedding endpoint
LLAMA_EMBED_BASE_URL = os.getenv("LLAMA_EMBED_BASE_URL", "http://localhost:11434/v1")


client = OpenAI(
    base_url=LLAMA_EMBED_BASE_URL,
    api_key="ollama"
)

def _get_raw_embedding(text: str) -> list:
    """Internal helper to fetch raw vectors from Ollama. Hidden from the LLM."""
    clean_text = sanitize_edge_metadata(text).strip()
    if not clean_text:
        logger.warning("Empty text provided for embedding")
        return []
    try:
        logger.info(f"Fetching embedding for text length {len(clean_text)} using model {EMBED_MODEL_NAME}")
        resp = client.embeddings.create(
            model=EMBED_MODEL_NAME, 
            input=[clean_text]
        )
        logger.debug("Successfully retrieved embedding vector")
        return resp.data[0].embedding
    except Exception as e:
        logger.error(f"Ollama embedding request failed: {e}")
        return []

def _compute_cosine(a: list, b: list) -> float:
    """Internal mathematical helper. Hidden from the LLM."""
    try:
        if not a or not b:
            logger.debug("One or both vectors empty for cosine computation")
            return 0.0
        arr_a, arr_b = np.array(a), np.array(b)
        norm_a = np.linalg.norm(arr_a)
        norm_b = np.linalg.norm(arr_b)
        if norm_a == 0 or norm_b == 0:
            logger.debug("Zero norm detected in cosine computation")
            return 0.0
        score = float(np.dot(arr_a, arr_b) / (norm_a * norm_b))
        logger.debug(f"Computed cosine similarity: {score}")
        return score
    except Exception as e:
        logger.error(f"Error during cosine similarity calculation: {e}")
        return 0.0

def compare_similarity(text_a: str, text_b: str) -> str:
    """
    Master tool exposed to the AI Agent. 
    Computes semantic similarity without flooding the LLM context window with raw vectors.
    """
    logger.info(f"Starting semantic comparison between two texts")
    
    vec_a = _get_raw_embedding(text_a)
    vec_b = _get_raw_embedding(text_b)
    
    if not vec_a or not vec_b:
        logger.error("Failed to generate embedding text vectors from Ollama backend.")
        return "[ERROR] Failed to generate embedding text vectors from Ollama backend."
        
    score = _compute_cosine(vec_a, vec_b)
    logger.info(f"Semantic similarity score calculated: {round(score, 4)}")
    return f"The semantic similarity score between the two text selections is: {round(score, 4)}"

# ---------------------------------------------------------------------------
# New RAG-like functionality
# ---------------------------------------------------------------------------

def index_file(file_path: str) -> str:
    """Read a single file, generate its embedding, and store it in the local SQLite DB."""
    logger.info(f"Indexing file: {file_path}")
    
    # Initialize DB table if not exists
    db_init("embeddings", "id INTEGER PRIMARY KEY, source TEXT, content TEXT, embedding TEXT")
    
    # Read content
    content = read_file(file_path)
    if content.startswith("[ERROR]"):
        return f"Failed to index file '{file_path}': {content}"
        
    # Generate embedding
    vec = _get_raw_embedding(content)
    if not vec:
        return f"[ERROR] Failed to generate embedding for '{file_path}'"
    
    # Store in DB
    data = {
        "source": file_path,
        "content": content[:2000],  # Truncate to avoid overly large DB entries
        "embedding": json.dumps(vec)
    }
    
    res = db_insert("embeddings", data)
    if res["status"] == "ok":
        return f"Successfully indexed file '{file_path}' with embedding ID {res['last_row_id']}"
    else:
        return f"[ERROR] Failed to insert into DB: {res['message']}"

def index_folder(folder_path: str) -> str:
    """Scan a folder for text/pdf files, index each one, and store embeddings in the local SQLite DB."""
    logger.info(f"Indexing folder: {folder_path}")
    
    clean_path = sanitize_edge_metadata(folder_path).strip('"').strip("'").replace("\\", "/")
    if not os.path.isdir(clean_path):
        return f"[ERROR] Directory does not exist: {clean_path}"
        
    # Initialize DB table
    db_init("embeddings", "id INTEGER PRIMARY KEY, source TEXT, content TEXT, embedding TEXT")
    
    files = list_files(clean_path)
    indexed_count = 0
    errors = []
    
    for fname in files:
        if not fname.lower().endswith(('.txt', '.pdf', '.md', '.json')):
            continue
        fpath = os.path.join(clean_path, fname)
        if os.path.isdir(fpath):
            continue
            
        res = index_file(fpath)
        if not res.startswith("[ERROR]"):
            indexed_count += 1
        else:
            errors.append(res)
            
    return f"Indexed {indexed_count} files in '{clean_path}'. Errors: {len(errors)}"

def search_embeddings(query_text: str, top_k: int = 5) -> str:
    """Query the local embedding DB for semantically similar content using cosine similarity."""
    logger.info(f"Searching embeddings for query: {query_text[:50]}...")
    
    query_vec = _get_raw_embedding(query_text)
    if not query_vec:
        return "[ERROR] Failed to generate embedding for query text"
        
    rows = db_query("embeddings", "")
    if not rows:
        return "[NO DATA] Embedding database is empty. Please index some files first."
    if "_error" in rows[0]:
        return f"[ERROR] DB query failed: {rows[0]['_error']}"
        
    results = []
    for row in rows:
        try:
            emb_str = row.get("embedding", "[]")
            vec = json.loads(emb_str)
            score = _compute_cosine(query_vec, vec)
            if score > 0.5:  # Filter low-confidence matches
                results.append({
                    "source": row["source"],
                    "content": row.get("content", ""),
                    "similarity_score": round(score, 4)
                })
        except Exception as e:
            logger.debug(f"Error processing row: {e}")
            
    # Sort by score descending
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    top_results = results[:top_k]
    if not top_results:
        return "[NO MATCHES] No sufficiently similar embeddings found."
        
    output = "Top matches:\n"
    for i, res in enumerate(top_results, 1):
        output += f"{i}. Source: {res['source']}\n   Score: {res['similarity_score']}\n   Content: {res['content'][:200]}...\n\n"
    return output.strip()





# _DB_DIR = os.path.join(_BASE_DIR, "embedding_db")
# _DB_FILE = os.path.join(_DB_DIR, "embeddings.db")

# def _ensure_db_dir():
#     """Ensure the database directory exists."""
#     if not os.path.exists(_DB_DIR):
#         os.makedirs(_DB_DIR)

# def _save_db():
#     """Save the current state of the in-memory or file-based DB to disk."""
#     _ensure_db_dir()
#     # If using sqlite3 module directly, we can just ensure the file is written.
#     # Since db_init/db_insert/db_query likely use a global connection or path,
#     # let's assume they point to _DB_FILE. If they use :memory:, we need to attach/backup.
#     # Assuming standard sqlite3 usage in database_tools.py points to _DB_FILE:
#     pass

# def _load_db():
#     """Load the DB from disk if it exists, otherwise initialize empty."""
#     _ensure_db_dir()
#     # If using :memory:, we might need to re-initialize or attach.
#     # For this implementation, let's assume db_init handles the path _DB_FILE.
#     pass

# # ... existing code ...

# def index_file(file_path: str) -> str:
#     """Read a single file, generate its embedding, and store it in the local SQLite DB."""
#     logger.info(f"Indexing file: {file_path}")
    
#     # Initialize DB table if not exists
#     db_init("embeddings", "id INTEGER PRIMARY KEY, source TEXT, content TEXT, embedding TEXT")
    
#     # Read content
#     content = read_file(file_path)
#     if content.startswith("[ERROR]"):
#         return f"Failed to index file '{file_path}': {content}"
        
#     # Generate embedding
#     vec = _get_raw_embedding(content)
#     if not vec:
#         return f"[ERROR] Failed to generate embedding for '{file_path}'"
    
#     # Store in DB
#     data = {
#         "source": file_path,
#         "content": content[:2000],  # Truncate to avoid overly large DB entries
#         "embedding": json.dumps(vec)
#     }
    
#     res = db_insert("embeddings", data)
#     if res["status"] == "ok":
#         _save_db() # Save after successful insert
#         return f"Successfully indexed file '{file_path}' with embedding ID {res['last_row_id']}"
#     else:
#         return f"[ERROR] Failed to insert into DB: {res['message']}"

# def index_folder(folder_path: str) -> str:
#     """Scan a folder for text/pdf files, index each one, and store embeddings in the local SQLite DB."""
#     logger.info(f"Indexing folder: {folder_path}")
    
#     clean_path = sanitize_edge_metadata(folder_path).strip('"').strip("'").replace("\\", "/")
#     if not os.path.isdir(clean_path):
#         return f"[ERROR] Directory does not exist: {clean_path}"
        
#     # Initialize DB table
#     db_init("embeddings", "id INTEGER PRIMARY KEY, source TEXT, content TEXT, embedding TEXT")
    
#     files = list_files(clean_path)
#     indexed_count = 0
#     errors = []
    
#     for fname in files:
#         if not fname.lower().endswith(('.txt', '.pdf', '.md', '.json')):
#             continue
#         fpath = os.path.join(clean_path, fname)
#         if os.path.isdir(fpath):
#             continue
            
#         res = index_file(fpath)
#         if not res.startswith("[ERROR]"):
#             indexed_count += 1
#         else:
#             errors.append(res)
            
#     _save_db() # Save after all indexing is done
#     return f"Indexed {indexed_count} files in '{clean_path}'. Errors: {len(errors)}"

# def search_embeddings(query_text: str, top_k: int = 5) -> str:
    """Query the local embedding DB for semantically similar content using cosine similarity."""
    logger.info(f"Searching embeddings for query: {query_text[:50]}...")
    
    _load_db() # Load latest state from disk
    
    query_vec = _get_raw_embedding(query_text)
    if not query_vec:
        return "[ERROR] Failed to generate embedding for query text"
        
    rows = db_query("embeddings", "")
    if not rows:
        return "[NO DATA] Embedding database is empty. Please index some files first."
    if "_error" in rows[0]:
        return f"[ERROR] DB query failed: {rows[0]['_error']}"
        
    results = []
    for row in rows:
        try:
            emb_str = row.get("embedding", "[]")
            vec = json.loads(emb_str)
            score = _compute_cosine(query_vec, vec)
            if score > 0.5:  # Filter low-confidence matches
                results.append({
                    "source": row["source"],
                    "content": row.get("content", ""),
                    "similarity_score": round(score, 4)
                })
        except Exception as e:
            logger.debug(f"Error processing row: {e}")
            
    # Sort by score descending
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    top_results = results[:top_k]
    if not top_results:
        return "[NO MATCHES] No sufficiently similar embeddings found."
        
    output = "Top matches:\n"
    for i, res in enumerate(top_results, 1):
        output += f"{i}. Source: {res['source']}\n   Score: {res['similarity_score']}\n   Content: {res['content'][:200]}...\n\n"
    return output.strip()    