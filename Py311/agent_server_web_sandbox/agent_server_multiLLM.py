# agent_server_memory.py - A local agent server with FastAPI web interface and tools support.

import logging
import asyncio
import inspect

logging.basicConfig(filename="app.log", level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

import datetime
import time
import threading
import os
import json
import base64
import textwrap
from typing import Dict, Any, List, AsyncGenerator

from pydantic import BaseModel

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import uvicorn

# from agent_tools.common_memory import memory_conn
# conn = memory_conn()
from agent_tools.memory import messages_history, SYSTEM_PROMPT
from agent_tools.memory import client, LLAMA_MODEL, API_KEY, LLAMA_BASE_URL
from agent_tools.memory import sanitize_edge_metadata

from agent_tools.memory import memory_consolidate_mid_to_long, memory_consolidate_short_to_mid
from agent_tools.memory import memory_get_for_prompt, memory_store, llm_memory
from agent_tools.memory import estimate_tokens_messages, check_and_cleanup_memory
from agent_tools.memory import recall_relevant_context, limit_recall_by_tokens
from agent_tools.memory import auto_memorise_and_reset, load_system_prompt

from agent_tools import TOOLS 


# agent_server.py - ADD TO CONFIG/UTILS SECTION
import re
from agent_tools.extract_tools import extract_urls
from openai import OpenAI

# --- ROUTING TABLE WITH MODELS ---

# MODEL_ROUTER = {
#     "director":      {"model": "qwen2-1.5b-moe-director", "endpoint": "..."},
#     "summarization": {"model": "phi-4-mini-instruct", "endpoint": "..."},
#     "extraction":    {"model": "phi-4-mini-instruct", "endpoint": "..."},
#     "translation":   {"model": "marianmt", "endpoint": "..."},
#     "code_fix":      {"model": "qwen3-desert.coder.moe-8x0.6b-i1", "endpoint": "..."},
#     "code_complex":  {"model": "carwin-moe-nano", "endpoint": "..."},
#     "reasoning":     {"model": "carwin-moe-nano", "endpoint": "..."},
#     "planning":      {"model": "carwin-moe-nano", "endpoint": "..."},
#     "creative_short":{"model": "phi-4-mini-instruct", "endpoint": "..."},
#     "creative":      {"model": "llama-3.2-8x4b-moe-v2-dark-champion", "endpoint": "..."},
#     "default":       {"model": "carwin-moe-nano", "endpoint": "..."}
# }



MODEL_ROUTER = {
    # qwen2-1.5b-instruct_moe_director # TODO: CLASSIFIER, GPU, 986mb
    # qwen2.5-7b-instruct-qwq-v0.1
    "summarization": {"model": "phi-4-mini-instruct", "endpoint": "http://127.0.0.1:5001/v1"}, #GPU, 2.94gb
    "code_fix":     {"model": "qwen3-desert.coder.moe-8x0.6b-i1", "endpoint": "http://127.0.0.1:5001/v1"}, #GPU, 1.55gb
    "extraction":   {"model": "phi-4-mini-instruct", "endpoint": "http://127.0.0.1:5001/v1"},
    # "translation":  {"model": "Qwen2.5‑7B", "endpoint": "http://127.0.0.1:5001/v1"}, #TODO: MARIANDB
    "reasoning":    {"model": "carwin-moe-nano", "endpoint": "http://127.0.0.1:5001/v1"}, #CPU, 12,34gb
    "planning":     {"model": "carwin-moe-nano", "endpoint": "http://127.0.0.1:5001/v1"},
    "creative":     {"model": "llama-3.2-8x4b-moe-v2-dark-champion-instruct-uncensored-abliterated-21b", "endpoint": "http://127.0.0.1:5001/v1"}, #CPU, 12.12gb
    "default":      {"model": "carwin-moe-nano", "endpoint": "http://127.0.0.1:5001/v1"}
}

# qwen3-desert.coder.moe-8x0.6b-i1, 1.55GB
# phi-4-mini-instruct, 2.94GB
# gemma4-e4b-qat-claude-mythos-distilled, 5.30GB
# llama-3.2-8x4b-moe-v2-dark-champion-instruct-uncensored-abliterated-21b, 12.12GB
# carwin-moe-nano, 12.34GB

# summarization	Qwen2.5‑7B	fast + accurate ? Phi‑3‑mini
# extraction	Phi‑3‑mini	best JSON/extraction
# translation	Qwen2.5‑7B	faster + more multilingual ??? Gemma‑2‑9B‑IT
# code_fix	DeepSeek‑Coder‑V2‑Lite	fast first pass ? Qwen2.5‑Coder‑7B
# creative	Gemma‑2‑9B‑IT	best creative writing ??? Mistral‑7B‑Instruct ? Phi‑3‑mini ? Qwen2.5‑7B
# planning	Qwen3.6‑35B‑A3B	deep reasoning
# reasoning	Qwen3.6‑35B‑A3B	deep reasoning
# default	Qwen3.6‑35B‑A3B	safe fallback

def compute_task_fingerprint(text: str) -> Dict[str, Any]:
    """Extracts structural signals from the prompt to aid routing."""
    has_code = bool(re.search(r'```', text))
    has_json = bool(re.search(r'\{.*\}', text, re.DOTALL))
    # has_urls = len(extract_urls(text > 0))
    has_urls = len(extract_urls(text))
    has_stacktrace = bool(re.search(r'Traceback|Error:|Exception:', text))
    
    # Simple complexity heuristic based on length and depth
    lines = text.split('\n')
    depth = max(len(line) for line in lines) if lines else 0
    
    return {
        "has_code": has_code,
        "has_json": has_json,
        "has_urls": has_urls,
        "has_stacktrace": has_stacktrace,
        "complexity_score": min(1.0, (len(text) / 2000) + (depth / 100)) # Normalized 0-1
    }

def classify_task(user_input: str) -> Dict[str, Any]:
    """Hybrid classifier: Rules -> Fingerprint -> Tiny Model Fallback."""
    fp = compute_task_fingerprint(user_input)
    text = user_input.lower().strip()
    
    # 1. Rule-based fast path (covers ~70% of cases instantly)
    if any(kw in text for kw in ["translate", "translation", "in finnish", "in swedish"]):
        return {"task_type": "translation", "confidence": 0.95, "fingerprint": fp}
    if any(kw in text for kw in ["extract", "list", "find", "count", "how many"]):
        return {"task_type": "extraction", "confidence": 0.90, "fingerprint": fp}
    if any(kw in text for kw in ["summarize", "summary", "brief", "overview"]):
        return {"task_type": "summarization", "confidence": 0.95, "fingerprint": fp}
    if any(kw in text for kw in ["fix", "debug", "error", "bug", "why does this fail"]):
        return {"task_type": "code_fix", "confidence": 0.85, "fingerprint": fp}
    if any(kw in text for kw in ["plan", "architecture", "design", "how to build"]):
        return {"task_type": "planning", "confidence": 0.80, "fingerprint": fp}
    if any(kw in text for kw in ["write", "create", "story", "poem", "idea", "brainstorm"]):
        return {"task_type": "creative", "confidence": 0.75, "fingerprint": fp}
        
    # 2. Fallback to a tiny model for deep classification (only if rules fail)
    # This runs only when heuristics are ambiguous, keeping latency low
    try:
        resp = client.chat.completions.create(
            model="phi-3-mini", 
            messages=[{"role": "user", "content": f"Classify this prompt into one of these categories: reasoning, code_fix, summarization, extraction, translation, creative, planning. Output ONLY the category name.\n\nPROMPT: {text}"}],
            temperature=0.0
        )
        task = resp.choices[0].message.content.strip().lower()
        return {"task_type": task, "confidence": 0.60, "fingerprint": fp} # Lower confidence for fallback
    except Exception as e:
        logging.warning(f"Classifier fallback failed: {e}")
        return {"task_type": "default", "confidence": 0.50, "fingerprint": fp}


app = FastAPI(title="Local Agent Server Console")
templates = Jinja2Templates(directory="templates")

# Setup dedicated uploads folder in project directory root
UPLOAD_DIR = os.path.abspath("uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Max upload size: 50 MB
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

class ChatPayload(BaseModel):
    message: str


def build_tool_schema() -> list[Dict[str, Any]]:
    schemas = []
    def add(name: str, description: str, params: dict):
        required_params = [k for k, v in params.items() if "default" not in v]
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": required_params,
                },
            },
        })

    add("web_search", "Search the web for information.", {"query": {"type": "string"}})
    add("web_scrape", "Scrape visible text from a web page.", {"url": {"type": "string"}})
    
    # web_deep_search(query: str, max_depth: int = 1) -> str:
    add("web_deep_search", "Automatic information query pipeline", {"query": {"type": "string"}})

    add("read_file", "Read a text file from disk.", {"path": {"type": "string"}})
    add("write_file", "Write text content to a file (overwrite)", {"path": {"type": "string"}, "content": {"type": "string"}})
    add("append_file", "Append text content to a file.", {"path": {"type": "string"}, "content": {"type": "string"}})
    add("copy_file", "Copy a file.", {"src": {"type": "string"}, "dst": {"type": "string"}})
    add("move_file", "Move or rename a file.", {"src": {"type": "string"}, "dst": {"type": "string"}})
    add("delete_file", "Delete a file.", {"path": {"type": "string"}})
    add("list_files", "List files in a directory.", {"path": {"type": "string"}})
    add("db_init", "Create a table if it does not already exist.", {
        "table_name": {"type": "string", "description": "Name of the table to create (sanitized)."},
        "schema": {"type": "string", "description": "Raw SQL column definitions, e.g. 'name TEXT, age INTEGER'."}
    })

    add("db_insert", "Insert a single row into the given table.", {
        "table_name": {"type": "string", "description": "Target table name (sanitized)."},
        "data": {"type": "object", "description": "Mapping of column names to values."}
    })

    add("db_query", "Query rows from a table.", {
        "table_name": {"type": "string", "description": "Target table name (sanitized)."},
        "where": {"type": "string", "description": "Optional WHERE clause without the keyword 'WHERE', e.g. 'age > 18'."}
    })

    add("db_update", "Update rows matching a WHERE clause.", {
        "table_name": {"type": "string", "description": "Target table name (sanitized)."},
        "data": {"type": "object", "description": "Mapping of column names to new values."},
        "where": {"type": "string", "description": "WHERE clause without the keyword 'WHERE'."}
    })

    add("db_delete", "Delete rows matching a WHERE clause.", {
        "table_name": {"type": "string", "description": "Target table name (sanitized)."},
        "where": {"type": "string", "description": "WHERE clause without the keyword 'WHERE'."}
    })    
    add("json_load", "Parse a JSON string into an object.", {"text": {"type": "string"}})
    add("json_dump", "Serialize an object to pretty JSON.", {"obj": {"type": "string"}})
    add("clean_text", "Normalize whitespace in text.", {"text": {"type": "string"}})
    add("split_lines", "Split text into non-empty trimmed lines.", {"text": {"type": "string"}})
    add("compare_similarity", "Compute the mathematical semantic similarity score between two text strings.", {"text_a": {"type": "string"}, "text_b": {"type": "string"}})
    add("detect_language", "Detect the language of text.", {"text": {"type": "string"}})
    add("summarize", "Generate a summary of the input text.", {"text": {"type": "string"}})
    add("translate", "Translate text using the local translation engine.", {"text": {"type": "string"}, "target_lang": {"type": "string"}})
    add("extract_numbers", "Extract numbers from text.", {"text": {"type": "string"}})
    add("extract_urls", "Extract URLs from text.", {"text": {"type": "string"}})
    add("extract_dates", "Extract dates from text.", {"text": {"type": "string"}})
    add("hash_file", "Compute hash of a file.", {"path": {"type": "string"}, "algo": {"type": "string", "default": "sha256"}})
    add("hash_text", "Compute hash of text.", {"text": {"type": "string"}, "algo": {"type": "string", "default": "sha256"}})
    add("analyze_image", "Analyze an image file locally and return a description.", {"path": {"type": "string"}})
    add("generate_image", "Generate an image using ComfyUI Stable Cascade.", {"prompt": {"type": "string"}})
    add("transcribe_audio", "Convert a local 16kHz mono WAV audio file into written text.", {"path": {"type": "string"}})
    add("speak_text", "Synthesize written text into a local Finnish spoken audio WAV file.", {"text": {"type": "string"}, "output_path": {"type": "string", "default": "C:/test/speech.wav"}})
    add("generate_music", "Generate music based on a text description using MusicGen locally on CPU.", {"prompt": {"type": "string"}, "duration": {"type": "integer", "default": 5, "description": "Duration of the generated music in seconds (default 5s due to CPU constraints)."}})
    # add("generate_sound_effect", "Generate sound effects based on a text description using AudioGen locally on CPU.", {"prompt": {"type": "string"}, "duration": {"type": "integer", "default": 3, "description": "Duration of the generated sound effect in seconds (default 3s)."}})
    add("run_python", "Execute a local Python script as a subprocess and capture stdout, stderr, and exit code.", {
        "path": {"type": "string", "description": "Absolute path to the Python script to execute."},
        "args": {"type": "string", "description": "Optional JSON array of command-line arguments to pass to the script.", "default": "[]"},
        "timeout": {"type": "integer", "default": 30, "description": "Max seconds to allow the script to run before forcefully terminating it."},
        "cwd": {"type": "string", "description": "Optional working directory for the process. If not provided, uses the script's directory."}}
    )

    add("create_symlink", "Create a symbolic link on Windows.", {
        "target": {"type": "string", "description": "Real file or folder path."},
        "link_path": {"type": "string", "description": "Where the symlink should be created."},
        "is_directory": {"type": "boolean", "description": "True if linking a directory, False for a file."}
    })

    add("get_memory_usage", "Get current memory usage of the system and Python process.", {})
    add("optimize_context_window", "Trim the context window to fit within a specified token limit.", {
        "max_tokens": {"type": "integer", "default": 32768, "description": "Maximum tokens to keep in context"}
    })
    add("cleanup_memory_db", "Clean up the agent memory database by removing old short-term memories and compacting.", {})
    add("get_system_health", "Get comprehensive system health metrics including CPU, memory, disk, and LLM status.", {})
    add("force_gc_and_free_memory", "Force garbage collection and attempt to free Python memory.", {})
    add("list_active_processes", "List all active processes with memory and CPU usage.", {})
    add("clear_upload_cache", "Clear the uploads directory of temporary files older than 1 hour.", {})
    add("save_agent_state", "Save current agent state (messages, system prompt) to a JSON file for persistence.", {
        "path": {"type": "string", "default": "agent_state.json", "description": "Path to save the state file"}
    })
    add("load_agent_state", "Load agent state from a JSON file and update global variables.", {
        "path": {"type": "string", "default": "agent_state.json", "description": "Path to load the state file from"}
    })

    logging.info(f"Built tool schema with {len(schemas)} tools.")
    for schema in schemas:
        logging.info(f"Tool: {schema['function']['name']} - {schema['function']['description']}") 
    return schemas

TOOLS_SCHEMA = build_tool_schema()


def _sanitize_args(arguments: dict) -> dict:
    """Sanitize every string value inside a tool-argument dict."""
    sanitized = {}
    for k, v in arguments.items():
        if isinstance(v, str):
            sanitized[k] = sanitize_edge_metadata(v)
        elif isinstance(v, (dict, list)):
            # Deep-sanitize nested dicts/lists that contain strings
            def _deep_clean(obj):
                if isinstance(obj, dict):
                    return {kk: _deep_clean(vv) for kk, vv in obj.items()}
                if isinstance(obj, list):
                    return [_deep_clean(item) for item in obj]
                if isinstance(obj, str):
                    return sanitize_edge_metadata(obj)
                return obj
            sanitized[k] = _deep_clean(v)
        else:
            sanitized[k] = v
    return sanitized


def execute_tool_call(name: str, arguments: dict) -> str:
    # Sanitize all string arguments before dispatching to the tool
    sanitized_args = _sanitize_args(arguments)

    if name not in TOOLS:
        logging.error(f"Unknown tool requested: {name}")
        return f"[ERROR] Unknown tool: {name}"
    fn = TOOLS[name]
    try:
        result = fn(**sanitized_args)
    except TypeError:
        result = fn(sanitized_args)
    except Exception as e:
        logging.error(f"Error occurred while executing tool {name}: {e}")
        return f"[ERROR] Tool {name} failed: {e}"

    if isinstance(result, (dict, list)):
        return json.dumps(result, ensure_ascii=False, indent=2)
    logging.info(f"Tool {name} executed successfully. Result type: {type(result)}, length: {len(str(result))} characters.")
    return str(result)

# agent_server.py - ADD TO UTILS SECTION
import hashlib
from datetime import datetime, timedelta

# Simple in-memory cache for routing decisions (TTL: 5 minutes)
ROUTING_CACHE = {}
CACHE_TTL = 300 # seconds

def get_cached_routing(task_type: str, fingerprint: dict) -> dict:
    """Retrieves cached routing if available."""
    cache_key = f"{task_type}:{hashlib.md5(json.dumps(fingerprint).encode()).hexdigest()}"
    entry = ROUTING_CACHE.get(cache_key)
    if entry and datetime.now() < entry['expires']:
        return entry['data']
    return None

def set_cached_routing(task_type: str, fingerprint: dict, data: dict):
    """Stores routing decision in cache."""
    cache_key = f"{task_type}:{hashlib.md5(json.dumps(fingerprint).encode()).hexdigest()}"
    ROUTING_CACHE[cache_key] = {
        'data': data,
        'expires': datetime.now() + timedelta(seconds=CACHE_TTL)
    }

def validate_output(task_type: str, output: str) -> Dict[str, Any]:
    """Validates output based on task-specific rules."""
    if task_type == "code_fix":
        has_backticks = '```' in output
        has_language = any(lang in output.lower() for lang in ['python', 'javascript', 'typescript', 'rust', 'go'])
        return {
            "quality": "good" if (has_backticks and has_language) else "bad",
            "needs_escalation": not (has_backticks and has_language)
        }
    elif task_type == "translation":
        # Simple check: does it contain non-target language words? (Simplified for demo)
        return {"quality": "good" if len(output) > 10 else "bad", "needs_escalation": len(output) < 10}
    elif task_type == "extraction":
        # Check for list format or JSON
        is_list = output.strip().startswith('-') or output.strip().startswith('*')
        is_json = output.strip().startswith('{')
        return {"quality": "good" if (is_list or is_json) else "bad", "needs_escalation": not (is_list or is_json)}
    else:
        # Default: check length
        return {"quality": "good" if len(output) > 50 else "bad", "needs_escalation": len(output) < 50}

# ---------------- ABORT MECHANISM (llama_set_abort_callback style) ----------------
_ABORT_REQUESTED = threading.Event()


@app.post("/api/abort")
async def api_abort():
    """Signal the running agent to stop generating."""
    _ABORT_REQUESTED.set()
    logging.info("Abort requested via API.")
    return {"status": "abort_requested"}

@app.get("/api/health")
async def api_health():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "uptime_seconds": time.time() - _START_TIME,
        "upload_dir": UPLOAD_DIR,
        "memory_status": "active" if messages_history else "empty"
    }

# Store start time for uptime calculation
_START_TIME = time.time()


def _check_abort() -> bool:
    """Return True if abort was requested, then clear the flag for next run."""
    if _ABORT_REQUESTED.is_set():
        _ABORT_REQUESTED.clear()
        logging.info("Abort flag cleared.")
        return True
    return False

# ---------------- API ENDPOINTS ----------------

@app.get("/", response_class=HTMLResponse)
async def render_interface():
    logging.info("Serving main interface HTML.")
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/view-image")
async def view_local_image(path: str):
    clean_path = path.strip('"').strip("'").replace("\\", "/")
    logging.info(f"Received request to view image at path: {clean_path}")
    if os.path.exists(clean_path):
        return FileResponse(clean_path)
    return HTMLResponse(status_code=404, content="Image not found")

@app.post("/api/clear")
async def clear_session_memory():
    global messages_history
    logging.info("Clearing session memory.")
    messages_history = [{"role": "system", "content": textwrap.dedent(SYSTEM_PROMPT).strip()}]
    return {"status": "cleared", "logs": ["Agent memory stack flushed successfully."]}

@app.post("/api/memorise")
async def api_memorise():
    global SYSTEM_PROMPT, messages_history   # <-- MUST be first

    # Flatten messages_history into readable text
    parts = []
    for m in messages_history:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                c["text"] for c in content if isinstance(c, dict) and c.get("type") == "text"
            )
        parts.append(f"[{role.upper()}]\n{content}")

    full_context = "\n\n".join(parts)

    # Check if context is empty or just system prompt
    if not full_context or len(full_context) < 10:
        return {"status": "ok", "summary": "No significant context to memorize."}

    prompt = f"""
Summarise the important, reusable information from the following context:
- decisions
- constraints
- plans
- conventions
- important facts
- long-term goals
Concentrate on code-relevant and agent-relevant information that would be useful to remember for future interactions.

Be concise but precise. This is SHORT-TERM memory.

CONTEXT:
{full_context}
"""

    distilled = llm_memory(prompt)
    memory_store("short", distilled)

    mid_term_memory = memory_consolidate_short_to_mid()
    long_term_memory = memory_consolidate_mid_to_long()

    logging.info("Memorisation complete. Short-term memory updated, and consolidation attempted.")
    logging.info(f"Short-term memory summary: {distilled[:200]}...")
    logging.info(f"Mid-term memory result: {mid_term_memory}")
    logging.info(f"Long-term memory result: {long_term_memory}")


    # Rebuild SYSTEM_PROMPT
    SYSTEM_PROMPT = load_system_prompt() + "\n\n"
    mem_rows = memory_get_for_prompt()
    if mem_rows:
        mem_text = "\n\n".join(f"[{lvl.upper()}]\n{summary}" for lvl, summary in mem_rows)
        SYSTEM_PROMPT += "\n\n" + mem_text

    logging.info(f"\n{SYSTEM_PROMPT}\n")

    # Reset context
    messages_history = [{"role": "system", "content": textwrap.dedent(SYSTEM_PROMPT).strip()}]

    return {"status": "ok", "summary": distilled}


@app.post("/api/sleep")
async def api_sleep():
    result = memory_consolidate_short_to_mid()
    return {"status": "slept", "result": str(result)}

# FIXED: Added missing @app.post decorator
@app.post("/api/deep-sleep")
async def api_deep_sleep():
    result = memory_consolidate_mid_to_long()
    return {"status": "deep_slept", "result": str(result)}

@app.post("/api/upload")
async def upload_file_handler(file: UploadFile = File(...)):
    """Receives file from browser, writes it locally, and sends path string back to frontend."""
    try:
        # Check size limit
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_SIZE // (1024*1024)} MB.")
        
        filename_str = str(file.filename)
        file_path = os.path.join(UPLOAD_DIR, filename_str).replace("\\", "/")
        
        with open(file_path, "wb") as f:
            f.write(content)
            
        _, ext = os.path.splitext(filename_str)
        ext_clean = ext.lower()
        
        is_image = ext_clean in [".png", ".jpg", ".jpeg", ".webp"]
        logging.info(f"File uploaded: {filename_str} (Type: {ext_clean}, Image: {is_image})")

        return {"status": "success", "local_path": file_path, "is_image": is_image}
    except Exception as e:
        import traceback
        logging.error(f"Error during file upload: {e}")
        return {"status": "error", "error": str(e)}


# --- GLOBAALIT AIKAMUUTTUJAT PÄÄAGENTIN AIKATUNNUS VARTEN ---
LAST_INTERACTION_TIME = time.time()


def _sse_chunk(event_type: str, data: Any) -> str:
    """Format a Server-Sent Event chunk."""
    payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
    logging.debug(f"Emitting SSE chunk: {payload}")
    return f"event: {event_type}\ndata: {payload}\n\n"

async def chat_stream(payload: ChatPayload) -> AsyncGenerator[str, None]:
    global messages_history, LAST_INTERACTION_TIME

    # --- AUTO-MEMORISATION WHEN CONTEXT TOO LARGE ---
    context_tokens = estimate_tokens_messages(messages_history)

    if context_tokens > 24000:  # soft threshold
        logging.info(f"Context size {context_tokens} > 24k, auto-memorising...")
        try:
            # Check if async or sync to avoid await errors
            if inspect.iscoroutinefunction(auto_memorise_and_reset):
                await auto_memorise_and_reset()
            else:
                # Run in thread to avoid blocking the event loop
                await asyncio.to_thread(auto_memorise_and_reset)
        except Exception as e:
            logging.error(f"Auto-memorisation failed: {e}")

    try:
        if check_and_cleanup_memory():
            yield _sse_chunk("status", "🧠 High memory detected. Auto-cleanup triggered.")
    except Exception as e:
        logging.error(f"Memory cleanup failed (likely SQLite lock): {e}")

    # --- Centralized sanitization of user input at the very first entry point ---
    raw_message = payload.message if isinstance(payload.message, str) else ""
    user_input = sanitize_edge_metadata(raw_message).strip()
    execution_logs = []
    # execution_logs: list[str] = []

    if not user_input:
        yield _sse_chunk("error", "Empty prompt.")
        return

    # --- DYNAAMINEN AIKATUNNISTUS (TIME CONTEXT) GENERATION ---
    now_time = datetime.now()
    current_time_str = now_time.strftime("%A, %B %d, %Y (Aika: %H:%M:%S)")

    current_unix_time = time.time()
    elapsed_minutes = int((current_unix_time - LAST_INTERACTION_TIME) / 60)
    LAST_INTERACTION_TIME = current_unix_time

    time_lock_prefix = (
        f"[TIME CONTEXT: The current host system date and time is strictly {current_time_str[:-3]} local time in Kuopio, Finland. "
        f"It has been exactly {elapsed_minutes} minutes since the user last interacted with you. "
    )

    is_image_syntax = user_input.lower().startswith("image:")
    words = user_input.split()
    is_raw_image_path = (
        any(
            any(words[i].lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"])
            for i in range(len(words))
        )
        if words
        else False
    )

    # --- MULTIMODAALINEN JA TEKSTIPOHJAINEN SYÖTTEEN KÄSITTELY ---
    if is_image_syntax or is_raw_image_path:
        if is_image_syntax:
            clean_input = user_input[6:].strip()
            parts = clean_input.split(" ", 1)
            img_path = parts[0]
            prompt_text = parts[1] if len(parts) > 1 else "Describe this image."
        else:
            if " " not in user_input:
                img_path = user_input
                prompt_text = "Describe this image."
            else:
                img_path = None
                prompt_text = None

        if img_path:
            img_path = img_path.strip('"').strip("'").replace("\\", "/")

            if not os.path.exists(img_path):
                error_msg = f"[System Error] File does not exist at path: {img_path}"
                logging.error(error_msg)
                yield _sse_chunk("error", error_msg)
                return

            try:
                with open(img_path, "rb") as f:
                    b64_data = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(img_path)[1].lower().replace(".", "")
                mime = (
                    f"image/{ext}"
                    if ext in ["png", "jpg", "jpeg", ".webp"]
                    else "image/png"
                )

                full_prompt_text = time_lock_prefix + prompt_text

                recalled = recall_relevant_context(full_prompt_text, limit=3)
                recalled = limit_recall_by_tokens(recalled)

                if recalled:
                    memory_block = "\n\n".join(
                        f"[{m['level'].upper()}] (Score: {m['score']})\n{m['text']}"
                        for m in recalled
                    )
                    full_prompt_text = f"RETRIEVED CONTEXT:\n{memory_block}\n\n{full_prompt_text}"
                    yield _sse_chunk("status", f"🧠 Recalled {len(recalled)} relevant memories.")

                logging.info(f"Injecting multimodal image with prompt: {full_prompt_text[:100]} and path: {img_path}")

                messages_history.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{b64_data}"
                                },
                            },
                        ],
                    }
                )

                execution_logs.append(f"Injected base64 context stream: {img_path}")
                logging.info(f"Multimodal image injected natively into messages_history: {img_path}")
            except Exception as e:
                error_msg = f"[System Error] Failed to read file: {e}"
                logging.error(error_msg)
                yield _sse_chunk("error", error_msg)
                return
        else:
            logging.warning(f"Image syntax detected but no valid path found in input: {user_input[:100]}")
          
            recalled = recall_relevant_context(user_input, limit=3)
            if recalled:
                memory_block = "\n\n".join(
                    f"[{m['level'].upper()}] (Score: {m['score']})\n{m['text']}"
                    for m in recalled
                )
                user_content = f"RETRIEVED CONTEXT:\n{memory_block}\n\n{time_lock_prefix + user_input}"
            else:
                user_content = time_lock_prefix + user_input

            messages_history.append({"role": "user", "content": user_content})
    else:
        logging.info(f"User prompt received: {user_input[:100]} (Elapsed time since last prompt: {elapsed_minutes} minutes)")
        messages_history.append(
            {"role": "user", "content": time_lock_prefix + user_input}
        )

    yield _sse_chunk("status", f"Received: {user_input[:200]}")

    # --- ROUTING LOGIC (Option A: choose model once per message) ---
    fingerprint = compute_task_fingerprint(user_input)
    cached = get_cached_routing("default", fingerprint)
    if cached:
        routing_config = cached
    else:
        classification = classify_task(user_input)
        task_type = classification["task_type"]
        confidence = classification["confidence"]
        fingerprint = classification["fingerprint"]

        if confidence < 0.70 and task_type != "default":
            logging.info(
                f"Low confidence ({confidence}) for '{task_type}'. Escalating to heavy model."
            )
            routing_config = MODEL_ROUTER["reasoning"]
        else:
            routing_config = MODEL_ROUTER.get(task_type, MODEL_ROUTER["default"])

        set_cached_routing(task_type, fingerprint, routing_config)

    task_type = classification["task_type"] if not cached else "default"

    max_escalations = 1
    escalation_attempt = 0

    while True:
        if _check_abort():
            yield _sse_chunk("status", "Aborted by user.")
            yield _sse_chunk("done", {"logs": execution_logs})
            break

        # Dynamic client selection (if using different endpoints)
        router_client = (
            OpenAI(
                base_url=routing_config["endpoint"],
                api_key=API_KEY,
                timeout=10800.0,
            )
            if routing_config["endpoint"] != LLAMA_BASE_URL
            else client
        )

        try:
            response = router_client.chat.completions.create(
                model=routing_config["model"],
                messages=messages_history,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.0,
            )
        except Exception as ex:
            error_msg = f"[Runtime Error] API endpoint failure: {ex}"
            logging.error(error_msg)
            yield _sse_chunk("error", error_msg)
            break

        msg = response.choices[0].message
        logging.info(f"LLM response received. Message: {str(msg.content)[:100]}. Tool calls: {len(msg.tool_calls) if msg.tool_calls else 0}")

        # --- TOOL CALL LOOP ---
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                if _check_abort():
                    yield _sse_chunk("status", "Aborted by user.")
                    yield _sse_chunk("done", {"logs": execution_logs})
                    return

                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")

                log_stmt = f"[Tool Call] -> {name}({args})"
                print(log_stmt)
                execution_logs.append(log_stmt)

                logging.info(f"EXECUTING TOOL: {name} with arguments: {json.dumps(args)}")

                yield _sse_chunk("tool_call", {"name": name, "args": args})

                result = execute_tool_call(name, args)

                logging.info(f"TOOL RESULT: {name} -> {str(result)[:100]}")

                yield _sse_chunk("tool_result", {"name": name, "result": str(result)})

                messages_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": result,
                    }
                )
            # After tools, loop again with same routing_config/model
            continue

        # --- FINAL CONTENT + VALIDATION / ESCALATION ---
        content = msg.content or ""
        logging.info(
            f"Final response content received: {content} (length: {len(content)} characters)"
        )

        # validation = validate_output(task_type, content)
        # if validation["needs_escalation"] and escalation_attempt < max_escalations:
        #     logging.info(
        #         f"Validation failed ({validation['quality']}). Escalating to heavy model."
        #     )
        #     messages_history.append(
        #         {"role": "assistant", "content": content + "\n[FAST_MODEL_OUTPUT]"}
        #     )
        #     routing_config = MODEL_ROUTER["reasoning"]
        #     escalation_attempt += 1
        #     continue

        validation = validate_output(task_type, content)
        if validation["needs_escalation"] and escalation_attempt < max_escalations:
            logging.info(
                f"Validation failed ({validation['quality']}). Escalating to heavy model."
            )

            # Do NOT prefill assistant content when escalating to a thinking-enabled model.
            # Optionally, you can add a lightweight marker as a user/system message if you want the heavy model
            # to know that a fast model already tried:
            messages_history.append({
                "role": "user",
                "content": "[SYSTEM NOTE] A faster model produced an unsatisfactory answer. "
                        "Please re-answer the last user request with full reasoning and higher quality."
            })

            routing_config = MODEL_ROUTER["reasoning"]
            escalation_attempt += 1
            continue


        messages_history.append({"role": "assistant", "content": content})
        yield _sse_chunk("response", content)
        break

    end_time_str = datetime.now().strftime("%d.%m.%Y @ %H:%M:%S")
    logging.info(
        f"Assistant response completed at {end_time_str}."
    )

    yield _sse_chunk("done", {"logs": execution_logs})


@app.post("/api/chat")
async def api_chat(payload: ChatPayload):
    logging.info(f"Chat API called with message: {payload.message[:100]}")
    return StreamingResponse(
        chat_stream(payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


if __name__ == "__main__":
    print("Starting Local Agent Server Web Interface on http://127.0.0.1:8000")
    logging.info("\n\n\nStarting Local Agent Server Web Interface on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)