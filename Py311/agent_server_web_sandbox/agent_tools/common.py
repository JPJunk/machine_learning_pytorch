import logging
import os

# Configure logging to write to app.log in the project root directory
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_FILE = os.path.join(_BASE_DIR, "app.log")

logging.basicConfig(
    filename=_LOG_FILE,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

import copy, datetime, json, re, sqlite3, textwrap

from openai import OpenAI

EDGE_PLACEHOLDER_PATTERN = re.compile(
    r"<WebsiteContent_[^>]+></WebsiteContent_[^>]+>"
)

def sanitize_edge_metadata(text: str) -> str:
    """Remove Edge's WebsiteContent_XXXX placeholder tags from any input."""
    if not isinstance(text, str):
        return text
    return EDGE_PLACEHOLDER_PATTERN.sub("", text).strip()

# ---------------- CONFIG ----------------
LLAMA_BASE_URL = os.getenv("LLAMA__BASE_URL", "http://localhost:5001/v1")
LLAMA_MODEL    = os.getenv("LLAMA__MODEL", "qwen3.6-35b-a3b-uncensored-genesis-v2-apex-mtp")  
API_KEY     = os.getenv("LLAMA__API_KEY", "not-needed")

client = OpenAI(base_url=LLAMA_BASE_URL, api_key=API_KEY, timeout=10800.0)
EMBED_MODEL_NAME = os.getenv("LLAMA_EMBED_MODEL", "bge-m3")


# ----------------  TIME CONTEXT ----------------
now_time = datetime.datetime.now()
current_time_str = now_time.strftime("%A, %B %d, %Y (Aika: %H:%M:%S)")

time_lock_prefix = (
    f"[TIME CONTEXT: The current host system date and time is strictly {current_time_str[:-3]} local time in Kuopio, Finland. "
    f"Treat this as the absolute final local time. Do NOT convert time zones."
    f"It has been exactly [elapsed_minutes] minutes since the user last interacted with you. "
    f"Use your internal reasoning guidelines to understand how long the gap was in human terms, and adapt your awareness accordingly.]\n"
)


# ---------------- TOKEN ESTIMATION ----------------
MAX_RECALL_TOKENS = 4000

def estimate_tokens(text: str) -> int:
    tokens = int(len(text.split()) / 0.75)
    logger.info(f"Current tokens count = {tokens}") 
    return tokens

def limit_recall_by_tokens(recalled):
    total = 0
    kept = []
    for m in recalled:
        t = estimate_tokens(m["text"])
        if total + t > MAX_RECALL_TOKENS:
            break
        kept.append(m)
        total += t
    return kept

def estimate_tokens_messages(messages):
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        total += estimate_tokens(str(content))
    return total


# ---------------- MEMORY DB SETUP ----------------
MEMORY_DB = os.path.abspath("agent_memory.db")

def memory_conn():
    conn = sqlite3.connect(MEMORY_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            level TEXT NOT NULL,
            summary TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vector_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            level TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    """)
    return conn
   
def memory_get_for_prompt():
    with memory_conn() as conn:
        rows = conn.execute("""
            SELECT level, summary
            FROM agent_memory
            ORDER BY
                CASE level
                    WHEN 'long' THEN 1
                    WHEN 'mid'  THEN 2
                    WHEN 'short' THEN 3
                END,
                created_at DESC
            LIMIT 6
        """).fetchall()
        logger.info(f"Fetched {len(rows)} memories")        
    return rows

# ---------------- AGENT STATE ----------------
def save_agent_state(path: str = "agent_state.json") -> str:
    try: 
        safe_messages = copy.deepcopy(messages_history)
        state = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "system_prompt": SYSTEM_PROMPT,
            "messages_history": safe_messages,
            "version": "1.0"
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        return f"✅ Agent state saved to {path} ({len(safe_messages)} messages)"
    except Exception as e:
        return f"❌ Error saving state: {str(e)}"

def load_agent_state(path: str = "agent_state.json") -> str:
    if not os.path.exists(path):
        return f"❌ State file not found: {path}"
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)  # FIXED: was json.load(state)
        
        global SYSTEM_PROMPT, messages_history
        
        if state.get("version") != "1.0":
            return "⚠️ Incompatible state file version"
            
        SYSTEM_PROMPT = state.get("system_prompt", "")
        messages_history = state.get("messages_history", [])
        
        return f"✅ Agent state loaded from {path}. Context restored ({len(messages_history)} messages)"
    except Exception as e:
        return f"❌ Error loading state: {str(e)}"


# ---------------- INITIALIZE ----------------
def load_system_prompt():
    path = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            logger.info(f"Loaded system prompt from {path} ({len(content)} characters)")
            return content
    except Exception:
        return ""

SYSTEM_PROMPT = load_system_prompt() + "\n\n" + time_lock_prefix + "\n\n"
logger.info(f"\n{SYSTEM_PROMPT}\n")
messages_history = [{"role": "system", "content": textwrap.dedent(SYSTEM_PROMPT).strip()}]