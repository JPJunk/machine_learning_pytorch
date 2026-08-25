# \agent_tools\memory.py
import os
import logging
import asyncio
from typing import Any, Dict, List

# Configure logging to write to app.log in the project root directory
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_FILE = os.path.join(_BASE_DIR, "app.log")

logging.basicConfig(
    filename=_LOG_FILE,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

import datetime, gc, json, os, psutil, sqlite3
import numpy as np

from typing import Dict, Any # , List, Optional

from .common import LLAMA_MODEL, EMBED_MODEL_NAME, client
from .common import messages_history, SYSTEM_PROMPT
from .common import memory_conn, sanitize_edge_metadata, estimate_tokens_messages, load_system_prompt, memory_get_for_prompt

# Global locks to avoid race conditions
context_lock = asyncio.Lock()
memory_lock = asyncio.Lock()
vector_memory_lock = asyncio.Lock()

# ---------------- MEMORY ----------------
def db_prune(max_rows: int = 100) -> dict:
    """
    Prunes old short-term memories if the database grows too large.
    Keeps only the most recent 'max_rows' entries in agent_memory.
    """
    try:
        with memory_conn() as conn:
            # Get all IDs
            rows = conn.execute("SELECT id FROM agent_memory ORDER BY created_at DESC").fetchall()
            
            if len(rows) <= max_rows:
                return {"status": "ok", "message": f"Memory size ({len(rows)}) is within limit."}
            
            ids_to_delete = [str(r[0]) for r in rows[max_rows:]]
            
            if ids_to_delete:
                conn.execute(
                    f"DELETE FROM agent_memory WHERE id IN ({','.join('?'*len(ids_to_delete))})",
                    ids_to_delete
                )
                return {"status": "ok", "pruned": len(ids_to_delete), "remaining": max_rows}
        
        return {"status": "ok", "message": "No pruning needed."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def memory_store(level: str, summary: str):
    with memory_conn() as conn:
        conn.execute(
            "INSERT INTO agent_memory (created_at, level, summary) VALUES (?, ?, ?)",
            (datetime.datetime.utcnow().isoformat(), level, summary),
        )
    logger.info(f"Inserted {level} level agent_memory ({summary}[:100]) to DB")

def llm_memory(prompt: str) -> str:
    """
    Use the SAME LLM as the chat system to generate memory summaries.
    No tools, no history, no system prompt — pure summarisation.
    """
    try:
        logger.info(f"Summarising: ({prompt}[:100])")
        resp = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You are the memory engine for this agent."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            tools=None,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"[Memory LLM Error] {e}")
        return ""

def force_gc_and_free_memory() -> Dict[str, Any]:
    """
    Force garbage collection and attempt to free Python memory.
    """
    before = psutil.Process(os.getpid()).memory_info().rss

    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    gc.freeze()

    after = psutil.Process(os.getpid()).memory_info().rss

    return {
        "status": "gc_complete",
        "memory_before_mb": round(before / (1024**2), 2),
        "memory_after_mb": round(after / (1024**2), 2),
        "freed_mb": round((before - after) / (1024**2), 2),
        "gc_counts": gc.get_count(),
    }

async def optimize_context_window(max_tokens: int = 32768) -> Dict[str, Any]:
    """
    Trim the conversation history to fit within a specified token limit.
    Preserves system prompt and most recent, highest-value messages.
    """
    async with context_lock:
        current_tokens = estimate_tokens_messages(messages_history)

        if current_tokens <= max_tokens:
            return {
                "status": "no_trim_needed",
                "current_tokens": current_tokens,
                "max_tokens": max_tokens,
                "tokens_removed": 0,
            }

        logger.warning(f"[CONTEXT] Trimming context: {current_tokens} → {max_tokens} tokens.")

        # Separate system and non-system messages
        system_msgs = [m for m in messages_history if m.get("role") == "system"]
        other_msgs = [m for m in messages_history if m.get("role") != "system"]

        trimmed: List[Dict[str, Any]] = []
        tokens_used = 0

        # Always keep all system messages (usually just one)
        for msg in system_msgs:
            trimmed.append(msg)
            tokens_used += estimate_tokens_messages([msg])

        # Sort other messages by recency (newest first)
        other_msgs = list(reversed(other_msgs))

        # Prefer non-tool messages; tool messages are often huge
        def msg_priority(m: Dict[str, Any]) -> int:
            role = m.get("role")
            if role == "user":
                return 0
            if role == "assistant":
                return 1
            if role == "tool":
                return 2
            return 3

        other_msgs.sort(key=msg_priority)

        for msg in other_msgs:
            # Skip multimodal content (lists with images)
            if isinstance(msg.get("content"), list):
                continue

            msg_tokens = estimate_tokens_messages([msg])
            if tokens_used + msg_tokens <= max_tokens:
                trimmed.append(msg)
                tokens_used += msg_tokens
            else:
                break

        messages_history.clear()
        messages_history.extend(trimmed)

        new_tokens = estimate_tokens_messages(messages_history)
        return {
            "status": "trimmed",
            "original_tokens": current_tokens,
            "new_tokens": new_tokens,
            "tokens_removed": current_tokens - new_tokens,
            "max_tokens_enforced": max_tokens,
        }
    
def cleanup_memory_db() -> Dict[str, Any]:
    """
    Clean up the agent memory database by removing old short-term memories
    and compacting the database file using VACUUM.
    """
    with memory_conn() as conn:
        total_records = conn.execute("SELECT COUNT(*) FROM agent_memory").fetchone()[0]
        vector_records = conn.execute("SELECT COUNT(*) FROM vector_memories").fetchone()[0]

        one_hour_ago = (datetime.datetime.utcnow() - datetime.timedelta(hours=1)).isoformat()
        deleted_short = conn.execute(
            "DELETE FROM agent_memory WHERE level = 'short' AND created_at < ?",
            (one_hour_ago,),
        ).rowcount

        conn.execute("VACUUM")

    db_path = os.path.abspath("agent_memory.db")
    new_size = os.path.getsize(db_path) / (1024**2)

    logger.info(
        f"[DB] Cleanup complete. Records: {total_records} → short_deleted={deleted_short}, "
        f"vector={vector_records}, size={round(new_size, 2)} MB"
    )

    return {
        "status": "cleanup_complete",
        "total_records_before": total_records,
        "vector_records_before": vector_records,
        "short_term_deleted": deleted_short,
        "new_db_size_mb": round(new_size, 2),
        "compact_successful": True,
    }

# ---------------- MEMORY CONSOLIDATION ----------------
async def memory_consolidate_short_to_mid() -> Dict[str, Any]:
    async with memory_lock:
        with memory_conn() as conn:
            rows = conn.execute("""
                SELECT id, summary
                FROM agent_memory
                WHERE level = 'short'
                ORDER BY created_at DESC
            """).fetchall()

        total_tokens = sum(len(r[1].split()) for r in rows)

        if len(rows) < 5 or total_tokens < 2500:
            return {"status": "skipped", "reason": "not enough short-term memory"}

        rows = rows[:10]
        joined = "\n\n---\n\n".join(r[1] for r in rows)

        prompt = f"""
Merge the following SHORT-TERM memories into a single MID-TERM memory.
Focus on code-relevant and agent-relevant information.
Keep only stable, reusable information. Remove noise.

SHORT-TERM MEMORIES:
{joined}
"""
        mid_summary = llm_memory(prompt)
        if not mid_summary.strip():
            logger.warning("[MEMORY] Short→Mid consolidation produced empty summary. Skipping.")
            return {"status": "skipped"}

        memory_store("mid", mid_summary)
        await store_vector_memory(mid_summary, "mid")

        ids = [str(r[0]) for r in rows]
        with memory_conn() as conn:
            conn.execute(
                f"DELETE FROM agent_memory WHERE id IN ({','.join('?'*len(ids))})",
                ids,
            )

        logger.info(f"[MEMORY] Consolidated {len(rows)} memories short → mid")
        return {"status": "ok", "merged": len(ids)}

async def memory_consolidate_mid_to_long() -> Dict[str, Any]:
    async with memory_lock:
        with memory_conn() as conn:
            rows = conn.execute("""
                SELECT id, summary
                FROM agent_memory
                WHERE level = 'mid'
                ORDER BY created_at DESC
            """).fetchall()

        total_tokens = sum(len(r[1].split()) for r in rows)

        if len(rows) < 5 or total_tokens < 5000:
            return {"status": "skipped", "reason": "not enough mid-term memory"}

        rows = rows[:10]
        joined = "\n\n---\n\n".join(r[1] for r in rows)

        prompt = f"""
Merge the following MID-TERM memories into a single LONG-TERM memory.
Keep only the most stable, identity-level, long-lasting information.

MID-TERM MEMORIES:
{joined}
"""
        long_summary = llm_memory(prompt)
        if not long_summary.strip():
            logger.warning("[MEMORY] Mid→Long consolidation produced empty summary. Skipping.")
            return {"status": "skipped"}

        memory_store("long", long_summary)
        await store_vector_memory(long_summary, "long")

        ids = [str(r[0]) for r in rows]
        with memory_conn() as conn:
            conn.execute(
                f"DELETE FROM agent_memory WHERE id IN ({','.join('?'*len(ids))})",
                ids,
            )

        logger.info(f"[MEMORY] Consolidated {len(rows)} memories mid → long")
        return {"status": "ok", "merged": len(ids)}


# --- AUTOMATIC MEMORY MONITORING ---
async def check_and_cleanup_memory() -> bool:
    """
    Monitors system RAM usage. If > 90%, triggers cleanup routines to free memory.
    """
    mem = psutil.virtual_memory()
    if mem.percent > 90:
        logger.warning(f"⚠️ HIGH MEMORY USAGE ({mem.percent}%): Triggering cleanup...")

        # 1. Trim context window
        await optimize_context_window(max_tokens=32768)

        # 2. Force GC
        force_gc_and_free_memory()

        # 3. Cleanup memory DB
        cleanup_memory_db()

        return True

    return False

async def auto_memorise_and_reset() -> None:
    """
    Summarise current messages_history into short-term memory,
    consolidate to mid/long, rebuild SYSTEM_PROMPT, and reset context.
    """
    async with memory_lock:
        global SYSTEM_PROMPT, messages_history

        # 1. Flatten conversation into a single text block
        parts: List[str] = []
        for m in messages_history:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            if isinstance(content, list):
                # multimodal: keep only text parts
                content = "\n".join(
                    c["text"]
                    for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            parts.append(f"[{role.upper()}]\n{content}")

        full_context = "\n\n".join(parts)
        logger.info(f"[MEMORY] Summarizing context (first 200 chars): {full_context[:200]}")

        prompt = f"""
Summarise the important, reusable information from the following context.
Focus on code-relevant and agent-relevant information.
Keep only stable, reusable information. Remove noise.

CONTEXT:
{full_context}
"""
        distilled = llm_memory(prompt)
        if not distilled.strip():
            logger.warning("[MEMORY] Distillation returned empty summary. Skipping store.")
        else:
            logger.info(f"[MEMORY] Distilled summary (first 200 chars): {distilled[:200]}")
            memory_store("short", distilled)
            await memory_consolidate_short_to_mid()
            await memory_consolidate_mid_to_long()

        # 3. Rebuild SYSTEM_PROMPT with long-term memory
        SYSTEM_PROMPT = load_system_prompt() + "\n\n"
        mem_rows = memory_get_for_prompt()
        if mem_rows:
            mem_text = "\n\n".join(f"[{lvl.upper()}]\n{summary}" for lvl, summary in mem_rows)
            SYSTEM_PROMPT += mem_text

        logger.info(f"[MEMORY] SYSTEM_PROMPT rebuilt. Length: {len(SYSTEM_PROMPT)}")

        # 4. Reset context to just system prompt
        messages_history.clear()
        messages_history.append({"role": "system", "content": SYSTEM_PROMPT})


# ---------------- VECTOR DB ----------------
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

def is_duplicate(new_vec, threshold=0.92):
    with memory_conn() as conn:
        rows = conn.execute("SELECT embedding FROM vector_memories").fetchall()

    for (emb_json,) in rows:
        try:
            existing_vec = json.loads(emb_json)
            score = _compute_cosine(new_vec, existing_vec)
            if score > threshold:
                return True
        except:
            continue

    return False

def get_all_vector_embeddings():
    with memory_conn() as conn:
        rows = conn.execute("SELECT embedding FROM vector_memories").fetchall()
    vectors = []
    for (emb_json,) in rows:
        try:
            vectors.append(json.loads(emb_json))
        except:
            pass
    return vectors

async def store_vector_memory(text: str, level: str) -> bool:
    """
    Generates an embedding for the given text and stores it in vector_memories.
    Returns True if successful, False otherwise.
    """
    async with vector_memory_lock:
        try:
            # Truncate text to avoid DB explosion
            if len(text) > 5000:
                text = text[:5000]

            vector = _get_raw_embedding(text)
            if not vector:
                logger.warning(f"[VECTOR] Embedding failed for text length {len(text)}. Not storing.")
                return False

            if is_duplicate(vector, threshold=0.92):
                logger.info("[VECTOR] Skipping duplicate vector memory.")
                return False

            embedding_json = json.dumps(vector, separators=(",", ":"))

            with memory_conn() as conn:
                conn.execute(
                    "INSERT INTO vector_memories (created_at, level, text, embedding) VALUES (?, ?, ?, ?)",
                    (datetime.datetime.utcnow().isoformat(), level, text, embedding_json),
                )

            logger.info(f"[VECTOR] Stored vector memory for {len(text)} chars (Level: {level})")
            return True

        except Exception as e:
            logger.error(f"[VECTOR] Error storing vector memory: {e}", exc_info=True)
            return False

def recall_relevant_context(user_input: str, limit: int = 3, threshold: float = 0.65) -> list[dict]:
    """
    Retrieves the most semantically relevant memories for a given user input.
    Returns a list of dicts: [{'score': float, 'level': str, 'text': str}, ...]
    """
    # 1. Embed the current user input
    query_vec = _get_raw_embedding(user_input)
    if not query_vec:
        logger.warning("Failed to embed user input for recall.")
        return []

    # 2. Fetch all vector memories from the DB
    with memory_conn() as conn:
        rows = conn.execute(
            "SELECT id, level, text, embedding FROM vector_memories"
        ).fetchall()

    if not rows:
        return []

    # 3. Compute cosine similarity and filter by threshold
    results = []
    for row in rows:
        _, level, text, emb_json = row
        try:
            stored_vec = json.loads(emb_json)
            # score = _compute_cosine(query_vec, stored_vec)
            WEIGHTS = {"long": 1.3, "mid": 1.1, "short": 1.0}

            score = _compute_cosine(query_vec, stored_vec) * WEIGHTS[level]

            if score >= threshold:
                results.append({
                    "score": round(score, 4),
                    "level": level,
                    "text": text
                })
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Invalid embedding JSON for row {row[0]}")

    # 4. Sort by score descending and return top N
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]
