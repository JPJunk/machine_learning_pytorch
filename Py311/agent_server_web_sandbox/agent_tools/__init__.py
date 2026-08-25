# init.py - Initialize the agent tools package and import all tool functions.
from .web_tools import web_search, web_scrape
from .fs_tools import read_file, write_file, append_file, copy_file, move_file, delete_file, list_files
from .text_tools import json_load, json_dump, clean_text, split_lines
from .embed_tools import compare_similarity, index_file, index_folder, search_embeddings
from .lang_tools import detect_language, summarize, translate
from .extract_tools import extract_numbers, extract_urls, extract_dates
from .hash_tools import hash_file, hash_text
from .image_tools import analyze_image, generate_image
from .audio_tools import transcribe_audio, speak_text
from .audio_generation import generate_music # , generate_sound_effect
from .database_tools import db_init, db_insert, db_query, db_update, db_delete
from .symlink_tools import create_symlink
from .run_python import run_python

# Import new system and memory state tools
from .memory import (
    optimize_context_window, cleanup_memory_db,
    force_gc_and_free_memory)
from .system_tools import (
    get_memory_usage,  
    get_system_health, list_active_processes, clear_upload_cache
)
from .common import save_agent_state, load_agent_state


TOOLS = {
    "web_search": web_search,
    "web_scrape": web_scrape,
    "read_file": read_file,
    "write_file": write_file,
    "append_file": append_file,
    "copy_file": copy_file,
    "move_file": move_file,
    "delete_file": delete_file,
    "list_files": list_files,
    "json_load": json_load,
    "json_dump": json_dump,
    "clean_text": clean_text,
    "split_lines": split_lines,
    "compare_similarity": compare_similarity,
    "detect_language": detect_language,
    "summarize": summarize,
    "translate": translate,
    "extract_numbers": extract_numbers,
    "extract_urls": extract_urls,
    "extract_dates": extract_dates,
    "hash_file": hash_file,
    "hash_text": hash_text,
    "analyze_image": analyze_image,
    "generate_image": generate_image,
    "transcribe_audio": transcribe_audio,
    "speak_text": speak_text,
    "generate_music": generate_music,
    # "generate_sound_effect": generate_sound_effect,
    "run_python": run_python,
    "db_init": db_init,
    "db_insert": db_insert,
    "db_query": db_query,
    "db_update": db_update,
    "db_delete": db_delete,
    "create_symlink": create_symlink,
    "get_memory_usage": get_memory_usage,
    "optimize_context_window": optimize_context_window,
    "cleanup_memory_db": cleanup_memory_db,
    "get_system_health": get_system_health,
    "force_gc_and_free_memory": force_gc_and_free_memory,
    "list_active_processes": list_active_processes,
    "clear_upload_cache": clear_upload_cache,
    "save_agent_state": save_agent_state,
    "load_agent_state": load_agent_state,
    "index_file": index_file,
    "index_folder": index_folder,
    "search_embeddings": search_embeddings

}