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

import psutil

import gc
import time
from typing import Dict, Any, List, Optional


def get_memory_usage() -> Dict[str, Any]:
    """
    Get current memory usage of the system and Python process.
    
    Returns:
        dict: Memory statistics including total, used, available, and Python-specific metrics.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    system_mem = psutil.virtual_memory()
    
    return {
        "system": {
            "total_gb": round(system_mem.total / (1024**3), 2),
            "used_gb": round(system_mem.used / (1024**3), 2),
            "available_gb": round(system_mem.available / (1024**3), 2),
            "usage_percent": system_mem.percent,
        },
        "python_process": {
            "rss_mb": round(mem_info.rss / (1024**2), 2),
            "vms_mb": round(mem_info.vms / (1024**2), 2),
            "percent": process.memory_percent(),
        },
        "gc_stats": {
            "collections": gc.get_count(),
            "thresholds": gc.get_threshold(),
        }
    }

def get_system_health() -> Dict[str, Any]:
    """
    Get comprehensive system health metrics including CPU, memory, disk, and LLM status.
    
    Returns:
        dict: Health metrics for all monitored components.
    """
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    
    # Check if LLM server is responding
    import urllib.request
    llm_status = "unknown"
    try:
        req = urllib.request.Request("http://127.0.0.1:5001/health")
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                llm_status = "healthy"
            else:
                llm_status = "unhealthy"
    except Exception:
        llm_status = "unreachable"
    
    return {
        "cpu": {
            "percent": cpu_percent,
            "cores": psutil.cpu_count(),
            "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0
        },
        "memory": {
            "total_gb": round(memory.total / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "percent": memory.percent
        },
        "disk": {
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "percent": disk.percent
        },
        "llm_server": {
            "status": llm_status,
            "url": "http://127.0.0.1:5001"
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

def list_active_processes() -> List[Dict[str, Any]]:
    """
    List all active processes with memory and CPU usage.
    
    Returns:
        list: List of process dictionaries sorted by memory usage.
    """
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
        try:
            pinfo = proc.info
            if pinfo['memory_info']:
                processes.append({
                    "pid": pinfo['pid'],
                    "name": pinfo['name'],
                    "cpu_percent": pinfo['cpu_percent'] or 0,
                    "memory_mb": round(pinfo['memory_info'].rss / (1024**2), 2)
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    # Sort by memory usage descending
    processes.sort(key=lambda x: x['memory_mb'], reverse=True)
    return processes[:20]  # Top 20 most memory-intensive

def clear_upload_cache() -> Dict[str, Any]:
    """
    Clear the uploads directory of temporary files older than 1 hour.
    
    Returns:
        dict: Cleanup results including files removed and space freed.
    """
    upload_dir = os.path.abspath("uploads")
    if not os.path.exists(upload_dir):
        return {"status": "no_upload_dir", "message": "Uploads directory does not exist"}
    
    now = time.time()
    one_hour = 3600
    files_removed = 0
    space_freed = 0
    
    for filename in os.listdir(upload_dir):
        filepath = os.path.join(upload_dir, filename)
        if os.path.isfile(filepath):
            file_age = now - os.path.getmtime(filepath)
            if file_age > one_hour:
                size = os.path.getsize(filepath)
                os.remove(filepath)
                files_removed += 1
                space_freed += size
    
    return {
        "status": "cache_cleared",
        "files_removed": files_removed,
        "space_freed_mb": round(space_freed / (1024**2), 2),
        "upload_dir": upload_dir
    }

def self_diagnose() -> dict:
    """
    Provides a snapshot of the system's health, focusing on RAM and CPU usage.
    Crucial for a CPU-only LLM setup to prevent swapping.
    """
    try:
        # Memory Info
        vm = psutil.virtual_memory()
        mem_info = {
            "total_gb": round(vm.total / (1024**3), 2),
            "available_gb": round(vm.available / (1024**3), 2),
            "used_percent": vm.percent,
            "swap_total_gb": round(psutil.swap_memory().total / (1024**3), 2),
            "swap_used_gb": round(psutil.swap_memory().used / (1024**3), 2)
        }

        # CPU Info
        cpu_info = {
            "percent_per_core": psutil.cpu_percent(interval=1, percpu=True),
            "load_avg_1m": os.getloadavg()[0] if hasattr(os, 'getloadavg') else None
        }

        # DB Size
        db_path = os.path.abspath("agent_memory.db")
        db_size_mb = 0
        if os.path.exists(db_path):
            db_size_mb = os.path.getsize(db_path) / (1024**2)

        return {
            "status": "healthy" if mem_info["used_percent"] < 90 else "warning",
            "memory": mem_info,
            "cpu": cpu_info,
            "db_size_mb": round(db_size_mb, 2)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
