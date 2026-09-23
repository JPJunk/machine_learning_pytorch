# agent_tools/symlink_tools.py
import os
import subprocess
from .common import logger  # Use the shared logger from common.py

def create_symlink(target: str, link_path: str, is_directory: bool = False) -> str:
    """
    Create a symbolic link on Windows 11.
    Works for both files and directories.
    Automatically removes existing broken links.
    """

    target = target.strip()
    link_path = link_path.strip()

    if not target:
        return "[ERROR] Missing target path."

    if not link_path:
        return "[ERROR] Missing link path."

    # If link exists, remove it
    if os.path.exists(link_path) or os.path.islink(link_path):
        try:
            os.remove(link_path)
        except Exception:
            pass

    # Ensure parent folder exists
    parent = os.path.dirname(link_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    # Windows mklink command
    cmd = ["cmd", "/c", "mklink"]
    if is_directory:
        cmd.append("/D")

    cmd.append(f'"{link_path}"')
    cmd.append(f'"{target}"')

    try:
        result = subprocess.check_output(" ".join(cmd), shell=True, stderr=subprocess.STDOUT)
        return result.decode("utf-8", errors="ignore")
    except subprocess.CalledProcessError as e:
        return f"[ERROR] mklink failed: {e.output.decode('utf-8', errors='ignore')}"
