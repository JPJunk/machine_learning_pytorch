# agent_tools/run_python.py
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

import subprocess
import sys
from typing import Dict, Any, List, Optional

def run_python(
    path: str,
    args: Optional[List[str]] = None,
    timeout: int = 30,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a Python script as a subprocess and capture stdout, stderr, and exit code.

    Parameters
    ----------
    path : str
        Path to the Python script to execute.
    args : list[str], optional
        Extra command-line arguments to pass to the script.
    timeout : int
        Max seconds to allow the script to run.
    cwd : str, optional
        Working directory for the process. If None, uses script's directory.

    Returns
    -------
    dict
        {
          "stdout": str,
          "stderr": str,
          "exit_code": int,
          "command": list[str]
        }
    """
    logger.info(f"Executing Python script: {path} with timeout {timeout}s")
    if not isinstance(path, str) or not os.path.isfile(path):
        logger.warning(f"File not found or invalid path: {path}")
        return {
            "stdout": "",
            "stderr": f"[run_python] File not found: {path}",
            "exit_code": -1,
            "command": []
        }

    if cwd is None:
        cwd = os.path.dirname(os.path.abspath(path)) or None

    cmd = [sys.executable, path]
    if args and isinstance(args, list):
        cmd.extend(args)

    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
        )
        out, err = proc.communicate(timeout=timeout)
        logger.info(f"Script completed with exit code {proc.returncode}")
        return {
            "stdout": out,
            "stderr": err,
            "exit_code": proc.returncode,
            "command": cmd,
        }
    except subprocess.TimeoutExpired:
        logger.warning(f"Script timed out after {timeout}s: {' '.join(cmd)}")
        proc.kill()
        try:
            out, err = proc.communicate()
        except Exception:
            out, err = "", ""
        return {
            "stdout": out,
            "stderr": (err or "") + "\n[run_python] TimeoutExpired",
            "exit_code": -2,
            "command": cmd,
        }
    except Exception as e:
        logger.error(f"Exception during script execution: {e}")
        return {
            "stdout": "",
            "stderr": f"[run_python] Exception: {e}",
            "exit_code": -3,
            "command": cmd,
        }
