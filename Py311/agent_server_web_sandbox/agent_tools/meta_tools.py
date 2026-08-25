# agent_tools/meta_tools.py - Autonomous Creative Game Evolution Engine with Robust Regex Parsing
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


import datetime
import re
from openai import OpenAI
from .common import sanitize_edge_metadata

def autonomous_code_evolution(path: str, goal: str = "Invent a creative computer game from scratch", max_iterations: int = 5) -> str:
    """Runs an isolated, state-cleared loop to autonomously invent, refactor, and evolve a game while maintaining a persistent timeline log."""
    logger.info(f"Starting autonomous code evolution for path: {path}")
    clean_path = sanitize_edge_metadata(path).strip('"').strip("'").replace("\\", "/")
    
    # KORJAUS: Määritellään sekä pelitiedoston kohdekansio että erillinen logikansio oikein
    target_dir = os.path.dirname(clean_path) or "C:/test"
    log_dir = "C:/AI_logs"
    log_path = os.path.join(log_dir, "Agent_game_creation_log.txt").replace("\\", "/")
    
    # Varmistetaan, että molemmat kansiot ovat fyysisesti olemassa levyllä
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    base_url = os.getenv("LLAMA__BASE_URL", "http://localhost:5001/v1")
    model_name = os.getenv("LLAMA__MODEL", "qwen3.6-35b-a3b-uncensored-genesis-v2-apex-mtp")
    client = OpenAI(base_url=base_url, api_key="not-needed", timeout=3600.0) # 15 min per sisäinen iteraatio
    
    # Alustetaan lokitiedosto isäntäkoneen ajalla
    init_time = datetime.datetime.now().strftime("%d.%m.%Y klo %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(f"\n==================================================================\n")
        log_f.write(f"🎮 NEW AUTONOMOUS GAME EVOLUTION SESSION STARTED: {init_time}\n")
        log_f.write(f"Target: {clean_path}\n==================================================================\n")

    engine_logs = [f"[Meta-Engine] Evolution engine activated. Target: {clean_path}"]
    
    for i in range(1, max_iterations + 1):
        logger.info(f"Running iteration {i}/{max_iterations}")
        current_time = datetime.datetime.now().strftime("%d.%m.%Y klo %H:%M:%S")
        engine_logs.append(f"[Meta-Engine] Running iteration {i}/{max_iterations}...")
        
        # 1. Luetaan vanha koodi jos olemassa
        existing_code = ""
        if os.path.exists(clean_path):
            logger.info(f"Reading existing code from {clean_path}")
            with open(clean_path, "r", encoding="utf-8") as f:
                existing_code = f.read()
        
        # 2. Pyydetään mallia suunnittelemaan, perustelemaan ja kirjoittamaan koodi
        clean_goal = sanitize_edge_metadata(goal)
        meta_prompt = f"""You are an elite, highly creative autonomous game designer and software engineer.
Target File Location: {clean_path}
Ecosystem: Windows 11 Pro, Python 3.11, Pygame.

Current Cycle: Iteration {i} of {max_iterations}.
Current Host Date & Time in Kuopio, Finland: {current_time}.

Your instructions:
1. If this is Iteration 1, INVENT a deeply unique, engaging, and atmospheric game concept from scratch. Do NOT output a basic template or empty loop. Impress the user with features!
2. If previous code exists below, analyze it deeply. Add rich graphical features, particle effects, sounds, math-based physics, advanced scoring mechanics, start screens, and dynamic difficulty scaling.
3. You MUST provide an engineering summary explaining WHAT you changed/added in this iteration and WHY, followed by the COMPLETE updated source code.
4. CRITICAL: Output the entire, fully functional Python file every time. Do not use placeholders or truncated code chunks.

Current codebase state:
```python
{existing_code if existing_code else "# File is empty. Invent the game concept and write the initial robust build now."}
```

Format your output with these two specific block tags so the parser can safely extract them:
[SUMMARY]
Write your explanation here (what you did and why).
[/SUMMARY]

[CODE]
```python
# Complete updated code goes here
```
[/CODE]
"""
        
        try:
            logger.info(f"Sending prompt to LLM {model_name} for iteration {i}")
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": meta_prompt}],
                temperature=0.4
            )
            
            raw_response = response.choices.message.content or ""
            
            # 3. KORJAUS: Erotetaan selitys (Summary) kestävällä Regex-haulla
            summary_part = "No standard summary block found."
            summary_match = re.search(r"\[SUMMARY\](.*?)\[/SUMMARY\]", raw_response, re.DOTALL | re.IGNORECASE)
            if summary_match:
                summary_part = summary_match.group(1).strip()
            else:
                # Varamekanismi jos Aggressive-malli hukkasi tagit mutta kirjoitti tekstiä ennen koodilohkoa
                pre_code = raw_response.split("```")[0].strip()
                if pre_code:
                    summary_part = pre_code

            # 4. KORJAUS: Kaapataan Python-koodi luotettavalla Regex-haulla
            new_code = ""
            code_match = re.search(r"```python(.*?)```", raw_response, re.DOTALL | re.IGNORECASE)
            if code_match:
                new_code = code_match.group(1).strip()
            else:
                # Toinen yritys ilman python-avainsanaa
                code_match_raw = re.search(r"\[CODE\](.*?)\[/CODE\]", raw_response, re.DOTALL | re.IGNORECASE)
                if code_match_raw:
                    new_code = code_match_raw.group(1).replace("```python", "").replace("```", "").strip()

            # 5. Kirjoitetaan koodi levylle vain jos se on validia
            if len(new_code) > 200 and "import pygame" in new_code.lower():
                logger.info(f"Writing iteration {i} to disk ({len(new_code)} chars)")
                with open(clean_path, "w", encoding="utf-8") as f:
                    f.write(new_code)
                engine_logs.append(f"[Meta-Engine] Success: Wrote iteration {i} to disk ({len(new_code)} chars).")
            else:
                logger.warning(f"Extracted code for iteration {i} was too short or missing Pygame. Skipping file write.")
                engine_logs.append(f"[Meta-Engine] Warning: Extracted code was too short or missing Pygame. Skipping file write to protect history.")
                if not new_code:
                    summary_part += "\n[ENGINE ERROR: Could not extract code block from raw response.]"
                else:
                    summary_part += f"\n[ENGINE WARNING: Extracted code was rejected (Length: {len(new_code)} chars).]"

            code_part = new_code if new_code else "[No valid code extracted in this iteration.]"

            # 6. Kirjoitetaan KESTÄVÄ LOKITIEDOSTO aikaleimalla (Append-tila)
            log_time = datetime.datetime.now().strftime("%d.%m.%Y @ %H:%M:%S")
            with open(log_path, "a", encoding="utf-8") as log_f:
                log_f.write(f"\n⚡ [ITERATION {i}/{max_iterations}] — TIMESTAMP: {log_time} in Kuopio, Finland\n")
                log_f.write(f"------------------------------------------------------------------\n")
                log_f.write(f"{summary_part}\n\n")
                log_f.write(f"{code_part}\n\n")
                log_f.write(f"{raw_response}\n\n")
                log_f.write(f"File state updated: {len(new_code)} characters processed.\n")
                log_f.write(f"------------------------------------------------------------------\n")
                
        except Exception as ex:
            logger.error(f"Critical error during iteration {i}: {ex}")
            engine_logs.append(f"[Meta-Engine] Critical error during iteration {i}: {ex}")
            break
            
    final_time = datetime.datetime.now().strftime("%d.%m.%Y klo %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(f"\n🏁 AUTONOMOUS EVOLUTION SESSION CONCLUDED: {final_time}\n")
        log_f.write(f"==================================================================\n")
        
    logger.info("Autonomous code evolution session concluded")
    return "\n".join(engine_logs) + f"\n\n[Meta-Engine Done] Creative process finished. Game saved at {clean_path}. Detailed timeline logs appended to {log_path}."
