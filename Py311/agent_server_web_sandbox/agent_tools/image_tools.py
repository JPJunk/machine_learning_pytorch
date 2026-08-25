# agent_tools/image_tools.py - Image analysis and generation tools using Qwen-VL and ComfyUI.
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

import base64
import json
import time
from openai import OpenAI
import requests
from .common import sanitize_edge_metadata

LLAMA_IMAGE_BASE_URL = os.getenv("LLAMA_IMAGE_BASE_URL", "http://localhost:5001/v1")
LLAMA_MODEL = os.getenv("LLAMA__MODEL", "qwen3.6-35b-a3b-uncensored-genesis-v2-apex-mtp")
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")

client = OpenAI(base_url=LLAMA_IMAGE_BASE_URL, api_key=os.getenv("LLAMA_IMAGE_API_KEY", "not-needed"))

def clean_prompt(text: str) -> str:
    """Strip common markdown/formatting characters from prompt text."""
    if not isinstance(text, str):
        logger.warning("Non-string input received for clean_prompt")
        return ""
    for char in ["*", "#", "_", "\n", "\r", "`"]:
        text = text.replace(char, " ")
    result = " ".join(text.split()).strip()
    logger.debug(f"Cleaned prompt length: {len(result)}")
    return result

def analyze_image(path: str) -> dict:
    """Reads a local file path, encodes it to base64, and prompts Qwen2.5-VL for details."""
    clean_path = sanitize_edge_metadata(path).strip('"').strip("'").replace("\\", "/")
    
    if not isinstance(clean_path, str) or not clean_path.strip():
        logger.warning("Invalid file path provided to analyze_image")
        return {"error": "Invalid file path provided."}

    if not os.path.exists(clean_path):
        logger.warning(f"File not found at path: {clean_path}")
        return {"error": f"File not found at path: {clean_path}"}
        
    try:
        logger.info(f"Reading image file for analysis: {clean_path}")
        with open(clean_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        
        ext = os.path.splitext(clean_path)[1].lower().replace(".", "")
        mime_type = f"image/{ext}" if ext in ["png", "jpg", "jpeg", "webp"] else "image/png"

        IMAGE_ANALYSIS_PROMPT = """
        Analyze this image with maximum depth and precision. Produce a single continuous text description that can be used directly as a generative image prompt for ComfyUI, Juggernaut, or Flux. Do not use sections, lists, headings, or formatting. Write one long, richly detailed paragraph.

        Describe everything visible in the image with vivid, expressive language: the overall scene, subjects, objects, environment, background, foreground, midground, lighting, shadows, reflections, textures, materials, colors, shapes, composition, perspective, depth, atmosphere, mood, and any fine details. Include information about camera angle, focal length feel, depth of field, sharpness, bokeh, exposure, and photographic qualities. Describe the artistic style, aesthetic, and any cinematic or painterly qualities. Mention the emotional tone and visual storytelling elements. Include descriptive adjectives, sensory details, and nuanced observations. Avoid guessing unknown facts; describe only what is visible. The final output must be a single, flowing, highly descriptive prompt suitable for image generation.
        """

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": IMAGE_ANALYSIS_PROMPT
                    }
                ]
            }
        ]

        logger.info(f"Sending image to vision model {LLAMA_MODEL} for analysis")
        resp = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=messages,
            temperature=0.1,
        )
        result = {"analysis": resp.choices[0].message.content.strip()}
        logger.info("Image analysis completed successfully")
        return result
    except Exception as e:
        logger.error(f"analyze_image failed raw trace output: {e}")
        return {"error": f"Failed to execute image analysis: {str(e)}"}

def generate_image(prompt: str) -> dict:
    """Sends a clean text prompt to your local ComfyUI layout engine safely."""
    logger.info(f"Generating image with prompt length: {len(prompt)}")
    if not isinstance(prompt, str):
        logger.warning("Non-string input received for generate_image")
        return {"error": "Invalid prompt type provided."}
        
    clean_prompt_text = sanitize_edge_metadata(prompt)
    prompt_text = clean_prompt(clean_prompt_text)
    
    if not prompt_text:
        logger.warning("Empty prompt after cleaning in generate_image")
        return {"error": "Empty prompt after cleaning."}

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workflow_path = os.path.join(script_dir, "stable_cascade_stage_c.json")
    
    if not os.path.exists(workflow_path):
        logger.warning(f"ComfyUI workflow layout file missing at: {workflow_path}")
        return {"error": f"ComfyUI workflow layout file missing at: {workflow_path}"}

    try:
        logger.info(f"Loading ComfyUI workflow from: {workflow_path}")
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed parsing workflow JSON configuration: {e}")
        return {"error": f"Failed parsing workflow JSON configuration: {str(e)}"}
    except Exception as e:
        logger.error(f"Failed reading workflow configuration layout: {e}")
        return {"error": f"Failed reading workflow configuration layout: {str(e)}"}

    # FIX: Convert all node IDs to integers. ComfyUI's validation loop expects integer keys
    # when resolving array references like [4, 0], but JSON loaders produce string keys ("4").
    workflow_data = {int(k): v for k, v in workflow_data.items()}

    text_node_found = False
    for node_id, node_config in workflow_data.items():
        if "inputs" in node_config and "text" in node_config["inputs"]:
            input_str = str(node_config.get("inputs", {})).lower()
            is_negative = "negative" in input_str or node_config.get("_meta", {}).get("title", "").lower().find("negative") != -1
            
            if not is_negative and node_config["inputs"]["text"] == "":
                workflow_data[node_id]["inputs"]["text"] = prompt_text
                text_node_found = True
            elif not text_node_found: # Fallback if no empty text node is specifically filtered
                workflow_data[node_id]["inputs"]["text"] = prompt_text
                text_node_found = True

    try:
        logger.info(f"Sending prompt to ComfyUI at {COMFYUI_URL}/prompt")
        resp = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow_data}, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"ComfyUI rejected prompt execution payload: {resp.text}")
            return {"error": f"ComfyUI rejected prompt execution payload: {resp.text}"}
        
        prompt_id = resp.json().get("prompt_id")
        output_file = None
        
        logger.info(f"Waiting for ComfyUI generation (tracking prompt_id: {prompt_id})")
        for _ in range(600):
            history_resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=5)
            if history_resp.status_code == 200 and prompt_id in history_resp.json():
                history_data = history_resp.json()[prompt_id]
                if "outputs" in history_data:
                    for n_id, n_out in history_data["outputs"].items():
                        if "images" in n_out and len(n_out["images"]) > 0:
                            output_file = n_out["images"][0]["filename"]
                            break
                if output_file:
                    break
            time.sleep(10.0)

        if not output_file:
            logger.warning("Generation completed or timed out but image output name tracking failed.")
            return {"error": "Generation completed or timed out but image output name tracking failed."}

        full_path = os.path.normpath(os.path.join("C:/ComfyUI-master/output", output_file))
        logger.info(f"Image generation successful, output saved to: {full_path}")
        return {"output_path": full_path}
    except requests.exceptions.Timeout:
        logger.error("Network timeout during ComfyUI orchestration workflow.")
        return {"error": "Network timeout during ComfyUI orchestration workflow."}
    except Exception as e:
        logger.error(f"Network exception during ComfyUI orchestration workflow: {e}")
        return {"error": f"Network exception during ComfyUI orchestration workflow: {str(e)}"}

def describe_and_recreate(path: str) -> dict:
    """Analyzes an image and recreates it with a refined prompt."""
    logger.info(f"Starting describe_and_recreate for path: {path}")
    if not isinstance(path, str):
        logger.warning("Non-string input received for describe_and_recreate")
        return {"error": "Invalid path type provided."}
        
    analysis_res = analyze_image(path)
    if "error" in analysis_res:
        logger.warning(f"Analysis failed during describe_and_recreate: {analysis_res.get('error')}")
        return analysis_res
        
    analysis = analysis_res["analysis"]
    try:
        logger.info("Sending analysis to LLM for prompt refinement")
        prompt_resp = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {"role": "system", "content": "Convert image analysis text into a clean Stable Cascade tag prompt."},
                {"role": "user", "content": sanitize_edge_metadata(analysis)}
            ],
            temperature=0.2,
        )
        refined_prompt = clean_prompt(prompt_resp.choices[0].message.content)
        logger.info("Starting image generation with refined prompt")
        generation_res = generate_image(refined_prompt)
        return {"analysis": analysis, "prompt": refined_prompt, "generation": generation_res}
    except Exception as e:
        logger.error(f"Image processing pipeline execution failure: {e}")
        return {"error": f"Image processing pipeline execution failure: {str(e)}"}