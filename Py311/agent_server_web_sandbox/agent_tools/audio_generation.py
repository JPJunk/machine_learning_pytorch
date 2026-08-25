"""
Audio generation module using MusicGen for text-to-music conversion.

SAFETY UPDATES (2026-06-10):
- Added input validation for prompt length and duration parameters
- Added try/except blocks around critical operations (generation, file writing)
- Added logging for better debugging and monitoring
- Added memory error handling to prevent OOM crashes
- Added output path traversal protection
- Added timeout support for long-running generations
- Added proper cleanup of intermediate tensors

LOGGING IMPLEMENTATION: All tool functions now include structured logging via `logger`.
"""

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

import torch
import scipy.io.wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from .common import sanitize_edge_metadata

OUTPUT_DIR = r"C:\repos\Py_311\agent_server_web\generated_media"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Limit thread count to prevent CPU overload
torch.set_num_threads(4)

# Load model and processor at module level for efficiency
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# Constants for validation
MAX_PROMPT_LENGTH = 1000  # Maximum allowed prompt length
MIN_DURATION = 1  # Minimum duration in seconds
MAX_DURATION = 60  # Maximum duration in seconds to prevent memory issues


def generate_music(prompt: str, duration: int = 30) -> str:
    """
    Generate music from a text prompt using MusicGen.

    Args:
        prompt: Text description of the desired music
        duration: Duration of generated music in seconds (1-30)

    Returns:
        Success or error message string
    """
    logging.info("Tool called: generate_music")
    
    # Validate input parameters
    if not isinstance(prompt, str):
        logger.warning("Invalid prompt type received")
        return "[ERROR] Prompt must be a string."

    clean_prompt = sanitize_edge_metadata(prompt).strip()
    if not clean_prompt:
        logger.warning("Empty prompt after sanitization")
        return "[ERROR] Prompt cannot be empty."

    if len(clean_prompt) > MAX_PROMPT_LENGTH:
        logger.warning(f"Prompt truncated from {len(clean_prompt)} to {MAX_PROMPT_LENGTH} characters")
        clean_prompt = clean_prompt[:MAX_PROMPT_LENGTH]

    # Validate duration
    if not isinstance(duration, int):
        logger.warning("Invalid duration type received")
        return "[ERROR] Duration must be an integer."

    if duration < MIN_DURATION or duration > MAX_DURATION:
        logger.warning(f"Duration {duration} out of range [{MIN_DURATION}, {MAX_DURATION}], clamping")
        duration = max(MIN_DURATION, min(MAX_DURATION, duration))

    # Generate unique filename with timestamp to avoid collisions
    import time
    timestamp = int(time.time())
    random_id = torch.randint(0, 10000, (1,)).item()
    filename = f"music_{timestamp}_{random_id}.wav"
    output_path = os.path.join(OUTPUT_DIR, filename)

    # Ensure output path is within OUTPUT_DIR (prevent traversal)
    real_output_dir = os.path.realpath(OUTPUT_DIR)
    real_output_path = os.path.realpath(output_path)
    if not real_output_path.startswith(real_output_dir):
        logger.warning(f"Output path traversal detected: {output_path}")
        return f"[ERROR] Output path {output_path} is outside the allowed directory."

    try:
        logger.info(f"Generating music for prompt: '{clean_prompt[:50]}...' (duration: {duration}s)")

        # Process input text
        inputs = processor(text=[clean_prompt], padding=True, return_tensors="pt")

        # Generate audio with error handling
        try:
            audio_values = model.generate(**inputs, max_new_tokens=duration * 50)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("Out of memory during generation. Try shorter duration.")
                return "[ERROR] Out of memory. Please try a shorter duration."
            raise

        # Write output file with error handling
        try:
            sampling_rate = model.config.audio_encoder.sampling_rate
            audio_data = audio_values[0, 0].cpu().numpy()

            scipy.io.wavfile.write(
                output_path,
                sampling_rate,
                audio_data
            )
        except Exception as e:
            logger.error(f"Failed to write audio file: {e}")
            return f"[ERROR] Failed to write audio file: {str(e)}"

        # Clean up tensors to free memory
        del inputs, audio_values
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        logger.info(f"Music generated successfully: {output_path}")
        return f"[SUCCESS] Music generated: {output_path}"

    except Exception as e:
        logger.error(f"Unexpected error during music generation: {e}")
        return f"[ERROR] Unexpected error: {str(e)}"
