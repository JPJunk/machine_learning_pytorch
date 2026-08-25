# agent_tools/audio_tools.py - Native Whisper.cpp and Piper TTS Integration
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
import wave
import numpy as np
from .common import sanitize_edge_metadata


def _is_safe_path(path: str, base_dir: str = "C:/test") -> bool:
    """Check if the resolved path is within the allowed base directory."""
    try:
        real_base = os.path.realpath(base_dir)
        real_path = os.path.realpath(path)
        is_safe = real_path.startswith(real_base + os.sep) or real_path == real_base
        logger.debug(f"Path safety check: '{path}' in '{base_dir}' -> {is_safe}")
        return is_safe
    except Exception as e:
        logger.warning(f"Error checking path safety for '{path}': {e}")
        return False

def _convert_to_whisper_format(input_path: str, output_path: str = None) -> str:
    """Convert audio file to 16kHz mono WAV format required by Whisper.cpp."""
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_whisper{ext}"
    
    logger.info(f"Converting audio to Whisper format: {input_path} -> {output_path}")
    
    try:
        with wave.open(input_path, 'rb') as wf_in:
            n_channels = wf_in.getnchannels()
            sample_width = wf_in.getsampwidth()
            frame_rate = wf_in.getframerate()
            n_frames = wf_in.getnframes()
            
            # Check if already compliant
            if frame_rate == 16000 and n_channels == 1:
                logger.info("Audio already in correct format (16kHz mono)")
                return input_path
            
            raw_data = wf_in.readframes(n_frames)
            
            # Convert to numpy array based on sample width
            if sample_width == 2:
                samples = np.frombuffer(raw_data, dtype=np.int16)
            elif sample_width == 3:
                # Handle 24-bit (pad to 32-bit first)
                raw_bytes = np.frombuffer(raw_data, dtype=np.uint8)
                if len(raw_bytes) % 3 != 0:
                    raw_bytes = np.pad(raw_bytes, (0, 3 - len(raw_bytes) % 3))
                chunks = raw_bytes.reshape(-1, 3)
                samples = (chunks[:, 0].astype(np.int32) << 16) | \
                          (chunks[:, 1].astype(np.int32) << 8) | \
                          chunks[:, 2].astype(np.int32)
            else:
                # Fallback for other widths (e.g., 4 bytes int32 or float)
                if sample_width == 4:
                    samples = np.frombuffer(raw_data, dtype=np.int32).astype(np.int16)
                else:
                    samples = np.frombuffer(raw_data, dtype=np.int8)
            
            # If stereo, take first channel (interleaved data)
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels)[:, 0]
            
            # Resample to 16000 Hz using simple linear interpolation
            target_rate = 16000
            duration = n_frames / frame_rate
            target_frames = int(duration * target_rate)
            
            if target_frames <= 0:
                raise ValueError("Target frames is zero or negative")
                
            indices = np.linspace(0, len(samples) - 1, target_frames)
            resampled = np.interp(indices, np.arange(len(samples)), samples).astype(np.int16)
            
            # Write output WAV
            with wave.open(output_path, 'wb') as wf_out:
                wf_out.setnchannels(1)
                wf_out.setsampwidth(2)  # 16-bit
                wf_out.setframerate(target_rate)
                wf_out.writeframes(resampled.tobytes())
        
        logger.info(f"Audio conversion successful: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Audio conversion failed for {input_path}: {e}")
        raise RuntimeError(f"Audio conversion failed: {e}")

def transcribe_audio(path: str, model_path: str = "C:/whisper/ggml-small.bin") -> str:
    """Transcribe a local 16kHz mono WAV audio file using local Whisper.cpp."""
    clean_path = sanitize_edge_metadata(path).strip('"').strip("'").replace("\\", "/")
    clean_model = sanitize_edge_metadata(model_path).strip('"').strip("'").replace("\\", "/")
    whisper_exe = "C:/whisper/whisper-cli.exe"  # Updated to non-deprecated binary

    logger.info(f"Transcribing audio: {clean_path} with model {clean_model}")
    
    if not os.path.exists(clean_path):
        logger.warning(f"Audio file not found: {clean_path}")
        return f"[ERROR] Audio file not found at: {clean_path}"
    
    # Safety: Check file size to prevent processing huge files (e.g., > 100MB)
    try:
        file_size = os.path.getsize(clean_path)
        if file_size > 100 * 1024 * 1024: # 100 MB limit
            logger.warning(f"Audio file too large: {file_size} bytes")
            return f"[ERROR] Audio file too large ({file_size} bytes). Max allowed is 100MB."
    except OSError as e:
        logger.error(f"Could not read file size for {clean_path}: {e}")
        return f"[ERROR] Could not read file size for {clean_path}: {e}"

    if not os.path.exists(whisper_exe):
        logger.warning(f"Whisper.cpp engine not found at: {whisper_exe}")
        return f"[ERROR] Whisper.cpp engine not found at: {whisper_exe}"
    if not os.path.exists(clean_model):
        logger.warning(f"Whisper model file not found at: {clean_model}")
        return f"[ERROR] Whisper model file not found at: {clean_model}"

    # Validate and convert WAV format if needed
    try:
        with wave.open(clean_path, "rb") as w:
            fps = w.getframerate()
            n_channels = w.getnchannels()
            if fps != 16000 or n_channels != 1:
                # Convert to 16kHz mono
                converted_path = f"{os.path.splitext(clean_path)[0]}_whisper.wav"
                logger.info(f"Converting audio format for Whisper compatibility: {clean_path} -> {converted_path}")
                clean_path = _convert_to_whisper_format(clean_path, converted_path)
    except Exception as wave_err:
        logger.error(f"Failed to verify WAV header details: {wave_err}")
        return f"[ERROR] Failed to verify WAV header details: {wave_err}"

    # whisper.cpp whisper-cli.exe command line arguments for clean text output
    cmd = [
        whisper_exe,
        "-m", clean_model,
        "-f", clean_path,
        "-l", "fi",          # Force Finnish language (and Savonian dialects)
        "-nt",               # No Timestamps - removes timestamps from text
        "-p", "1"            # Use 1 thread (already limited in the call pool)
    ]

    try:
        logger.info(f"Running Whisper.cpp command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=60) # Added timeout
        if result.returncode == 0:
            output_text = result.stdout.strip()
            if not output_text:
                logger.info("Whisper processed audio but found no speech")
                return "[System] Audio was processed, but no spoken speech could be extracted."
            logger.info(f"Transcription successful: {len(output_text)} characters")
            return output_text
        logger.warning(f"Whisper execution failed: {result.stderr}")
        return f"[ERROR] Whisper execution failed: {result.stderr}"
    except subprocess.TimeoutExpired:
        logger.error("Whisper transcription timed out (60s limit)")
        return "[ERROR] Whisper transcription timed out (60s limit)."
    except Exception as e:
        logger.error(f"Runtime audio transcription exception: {e}")
        return f"[ERROR] Runtime audio transcription exception: {str(e)}"


def speak_text(text: str, output_path: str = "C:/repos/Py_311/agent_server_web/generated_media/speech.wav") -> str:
    """Convert text to local Finnish speech wav file using high-performance Piper TTS."""
    clean_text = sanitize_edge_metadata(text).strip()
    clean_output = sanitize_edge_metadata(output_path).strip('"').strip("'").replace("\\", "/")
    piper_exe = "C:/piper/piper.exe"
    model_path = "C:/piper/fi_FI-harri-medium.onnx"

    logger.info(f"Synthesizing text to speech: '{clean_text[:50]}...' -> {clean_output}")
    
    if not clean_text:
        logger.warning("Empty text provided for TTS")
        return "[ERROR] Cannot speak an empty string."
    
    # Safety: Ensure output path is within allowed directory (e.g., C:/repos/Py_311/agent_server_web/generated_media)
    if not _is_safe_path(clean_output, "C:/repos/Py_311/agent_server_web/generated_media"):
        logger.warning(f"Output path '{clean_output}' is outside the allowed directory")
        return f"[ERROR] Output path '{clean_output}' is outside the allowed directory 'C:/repos/Py_311/agent_server_web/generated_media'."

    if not os.path.exists(piper_exe):
        logger.warning(f"Piper TTS engine not found at: {piper_exe}")
        return f"[ERROR] Piper TTS engine not found at: {piper_exe}"
    if not os.path.exists(model_path):
        logger.warning(f"Piper Finnish ONNX voice model not found at: {model_path}")
        return f"[ERROR] Piper Finnish ONNX voice model not found at: {model_path}"

    # Ensure output directory exists (e.g., C:/test/)
    out_dir = os.path.dirname(clean_output)
    if out_dir and not os.path.exists(out_dir):
        logger.info(f"Creating output directory: {out_dir}")
        os.makedirs(out_dir, exist_ok=True)

    # Execute Piper CLI via IPC pipe (stdin -> stdout) directly to file
    try:
        logger.info(f"Starting Piper TTS process for model: {model_path}")
        process = subprocess.Popen(
            [piper_exe, "-m", model_path, "-f", clean_output],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8"
        )
        
        # Feed text live to Piper and wait for completion
        _, stderr_data = process.communicate(input=clean_text)
        
        if process.returncode == 0 and os.path.exists(clean_output) and os.path.getsize(clean_output) > 0:
            logger.info(f"Text-to-speech synthesis successful: {clean_output}")
            return f"Success: Local text synthesized to speech wave successfully at: {clean_output}"
        logger.error(f"Piper synthesis pipeline failed: {stderr_data}")
        return f"[ERROR] Piper synthesis pipeline failed: {stderr_data}"
    except Exception as e:
        logger.error(f"Runtime TTS processing exception: {e}")
        return f"[ERROR] Runtime TTS processing exception: {str(e)}"
