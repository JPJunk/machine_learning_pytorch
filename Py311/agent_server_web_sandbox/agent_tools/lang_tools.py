# agent_tools/lang_tools.py - Language detection and translation tools for the agent server, utilizing OPUS-CAT for high-speed translations and a local LLM for summarization.
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

import re
from openai import OpenAI

import torch
from transformers import MarianMTModel, MarianTokenizer


from .common import LLAMA_BASE_URL, LLAMA_MODEL
from .common import sanitize_edge_metadata

# Point to the official OPUS-CAT global plugin bridge service endpoint
# OPUS_CAT_URL = os.getenv("OPUS_CAT_URL", "http://localhost:8500/MTRestService/TranslatePost")

# Standard environment routing matching your agent server core configuration
# LLAMA_BASE_URL = os.getenv("LLAMA__BASE_URL", "http://localhost:5001/v1")
# LLAMA_MODEL = os.getenv("LLAMA__MODEL", "qwen3.6-35b-a3b-uncensored-genesis-v2-apex-mtp")
# LLAMA_MODEL = os.getenv("LLAMA__MODEL", "Carwin-MoE-Nano-GGUF")

client = OpenAI(
    base_url=LLAMA_BASE_URL,
    api_key=os.getenv("LLAMA__API_KEY", "not-needed")
)

# TODO: Future enhancement - Add support for additional languages 
# and more robust detection heuristics, potentially integrating 
# a lightweight local language detection library for improved accuracy.

def detect_language(text: str) -> str:
    """Lightweight heuristic language detector optimizing for Finnish, English, Swedish, German."""
    logger.info(f"Detecting language for text length: {len(text)}")
    if not isinstance(text, str) or not text.strip():
        logger.warning("Empty or non-string input for detect_language")
        return "unknown"
        
    clean_text = sanitize_edge_metadata(text)
    text_lower = clean_text.lower()

    # Language detection needs to be include all languages that are included in translate function

    # Finnish structural indicators (looks for common vowel forms and vocabulary tokens)
    if any(ch in text_lower for ch in ["ä", "ö"]):
        if re.search(r"\b(että|mutta|koska|olen|emme|teet|tämä|tuo|siellä|ja|on|se)\b", text_lower):
            logger.info("Detected Finnish language")
            return "fi"

    # Swedish structural indicators
    if any(ch in text_lower for ch in ["å", "ä", "ö"]):
        if re.search(r"\b(och|inte|jag|du|han|hon|det|vi|ni|de|är|en|ett)\b", text_lower):
            logger.info("Detected Swedish language")
            return "sv"

    # German structural indicators
    if re.search(r"\b(und|nicht|ich|du|wir|sie|das|ein|eine|ist|sind|mit|von)\b", text_lower):
        logger.info("Detected German language")
        return "de"
    
    # Spanish structural indicators
    if re.search(r"\b(y|no|yo|tú|él|ella|nosotros|vosotros|ellos|es|son|con|de)\b", text_lower):
        logger.info("Detected Spanish language")
        return "es"

    # English structural indicators
    if re.search(r"\b(the|and|is|are|you|this|that|with|from|have|it|of|for|in)\b", text_lower):
        logger.info("Detected English language")
        return "en"
    
    # Russian structural indicators
    if re.search(r"\b(и|не|я|ты|он|она|мы|вы|они|это|в|на|с|по)\b", text_lower):
        logger.info("Detected Russian language")
        return "ru"
    
    # Chinese structural indicators
    if re.search(r"[\u4e00-\u9fff]", text_lower):
        logger.info("Detected Chinese language")
        return "zh"

    logger.debug("Language detection returned unknown")
    return "unknown"

def summarize(text: str, language: str = "en") -> str:
    """Creates a concise layout overview summary tracking the specified destination language profile."""
    logger.info(f"Summarizing text (length: {len(text)}) in language: {language}")
    if not isinstance(text, str) or not isinstance(language, str):
        logger.warning("Invalid input types for summarization")
        return "[ERROR] Invalid input types for summarization."
        
    clean_text = sanitize_edge_metadata(text)
    if not clean_text.strip():
        logger.warning("Empty input provided to summarize")
        return "[ERROR] Empty summarization input string."

    prompt = f"Summarize the following text cleanly. Write your final structural summary output utilizing language profile formatting matching code '{language}':\n\n{clean_text}"
    try:
        logger.info(f"Sending request to LLM {LLAMA_MODEL} for summarization")
        resp = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        result = resp.choices[0].message.content.strip()
        logger.info("Summarization completed successfully")
        return result
    except Exception as e:
        logger.error(f"Local model text compression process failed: {e}")
        return f"[ERROR] Local model text compression process failed: {str(e)}"

# Pidetään mallit globaalissa muistivälimuistissa nopeita jatkokutsuja varten
_tokenizers = {}
_models = {}

def translate(text: str, target_lang: str) -> str:
    """
    Kääntää tekstiä paikallisesti käyttäen virallisia MarianMT (Helsinki-NLP -malleja.
    Korvaa kankeat ja rajoitetut ulkoiset .exe-prosessit kokonaan.
    """
    logger.info(f"Translating text (length: {len(text)}) to language: {target_lang}")
    if not isinstance(text, str) or not isinstance(target_lang, str):
        logger.warning("Invalid input types for translation")
        return "[ERROR] Invalid input types for translation."

    clean_text = sanitize_edge_metadata(text).strip()
    if not clean_text:
        logger.warning("Empty text provided for translation")
        return "[ERROR] Käännettävä teksti on tyhjä."

    # Normalisoidaan kohdekieli standardimuotoon
    target_lang = sanitize_edge_metadata(target_lang).lower().strip()

    if target_lang in ["fi", "fin", "finnish"]:
        model_name = "Helsinki-NLP/opus-mt-en-fi"
    elif target_lang in ["en", "eng", "english"]:
        model_name = "Helsinki-NLP/opus-mt-fi-en"
    elif target_lang in ["sv", "swe", "swedish"]:
        model_name = "Helsinki-NLP/opus-mt-en-sv"
    elif target_lang in ["es", "spa", "spanish"]:
        model_name = "Helsinki-NLP/opus-mt-en-es"
    # elif target_lang in ["fr", "fre", "french"]:
    #     model_name = "Helsinki-NLP/opus-mt-en-fr"
    # elif target_lang in ["it", "ita", "italian"]:
    #     model_name = "Helsinki-NLP/opus-mt-en-it"
    # elif target_lang in ["de", "ger", "german"]:
    #     model_name = "Helsinki-NLP/opus-mt-en-de"
    elif target_lang in ["ru", "rus", "russian"]:
        model_name = "Helsinki-NLP/opus-mt-en-ru"
    elif target_lang in ["zh", "chi", "chinese"]:
        model_name = "Helsinki-NLP/opus-mt-en-zh"
    else:
        logger.warning(f"Unsupported target language: {target_lang}")
        return f"[ERROR] Tuoneton kohdekieli parametreissa: {target_lang}"

    try:
        # Ladataan malli ja tokenisoija välimuistiin vain ensimmäisellä ajokerralla
        if model_name not in _models:
            logger.info(f"Loading translation model: {model_name}")
            # Pakotetaan suoritus käyttämään Core i5 prosessorisäikeitä optimaalisesti
            torch.set_num_threads(4)
            _tokenizers[model_name] = MarianTokenizer.from_pretrained(model_name)
            _models[model_name] = MarianMTModel.from_pretrained(model_name)

        tokenizer = _tokenizers[model_name]
        model = _models[model_name]

        # Suoritetaan tokenisointi ja dynaaminen tekstikäännös
        logger.info(f"Translating text using {model_name}")
        input_ids = tokenizer(clean_text, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            translated_tokens = model.generate(**input_ids)
        
        # Puretaan valmiit tokenit puhtaaksi tekstiksi
        final_result = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        logger.info("Translation completed successfully")
        return final_result.strip()

    except Exception as err:
        logger.error(f"Paikallisen Python-käännösmoottorin virhe: {err}")
        return f"[ERROR] Paikallisen Python-käännösmoottorin virhe: {str(err)}"
