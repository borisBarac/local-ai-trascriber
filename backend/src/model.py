import os
import logging
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.utils import is_flash_attn_2_available

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Configuration ---
MODEL_ID = os.getenv("MODEL_ID", "openai/whisper-large-v3")
USE_BFLOAT16 = False  # Set to True if your GPU supports it


# --- Device and Data Type Setup ---
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        return "mps"
    else:
        return "cpu"


def get_torch_and_np_dtypes(device, use_bfloat16):
    if device == "cuda":
        torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        np_dtype = np.float16
    elif device == "mps":
        torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        np_dtype = np.float16
    else:
        torch_dtype = torch.float32
        np_dtype = np.float32
    return torch_dtype, np_dtype


# Singleton cache for model pipeline
_PIPELINE_CACHE = {}


def setup_transcription_pipeline(model_id=MODEL_ID, use_bfloat16=USE_BFLOAT16):
    cache_key = (model_id, use_bfloat16)
    if cache_key in _PIPELINE_CACHE:
        logger.info(f"Returning cached pipeline for {cache_key}")
        return _PIPELINE_CACHE[cache_key]

    device = get_device()
    torch_dtype, np_dtype = get_torch_and_np_dtypes(device, use_bfloat16)
    logger.info(
        f"Using device: {device}, torch_dtype: {torch_dtype}, np_dtype: {np_dtype}"
    )

    # --- Model Loading ---
    attention = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
    logger.info(f"Using attention: {attention}")

    logger.info(f"Loading Whisper model: {model_id}")
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation=attention,
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        logger.error(f"Error loading ASR model: {e}")
        raise

    transcribe_pipeline = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"language": "english"},  # <--- Force English
    )
    logger.info(f"Whisper model loaded successfully: {model_id}")
    result = (transcribe_pipeline, device, torch_dtype, np_dtype)
    _PIPELINE_CACHE[cache_key] = result
    return result
