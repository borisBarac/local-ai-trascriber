import os
import platform
import logging
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.pipelines import pipeline
from transformers.utils.import_utils import is_flash_attn_2_available

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_ID = os.getenv("MODEL_ID", "openai/whisper-medium")
MLX_MODEL_ID = os.getenv("MLX_MODEL_ID", "mlx-community/whisper-medium")
USE_BFLOAT16 = False


def _is_apple_silicon():
    return platform.system() == "Darwin" and torch.backends.mps.is_available()


def _mlx_available():
    try:
        import mlx_whisper  # noqa: F401
        return True
    except ImportError:
        return False


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
        torch_dtype = torch.float32
        np_dtype = np.float32
    else:
        torch_dtype = torch.float32
        np_dtype = np.float32
    return torch_dtype, np_dtype


_PIPELINE_CACHE = {}


def setup_transcription_pipeline(model_id=MODEL_ID, use_bfloat16=USE_BFLOAT16):
    cache_key = (model_id, use_bfloat16)
    if cache_key in _PIPELINE_CACHE:
        logger.info(f"Returning cached pipeline for {cache_key}")
        return _PIPELINE_CACHE[cache_key]

    if _is_apple_silicon() and _mlx_available():
        logger.info(f"Apple Silicon detected with MLX available — using mlx-whisper backend")
        logger.info(f"MLX model: {MLX_MODEL_ID}")
        result = ("mlx", MLX_MODEL_ID, None, None)
        _PIPELINE_CACHE[cache_key] = result
        return result

    device = get_device()
    torch_dtype, np_dtype = get_torch_and_np_dtypes(device, use_bfloat16)
    logger.info(
        f"Using device: {device}, torch_dtype: {torch_dtype}, np_dtype: {np_dtype}"
    )

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
        generate_kwargs={"language": "english"},
    )
    logger.info(f"Whisper model loaded successfully: {model_id}")
    result = (transcribe_pipeline, device, torch_dtype, np_dtype)
    _PIPELINE_CACHE[cache_key] = result
    return result
