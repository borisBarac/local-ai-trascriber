import os
import logging
import ffmpeg
import numpy as np
import torch
import asyncio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.utils import is_flash_attn_2_available

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_ID = os.getenv("MODEL_ID", "openai/whisper-large-v3")
USE_BFLOAT16 = False  # Set to True if your GPU supports it
CHUNK_TIMEOUT = 30  # Timeout for reading audio chunks (seconds)
TRANSCRIPTION_TIMEOUT = 60  # Timeout for transcription processing (seconds)


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


device = get_device()
torch_dtype, np_dtype = get_torch_and_np_dtypes(device, USE_BFLOAT16)
logger.info(f"Using device: {device}, torch_dtype: {torch_dtype}, np_dtype: {np_dtype}")

# --- Model Loading ---
attention = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
logger.info(f"Using attention: {attention}")

logger.info(f"Loading Whisper model: {MODEL_ID}")
try:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation=attention,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
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
)
logger.info(f"Whisper model loaded successfully: {MODEL_ID}")


# --- Transcription Generator ---
async def transcribe_audio_stream(file_path: str):
    """
    Reads an audio file, converts it to a 16kHz mono stream,
    and yields transcribed text chunks.
    """
    logger.info(f"Starting transcription for: {file_path}")
    try:
        # Use ffmpeg to read the audio and convert it
        process = (
            ffmpeg.input(file_path)
            .output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar="16k")
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

        # Get event loop for async operations
        loop = asyncio.get_event_loop()

        # Process the stream in chunks
        while True:
            try:
                in_bytes = await asyncio.wait_for(
                    loop.run_in_executor(None, process.stdout.read, 4096 * 10),
                    timeout=CHUNK_TIMEOUT,
                )  # Read 10 seconds of audio
                if not in_bytes:
                    break

                audio_array = (
                    np.frombuffer(in_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                )

                # Skip very short chunks that might cause issues
                if len(audio_array) < 1600:  # Less than 0.1 seconds at 16kHz
                    logger.debug(f"Skipping short chunk of {len(audio_array)} samples")
                    continue

                try:
                    # Run transcription with timeout protection
                    outputs = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: transcribe_pipeline(
                                {"sampling_rate": 16000, "raw": audio_array},
                                batch_size=1,
                                generate_kwargs={"task": "transcribe"},
                            ),
                        ),
                        timeout=TRANSCRIPTION_TIMEOUT,
                    )
                    text = outputs["text"].strip()
                    if text:
                        yield text + " "
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Transcription timeout after {TRANSCRIPTION_TIMEOUT}s, skipping chunk"
                    )
                    yield "[TIMEOUT] "
                    continue
                except Exception as transcription_error:
                    logger.error(
                        f"Error during transcription processing: {transcription_error}"
                    )
                    yield f"[TRANSCRIPTION_ERROR: {transcription_error}] "
                    continue

            except asyncio.TimeoutError:
                logger.error(f"Audio read timeout after {CHUNK_TIMEOUT}s")
                yield "[READ_TIMEOUT] "
                break

        # Wait for process to complete with timeout
        try:
            await asyncio.wait_for(loop.run_in_executor(None, process.wait), timeout=10)
            logger.info("Transcription finished.")
        except asyncio.TimeoutError:
            logger.warning("FFmpeg process cleanup timeout")
            if process and process.poll() is None:
                process.terminate()

    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        yield f"[ERROR: {e}]"
