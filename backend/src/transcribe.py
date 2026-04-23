import logging
import threading
import ffmpeg
import numpy as np
import asyncio

from .model import setup_transcription_pipeline, _PIPELINE_CACHE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Audio Configuration ---
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_BYTES = 4096 * 10
TARGET_BUFFER_SECONDS = 5
MIN_AUDIO_SAMPLES = AUDIO_SAMPLE_RATE // 10


CHUNK_TIMEOUT = 30
TRANSCRIPTION_TIMEOUT = 60

TRANSCRIPTION_PROCESSES = {}
TRANSCRIPTION_LOCK = threading.Lock()

BACKEND_TYPE = None
MLX_MODEL_ID = None
transcribe_pipeline = None
device = None
torch_dtype = None
np_dtype = None

STREAM_END_MARKER = "###STREAM_END###"


def _ensure_pipeline_loaded():
    global BACKEND_TYPE, MLX_MODEL_ID, transcribe_pipeline, device, torch_dtype, np_dtype
    if BACKEND_TYPE is not None:
        return
    backend_info = setup_transcription_pipeline()
    BACKEND_TYPE = backend_info[0]
    if BACKEND_TYPE == "mlx":
        MLX_MODEL_ID = backend_info[1]
    else:
        transcribe_pipeline = backend_info[0]
        device = backend_info[1]
        torch_dtype = backend_info[2]
        np_dtype = backend_info[3]


def _transcribe_mlx_sync(audio_array):
    import mlx_whisper
    return mlx_whisper.transcribe(
        audio_array,
        path_or_hf_repo=MLX_MODEL_ID,
        language="en",
        task="transcribe",
    )


# --- Transcription Generator ---
async def transcribe_audio_stream(file_path: str, transcription_id: str):
    _ensure_pipeline_loaded()
    logger.info(f"Starting transcription for: {file_path} (ID: {transcription_id}) [backend={BACKEND_TYPE}]")
    ffmpeg_process = None
    try:
        ffmpeg_process = (
            ffmpeg.input(file_path)
            .output(
                "pipe:",
                format="s16le",
                acodec="pcm_s16le",
                ac=1,
                ar="16k",
            )
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        with TRANSCRIPTION_LOCK:
            TRANSCRIPTION_PROCESSES[transcription_id] = ffmpeg_process

        loop = asyncio.get_event_loop()
        audio_buffer = []
        target_buffer_samples = TARGET_BUFFER_SECONDS * AUDIO_SAMPLE_RATE

        async def transcribe_buffer(buffer_to_process):
            if not buffer_to_process:
                return ""

            full_audio = np.concatenate(buffer_to_process)
            if len(full_audio) < MIN_AUDIO_SAMPLES:
                logger.debug(f"Skipping short buffer of {len(full_audio)} samples")
                return ""

            try:
                if BACKEND_TYPE == "mlx":
                    outputs = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: _transcribe_mlx_sync(full_audio),
                        ),
                        timeout=TRANSCRIPTION_TIMEOUT,
                    )
                else:
                    outputs = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: transcribe_pipeline(
                                {
                                    "raw": full_audio,
                                    "sampling_rate": AUDIO_SAMPLE_RATE,
                                },
                                batch_size=1,
                                generate_kwargs={
                                    "task": "transcribe",
                                    "language": "english",
                                },
                            ),
                        ),
                        timeout=TRANSCRIPTION_TIMEOUT,
                    )
                text = outputs["text"].strip()
                if text.strip():
                    return text + " "
                else:
                    return ""
            except asyncio.TimeoutError:
                logger.warning(
                    f"Transcription timeout after {TRANSCRIPTION_TIMEOUT}s, skipping chunk"
                )
                return "[TIMEOUT] "
            except Exception as transcription_error:
                logger.error(
                    f"Error during transcription processing: {transcription_error}"
                )
                return f"[TRANSCRIPTION_ERROR: {transcription_error}] "

        # Process the stream in chunks (yield per chunk, not per buffer)
        while True:
            try:
                in_bytes = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, ffmpeg_process.stdout.read, AUDIO_CHUNK_BYTES
                    ),
                    timeout=CHUNK_TIMEOUT,
                )
                if not in_bytes:
                    yield STREAM_END_MARKER
                    break

                audio_array = (
                    np.frombuffer(in_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                )
                # Transcribe and yield for each chunk
                result = await transcribe_buffer([audio_array])
                # Only yield non-empty, non-whitespace results
                if result and result.strip():
                    yield result

            except asyncio.TimeoutError:
                logger.error(f"Audio read timeout after {CHUNK_TIMEOUT}s")
                yield "[READ_TIMEOUT] "
                break

        # No need to process remaining buffer, as each chunk is processed individually

        # Wait for process to complete with timeout
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, ffmpeg_process.wait), timeout=10
            )
            logger.info(f"Transcription finished. (ID: {transcription_id})")
        except asyncio.TimeoutError:
            logger.warning("FFmpeg process cleanup timeout")
            if ffmpeg_process and ffmpeg_process.poll() is None:
                ffmpeg_process.terminate()

    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        yield f"[ERROR: {e}]"
    finally:
        # Cleanup process from registry
        with TRANSCRIPTION_LOCK:
            TRANSCRIPTION_PROCESSES.pop(transcription_id, None)


def kill_transcription(transcription_id: str):
    """
    Kill the ffmpeg process associated with a transcription ID.
    """
    with TRANSCRIPTION_LOCK:
        process = TRANSCRIPTION_PROCESSES.get(transcription_id)
        if process:
            try:
                process.terminate()
                logger.info(
                    f"Terminated transcription process with ID: {transcription_id}"
                )
            except Exception as e:
                logger.error(f"Failed to terminate process {transcription_id}: {e}")
            finally:
                TRANSCRIPTION_PROCESSES.pop(transcription_id, None)
        else:
            logger.warning(
                f"No active transcription process found for ID: {transcription_id}"
            )
