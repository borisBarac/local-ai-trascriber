import logging
import threading
import ffmpeg
import numpy as np
import asyncio
import uuid

from .model import setup_transcription_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_TIMEOUT = 30  # Timeout for reading audio chunks (seconds)
TRANSCRIPTION_TIMEOUT = 60  # Timeout for transcription processing (seconds)

TRANSCRIPTION_PROCESSES = {}
TRANSCRIPTION_LOCK = threading.Lock()

transcribe_pipeline, device, torch_dtype, np_dtype = setup_transcription_pipeline()


# --- Transcription Generator ---
async def transcribe_audio_stream(file_path: str, transcription_id: str):
    """
    Reads an audio file, converts it to a 16kHz mono stream,
    and yields transcribed text chunks. Each transcription has a unique ID.
    """
    logger.info(f"Starting transcription for: {file_path} (ID: {transcription_id})")
    process = None
    try:
        # Use ffmpeg to read the audio and convert it
        process = (
            ffmpeg.input(file_path)
            .output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar="16k")
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        # Register process by ID
        with TRANSCRIPTION_LOCK:
            TRANSCRIPTION_PROCESSES[transcription_id] = process

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
                                generate_kwargs={
                                    "task": "transcribe",
                                    "language": "english",
                                },
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
            logger.info(f"Transcription finished. (ID: {transcription_id})")
        except asyncio.TimeoutError:
            logger.warning("FFmpeg process cleanup timeout")
            if process and process.poll() is None:
                process.terminate()

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
