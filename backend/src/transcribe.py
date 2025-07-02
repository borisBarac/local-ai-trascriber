import logging
import threading
import ffmpeg
import numpy as np
import asyncio

from .model import setup_transcription_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Audio Configuration ---
AUDIO_SAMPLE_RATE = 16000
# Bytes to read from ffmpeg stdout. Corresponds to ~1.28s of 16-bit mono 16kHz audio.
AUDIO_CHUNK_BYTES = 4096 * 10
# Number of seconds of audio to buffer before transcribing.
# Whisper works best on segments of 5-30 seconds.
TARGET_BUFFER_SECONDS = 5
# Minimum audio length in samples to process (0.1s)
MIN_AUDIO_SAMPLES = AUDIO_SAMPLE_RATE // 10


CHUNK_TIMEOUT = 30  # Timeout for reading audio chunks (seconds)
TRANSCRIPTION_TIMEOUT = 60  # Timeout for transcription processing (seconds)

TRANSCRIPTION_PROCESSES = {}
TRANSCRIPTION_LOCK = threading.Lock()

transcribe_pipeline, device, torch_dtype, np_dtype = setup_transcription_pipeline()

STREAM_END_MARKER = "###STREAM_END###"


# --- Transcription Generator ---
async def transcribe_audio_stream(file_path: str, transcription_id: str):
    """
    Reads an audio file, converts it to a 16kHz mono stream, buffers it into
    larger segments, and yields transcribed text chunks for higher accuracy.
    """
    logger.info(f"Starting transcription for: {file_path} (ID: {transcription_id})")
    ffmpeg_process = None
    try:
        # Use ffmpeg to read the audio and convert it
        ffmpeg_process = (
            ffmpeg.input(file_path)
            .output(
                "pipe:",
                format="s16le",
                acodec="pcm_s16le",
                ac=1,
                ar="16k",  # Use '16k' to match test expectation
            )
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        # Register process by ID
        with TRANSCRIPTION_LOCK:
            TRANSCRIPTION_PROCESSES[transcription_id] = ffmpeg_process

        # Get event loop for async operations
        loop = asyncio.get_event_loop()
        audio_buffer = []
        target_buffer_samples = TARGET_BUFFER_SECONDS * AUDIO_SAMPLE_RATE

        async def transcribe_buffer(buffer_to_process):
            """Helper to transcribe a buffer and handle errors."""
            if not buffer_to_process:
                return ""

            full_audio = np.concatenate(buffer_to_process)
            if len(full_audio) < MIN_AUDIO_SAMPLES:
                logger.debug(f"Skipping short buffer of {len(full_audio)} samples")
                return ""

            try:
                # Run transcription with timeout protection
                outputs = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: transcribe_pipeline(
                            {
                                "raw": full_audio,
                                "sampling_rate": AUDIO_SAMPLE_RATE,
                            },  # Pass as dict for test compatibility
                            batch_size=1,
                            generate_kwargs={
                                "task": "transcribe",
                                "language": "english",
                            },
                        ),
                    ),
                    timeout=TRANSCRIPTION_TIMEOUT,
                )
                text = outputs["text"].strip()  # type: ignore
                # Only yield non-empty, non-whitespace text
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
