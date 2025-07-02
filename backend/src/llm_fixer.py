from typing import AsyncGenerator, Callable, Optional
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM as Ollama
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from enum import Enum

load_dotenv()


class ModelType(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


MODEL_TYPE = ModelType(os.getenv("MODEL_TYPE", ModelType.OLLAMA.value))
OLLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3.2:1b")  # Default model for Ollama
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")  # Default model for OpenAI

if MODEL_TYPE == ModelType.OLLAMA:
    llm = Ollama(
        model=OLLAMA_MODEL,
    )
elif MODEL_TYPE == ModelType.OPENAI:
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=SecretStr(os.getenv("OPENAI_API_KEY") or "FAKE_KEY"),
        streaming=True,
    )
else:
    raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}. Use 'ollama' or 'openai'.")

# Example n-shot prompt examples
N_SHOT_EXAMPLES = [
    {
        "input": "I go store yesterday buy apple.",
        "output": "I went to the store yesterday to buy an apple.",
    },
    {
        "input": "She no understand what I say.",
        "output": "She doesn't understand what I am saying.",
    },
    {
        "input": "He want go park after lunch.",
        "output": "He wants to go to the park after lunch.",
    },
]


# Build the correction chain: LLM + output parser with n-shot prompt
def build_correction_chain() -> Callable[[str], AsyncGenerator[str, None]]:
    parser = StrOutputParser()

    def build_n_shot_prompt(text: str) -> str:
        examples = "\n".join(
            f"Transcript: {ex['input']}\nCorrected: {ex['output']}"
            for ex in N_SHOT_EXAMPLES
        )
        prompt = (
            "You are an expert transcription corrector. "
            "Given the following possibly error-prone transcript chunk, "
            "fix any transcription errors, improve grammar, and make the text more natural. "
            "Return only the improved text.\n\n"
            f"{examples}\n"
            f"Transcript: {text}\nCorrected: "
        )
        return prompt

    async def correction_chain(text: str) -> AsyncGenerator[str, None]:
        prompt = build_n_shot_prompt(text)
        if MODEL_TYPE == ModelType.OLLAMA:
            async for chunk in llm.astream(prompt):
                yield str(chunk)
        else:
            async for chunk in (llm | parser).astream(prompt):
                yield chunk

    return correction_chain


async def correct_transcription_stream(
    transcription_stream: Optional[AsyncGenerator[str, None]],
    correction_chain: Callable[[str], AsyncGenerator[str, None]],
    buffer: Optional[str] = None,
    debounce_delay: float = 1.0,  # Seconds to wait for a pause in transcription
) -> AsyncGenerator[str, None]:
    """
    A more robust stream processor that debounces the input.

    It collects incoming text chunks and only processes them after there has
    been a pause in the stream for `debounce_delay` seconds. This is more
    efficient for live transcription as it sends more complete thoughts to
    the LLM instead of processing every tiny chunk.
    """
    import asyncio

    if buffer is not None:
        # If a static buffer is provided, process it directly without debouncing.
        async for corrected in correction_chain(buffer):
            yield corrected
        return

    if transcription_stream is None:
        raise ValueError("Either transcription_stream or buffer must be provided.")

    # If debounce_delay is 0, process each chunk individually (for test compatibility)
    if debounce_delay == 0:
        async for chunk in transcription_stream:
            if chunk and isinstance(chunk, str):
                async for corrected in correction_chain(chunk):
                    yield corrected
        return

    output_queue = asyncio.Queue()
    debounce_buffer: list[str] = []
    processing_task: Optional[asyncio.Task] = None

    async def process_and_empty_buffer():
        """Processes the current buffer and puts the result on the output queue."""
        nonlocal debounce_buffer
        if not debounce_buffer:
            return

        text_to_correct = "".join(debounce_buffer)
        debounce_buffer = []  # Clear buffer for the next batch

        try:
            async for corrected_chunk in correction_chain(text_to_correct):
                await output_queue.put(corrected_chunk)
        except Exception as e:
            await output_queue.put(f"[CORRECTION_ERROR: {e}]")

    async def stream_consumer():
        """Reads from the input stream and manages the debouncing logic."""
        nonlocal processing_task
        try:
            async for chunk in transcription_stream:
                if chunk and isinstance(chunk, str):
                    debounce_buffer.append(chunk)

                    if processing_task:
                        processing_task.cancel()

                    async def delayed_processing():
                        await asyncio.sleep(debounce_delay)
                        await process_and_empty_buffer()

                    processing_task = asyncio.create_task(delayed_processing())
        finally:
            # After the stream ends, ensure the final buffer is processed.
            if processing_task and not processing_task.done():
                try:
                    await processing_task
                except asyncio.CancelledError:
                    await process_and_empty_buffer()
            else:
                await process_and_empty_buffer()

            await output_queue.put(None)  # Sentinel to signal the end

    # Start the consumer task in the background.
    asyncio.create_task(stream_consumer())

    # Yield corrected chunks from the output queue.
    while True:
        item = await output_queue.get()
        if item is None:
            break
        yield item


# Usage:
# correction_chain = build_correction_chain()
# async for fixed in correct_transcription_stream(transcription_stream, correction_chain):
#     print(fixed, end="", flush=True)
