import asyncio
from typing import AsyncGenerator, Callable
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM as Ollama
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from enum import Enum
import logging

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

# Initialize logger
logger = logging.getLogger("llm_fixer")

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


def create_corrector(
    model_type: ModelType | None = None,
    examples: list | None = None,
    system_prompt: str | None = None,
) -> Callable[[str], AsyncGenerator[str, None]]:
    """Factory function to create a corrector with custom options."""
    _model_type = model_type or MODEL_TYPE
    _examples = examples or N_SHOT_EXAMPLES
    _system_prompt = system_prompt or (
        "You are an expert transcription corrector. "
        "Fix any transcription errors, improve grammar, and make the text more natural. "
        "Return ONLY the corrected text without any explanations or additional formatting."
    )

    async def corrector(text: str) -> AsyncGenerator[str, None]:
        examples_text = "\n".join(
            f"Transcript: {ex['input']}\nCorrected: {ex['output']}" for ex in _examples
        )
        prompt = (
            f"{_system_prompt}\n\n"
            f"Examples:\n{examples_text}\n\n"
            f"Now correct this transcript:\n"
            f"Transcript: {text}\nCorrected:"
        )

        try:
            if _model_type == ModelType.OLLAMA:
                async for chunk in llm.astream(prompt):
                    if chunk and str(chunk).strip():
                        yield str(chunk)
            else:
                parser = StrOutputParser()
                async for chunk in (llm | parser).astream(prompt):
                    if chunk and chunk.strip():
                        yield chunk
        except Exception as e:
            logger.error(f"Error in factory corrector: {e}")
            yield f"[CORRECTION_ERROR: {e}]"

    return corrector


# Current implementation using Option 3: Factory pattern
def build_correction_chain() -> Callable[[str], AsyncGenerator[str, None]]:
    """Returns a factory-created corrector with default settings."""
    return create_corrector()


async def fix_text(text: str) -> str:
    correction_chain = build_correction_chain()
    corrected_text = []

    async for chunk in correction_chain(text):
        corrected_text.append(chunk)
        print(f"Chunk: {chunk}")
    return "".join(corrected_text).strip()


async def _fake_main():
    text = "An Intersting Txt With Speling And Fornat Errors"
    corrected_text = await fix_text(text)
    print(f"Original: {text}")
    print(f"Corrected: {corrected_text}")


# asyncio.run(_fake_main())
