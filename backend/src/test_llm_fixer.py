import pytest
from .llm_fixer import build_correction_chain, correct_transcription_stream


@pytest.mark.asyncio
async def test_build_correction_chain_prompt():
    # Test the prompt construction logic by calling the internal prompt builder
    chain = build_correction_chain()
    # The chain should be a callable returning an async generator
    assert callable(chain)
    # We can't test the actual LLM call, but we can check the prompt format indirectly
    # by checking that the generator yields something for a sample input
    # (This will likely error if the LLM is not available, so we just check type)
    gen = chain("I go store yesterday buy apple.")
    assert hasattr(gen, "__aiter__")


@pytest.mark.asyncio
async def test_correct_transcription_stream():
    # Simulate a transcription stream
    async def fake_transcription_stream():
        yield "He want go park after lunch."
        yield "She no understand what I say."

    # Simulate a correction chain
    async def fake_correction_chain(text):
        yield f"fixed: {text}"

    # Run the function
    result = []
    async for chunk in correct_transcription_stream(
        fake_transcription_stream(), fake_correction_chain, debounce_delay=0
    ):
        result.append(chunk)
    assert result == [
        "fixed: He want go park after lunch.",
        "fixed: She no understand what I say.",
    ]


@pytest.mark.asyncio
async def test_correct_transcription_stream_skips_empty():
    async def fake_transcription_stream():
        yield ""
        yield "valid text"

    async def fake_correction_chain(text):
        yield f"corrected: {text}"

    result = []
    async for chunk in correct_transcription_stream(
        fake_transcription_stream(), fake_correction_chain
    ):
        result.append(chunk)
    assert result == ["corrected: valid text"]


@pytest.mark.asyncio
async def test_correct_transcription_stream_debounce():
    import asyncio

    results = []

    # Simulate a transcription stream with two chunks close together
    async def fake_transcription_stream():
        yield "first chunk. "
        await asyncio.sleep(0.1)  # Less than debounce_delay
        yield "second chunk."

    # Correction chain just echoes the input
    async def fake_correction_chain(text):
        yield f"corrected: {text}"

    # Use a debounce_delay longer than the pause between chunks
    async for chunk in correct_transcription_stream(
        fake_transcription_stream(), fake_correction_chain, debounce_delay=0.2
    ):
        results.append(chunk)
    # Both chunks should be combined into a single correction
    assert results == ["corrected: first chunk. second chunk."]
