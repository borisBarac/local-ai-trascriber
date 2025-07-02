import pytest
from ..llm_fixer import build_correction_chain, fix_text, create_corrector, ModelType


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


def test_build_correction_chain_returns_async_generator():
    chain = build_correction_chain()
    # Should return a callable
    assert callable(chain)
    # Should return an async generator when called
    gen = chain("test input")
    assert hasattr(gen, "__aiter__")
    assert hasattr(gen, "__anext__")


def test_create_corrector_factory():
    # Test the factory function with default parameters
    corrector = create_corrector()
    assert callable(corrector)

    # Test with custom parameters
    custom_examples = [{"input": "test input", "output": "test output"}]
    custom_corrector = create_corrector(
        model_type=ModelType.OLLAMA,
        examples=custom_examples,
        system_prompt="Custom system prompt",
    )
    assert callable(custom_corrector)


def test_create_corrector_returns_async_generator():
    corrector = create_corrector()
    gen = corrector("test input")
    assert hasattr(gen, "__aiter__")
    assert hasattr(gen, "__anext__")


@pytest.mark.asyncio
async def test_fix_text():
    async def fake_correction_chain(text):
        yield f"fixed: {text}"

    # Patch build_correction_chain to return our fake
    from .. import llm_fixer

    orig = llm_fixer.build_correction_chain
    llm_fixer.build_correction_chain = lambda: fake_correction_chain
    try:
        result = await fix_text("He want go park after lunch.")
        assert result == "fixed: He want go park after lunch."
    finally:
        llm_fixer.build_correction_chain = orig


@pytest.mark.asyncio
async def test_fix_text_with_chunks():
    async def fake_correction_chain_chunks(text):
        words = f"corrected: {text}".split()
        for word in words:
            yield f"{word} "

    # Patch build_correction_chain to return our fake
    from .. import llm_fixer

    orig = llm_fixer.build_correction_chain
    llm_fixer.build_correction_chain = lambda: fake_correction_chain_chunks
    try:
        result = await fix_text("test input")
        assert result == "corrected: test input"
    finally:
        llm_fixer.build_correction_chain = orig


@pytest.mark.asyncio
async def test_create_corrector_with_custom_examples():
    custom_examples = [{"input": "hello world", "output": "Hello, World!"}]

    async def mock_llm_astream(prompt):
        # Verify the custom example is in the prompt
        assert "hello world" in prompt
        assert "Hello, World!" in prompt
        yield "mocked response"

    # Test that custom examples are used in prompt construction
    from .. import llm_fixer

    orig_llm = llm_fixer.llm

    class MockLLM:
        def astream(self, prompt):
            return mock_llm_astream(prompt)

    llm_fixer.llm = MockLLM()

    try:
        corrector = create_corrector(examples=custom_examples)
        result_gen = corrector("test")
        result = []
        async for chunk in result_gen:
            result.append(chunk)
        assert "".join(result) == "mocked response"
    finally:
        llm_fixer.llm = orig_llm


@pytest.mark.asyncio
async def test_create_corrector_error_handling():
    async def failing_astream(prompt):
        # This is an async generator that immediately raises
        raise Exception("LLM connection failed")
        yield  # This is never reached, but makes it an async generator

    from .. import llm_fixer

    orig_llm = llm_fixer.llm

    class FailingLLM:
        def astream(self, prompt):
            async def gen(prompt):
                raise Exception("LLM connection failed")
                yield  # never reached

            return gen(prompt)

    llm_fixer.llm = FailingLLM()

    try:
        corrector = create_corrector()
        result_gen = corrector("test")
        result = []
        async for chunk in result_gen:
            result.append(chunk)
        result_text = "".join(result)
        assert "CORRECTION_ERROR" in result_text
        assert "LLM connection failed" in result_text
    finally:
        llm_fixer.llm = orig_llm
