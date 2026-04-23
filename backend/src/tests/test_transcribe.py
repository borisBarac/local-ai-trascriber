import uuid
import pytest
import os
from unittest.mock import patch, Mock
import numpy as np
from ..transcribe import transcribe_audio_stream


class TestTranscribeAudioStream:
    """Test cases for the transcribe_audio_stream function."""

    @pytest.fixture
    def audio_file_path(self):
        """Fixture providing path to test audio file."""
        return os.path.join(os.path.dirname(__file__), "temp_audio", "output.wav")

    @pytest.fixture
    def short_audio_file_path(self):
        """Fixture providing path to short test audio file."""
        return os.path.join(os.path.dirname(__file__), "temp_audio", "output_short.wav")

    @pytest.fixture
    def mock_transcribe_pipeline(self):
        """Mock the transcribe_pipeline to avoid loading the actual model."""
        with patch("src.transcribe.BACKEND_TYPE", "pytorch"), \
             patch("src.transcribe.transcribe_pipeline") as mock_pipeline:
            mock_pipeline.return_value = {"text": "Hello world"}
            yield mock_pipeline

    @pytest.fixture
    def mock_ffmpeg_process(self):
        """Mock ffmpeg process for controlled testing."""
        with patch("src.transcribe.ffmpeg") as mock_ffmpeg:
            mock_process = Mock()
            mock_process.stdout.read.side_effect = [
                # First chunk - 2 seconds of dummy audio data
                np.random.randint(
                    -32768, 32767, size=4096 * 2, dtype=np.int16
                ).tobytes(),
                # Second chunk
                np.random.randint(
                    -32768, 32767, size=4096 * 2, dtype=np.int16
                ).tobytes(),
                # End of stream
                b"",
            ]
            mock_process.wait.return_value = 0

            mock_ffmpeg.input.return_value.output.return_value.run_async.return_value = mock_process
            yield mock_ffmpeg, mock_process

    @pytest.mark.asyncio
    async def test_transcribe_audio_stream_success(
        self, audio_file_path, mock_transcribe_pipeline, mock_ffmpeg_process
    ):
        """Test successful transcription of audio stream."""
        mock_ffmpeg, mock_process = mock_ffmpeg_process

        # Configure the mock pipeline to return different text for each chunk
        mock_transcribe_pipeline.side_effect = [{"text": "Hello"}, {"text": "world"}]

        results = []
        transcription_id = str(uuid.uuid4())
        async for text_chunk in transcribe_audio_stream(
            audio_file_path, transcription_id
        ):
            results.append(text_chunk)

        # Verify results
        assert len(results) == 3  # 2 chunks + stream end marker
        assert results[0] == "Hello "
        assert results[1] == "world "
        assert results[2] == "###STREAM_END###"

        # Verify ffmpeg was called correctly
        mock_ffmpeg.input.assert_called_once_with(audio_file_path)
        mock_ffmpeg.input.return_value.output.assert_called_once_with(
            "pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar="16k"
        )

        # Verify pipeline was called for each chunk
        assert mock_transcribe_pipeline.call_count == 2

        # Verify process.wait() was called
        mock_process.wait.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_audio_stream_with_empty_text(
        self, audio_file_path, mock_transcribe_pipeline, mock_ffmpeg_process
    ):
        """Test transcription when some chunks return empty text."""
        mock_ffmpeg, mock_process = mock_ffmpeg_process

        # Configure the mock pipeline to return empty text and valid text
        mock_transcribe_pipeline.side_effect = [
            {"text": ""},  # Empty text should be filtered out
            {"text": "Valid text"},
        ]

        results = []
        async for text_chunk in transcribe_audio_stream(
            audio_file_path, str(uuid.uuid4())
        ):
            results.append(text_chunk)

        # Verify only non-empty text is yielded
        assert len(results) == 2  # valid text + stream end marker
        assert results[0] == "Valid text "
        assert results[1] == "###STREAM_END###"

    @pytest.mark.asyncio
    async def test_transcribe_audio_stream_with_whitespace_only(
        self, audio_file_path, mock_transcribe_pipeline, mock_ffmpeg_process
    ):
        """Test transcription when chunks return only whitespace."""
        mock_ffmpeg, mock_process = mock_ffmpeg_process

        # Configure the mock pipeline to return whitespace-only text
        mock_transcribe_pipeline.side_effect = [
            {"text": "   "},  # Whitespace only should be filtered out
            {"text": "Real content"},
        ]

        results = []
        async for text_chunk in transcribe_audio_stream(
            audio_file_path, str(uuid.uuid4())
        ):
            results.append(text_chunk)

        # Verify only non-whitespace text is yielded
        assert len(results) == 2  # real content + stream end marker
        assert results[0] == "Real content "
        assert results[1] == "###STREAM_END###"

    @pytest.mark.asyncio
    async def test_transcribe_audio_stream_file_not_found(
        self, mock_transcribe_pipeline
    ):
        """Test transcription with non-existent file."""
        non_existent_file = "/path/to/non/existent/file.wav"

        with patch("src.transcribe.ffmpeg") as mock_ffmpeg:
            # Simulate ffmpeg error for non-existent file
            mock_ffmpeg.input.return_value.output.return_value.run_async.side_effect = (
                Exception("File not found")
            )

            results = []
            async for text_chunk in transcribe_audio_stream(
                non_existent_file, str(uuid.uuid4())
            ):
                results.append(text_chunk)

            # Should yield error message
            assert len(results) == 1
            assert results[0].startswith("[ERROR:")

    @pytest.mark.asyncio
    async def test_transcribe_audio_stream_pipeline_error(
        self, audio_file_path, mock_transcribe_pipeline, mock_ffmpeg_process
    ):
        """Test transcription when pipeline throws an error."""
        mock_ffmpeg, mock_process = mock_ffmpeg_process

        # Configure the mock pipeline to throw an error
        mock_transcribe_pipeline.side_effect = Exception("Pipeline processing error")

        results = []
        async for text_chunk in transcribe_audio_stream(
            audio_file_path, str(uuid.uuid4())
        ):
            results.append(text_chunk)

        # Should yield an error message for each chunk and stream end marker
        assert len(results) == 3
        assert results[0].startswith("[TRANSCRIPTION_ERROR:")
        assert "Pipeline processing error" in results[0]
        assert results[1].startswith("[TRANSCRIPTION_ERROR:")
        assert "Pipeline processing error" in results[1]
        assert results[2] == "###STREAM_END###"

    @pytest.mark.asyncio
    async def test_transcribe_audio_stream_audio_processing(
        self, audio_file_path, mock_transcribe_pipeline, mock_ffmpeg_process
    ):
        """Test that audio data is processed correctly."""
        mock_ffmpeg, mock_process = mock_ffmpeg_process

        # Create specific test audio data
        test_audio_data = np.random.randint(
            -32768, 32767, size=4096 * 2, dtype=np.int16
        )
        mock_process.stdout.read.side_effect = [
            test_audio_data.tobytes(),
            b"",  # End of stream
        ]

        mock_transcribe_pipeline.return_value = {"text": "Test"}

        results = []
        async for text_chunk in transcribe_audio_stream(
            audio_file_path, str(uuid.uuid4())
        ):
            results.append(text_chunk)

        # Verify the pipeline was called with correct audio format
        call_args = mock_transcribe_pipeline.call_args[0][0]
        assert call_args["sampling_rate"] == 16000
        assert "raw" in call_args

        # Verify audio array conversion
        expected_audio = test_audio_data.astype(np.float32) / 32768.0
        np.testing.assert_array_almost_equal(
            call_args["raw"], expected_audio, decimal=5
        )

    @pytest.mark.asyncio
    async def test_transcribe_audio_stream_pipeline_parameters(
        self, audio_file_path, mock_transcribe_pipeline, mock_ffmpeg_process
    ):
        """Test that the pipeline is called with correct parameters."""
        mock_ffmpeg, mock_process = mock_ffmpeg_process
        mock_transcribe_pipeline.return_value = {"text": "Test"}

        results = []
        async for text_chunk in transcribe_audio_stream(
            audio_file_path, str(uuid.uuid4())
        ):
            results.append(text_chunk)

        # Verify pipeline call parameters
        args, kwargs = mock_transcribe_pipeline.call_args
        assert kwargs["batch_size"] == 1
        assert kwargs["generate_kwargs"] == {
            "task": "transcribe",
            "language": "english",
        }

    @pytest.mark.asyncio
    async def test_transcribe_audio_stream_real_file(self, audio_file_path):
        """Integration test with real audio file (if it exists)."""
        if not os.path.exists(audio_file_path):
            pytest.skip(f"Test audio file not found: {audio_file_path}")

        # This test runs with the actual transcription pipeline
        # It's more of an integration test
        results = []
        chunk_count = 0

        async for text_chunk in transcribe_audio_stream(
            audio_file_path, str(uuid.uuid4())
        ):
            results.append(text_chunk)
            chunk_count += 1
            # Limit chunks to avoid long test times
            if chunk_count >= 3:
                break

        # Basic validation - should get some results
        assert len(results) > 0

        # Each result should be a string
        for result in results:
            assert isinstance(result, str)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
