"""
Integration tests for Blackwell-Flow.

These tests verify the complete pipeline works correctly with mocked models.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from blackwell_flow.audio_engine import AsyncAudioRecorder
from blackwell_flow.config import AppConfig
from blackwell_flow.inference_engine import AsyncInferenceEngine
from blackwell_flow.text_injector import AsyncTextInjector

from .conftest import audio_to_wav_bytes, generate_speech_like_audio


class TestAsyncAudioRecorder:
    """Tests for AsyncAudioRecorder."""

    @pytest.mark.asyncio
    @patch("blackwell_flow.audio_engine.SileroVAD")
    async def test_initialize(
        self,
        mock_vad_class: MagicMock,
        mock_audio_config,
    ) -> None:
        """Test async initialization."""
        mock_vad = MagicMock()
        mock_vad_class.return_value = mock_vad

        recorder = AsyncAudioRecorder(mock_audio_config)
        await recorder.initialize()

        mock_vad.load.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_recording_property(self, mock_audio_config) -> None:
        """Test is_recording property."""
        recorder = AsyncAudioRecorder(mock_audio_config)

        assert recorder.is_recording is False


class TestAsyncInferenceEngine:
    """Tests for AsyncInferenceEngine."""

    @pytest.mark.asyncio
    @patch("llama_cpp.Llama")
    @patch("faster_whisper.WhisperModel")
    async def test_load_models(
        self,
        mock_whisper: MagicMock,
        mock_llama: MagicMock,
        mock_app_config: AppConfig,
    ) -> None:
        """Test async model loading."""
        engine = AsyncInferenceEngine(mock_app_config)
        await engine.load_models()

        mock_whisper.assert_called_once()
        mock_llama.assert_called_once()
        assert engine.is_loaded is True

    @pytest.mark.asyncio
    @patch("llama_cpp.Llama")
    @patch("faster_whisper.WhisperModel")
    async def test_process(
        self,
        mock_whisper_class: MagicMock,
        mock_llama_class: MagicMock,
        mock_app_config: AppConfig,
        mock_whisper_model: MagicMock,
        mock_llm_model: MagicMock,
    ) -> None:
        """Test async processing."""
        mock_whisper_class.return_value = mock_whisper_model
        mock_llama_class.return_value = mock_llm_model

        engine = AsyncInferenceEngine(mock_app_config)
        await engine.load_models()

        audio = generate_speech_like_audio(duration=1.0)
        audio_bytes = audio_to_wav_bytes(audio)

        final_text, raw_transcript = await engine.process(
            audio_bytes,
            style="professional",
            enable_refinement=True,
        )

        assert isinstance(final_text, str)
        assert isinstance(raw_transcript, str)


class TestAsyncTextInjector:
    """Tests for AsyncTextInjector."""

    @pytest.mark.asyncio
    @patch("blackwell_flow.text_injector.time.sleep")
    @patch("blackwell_flow.text_injector.pyautogui")
    @patch("blackwell_flow.text_injector.pyperclip")
    async def test_inject(
        self,
        mock_pyperclip: MagicMock,
        mock_pyautogui: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """Test async text injection."""
        mock_pyperclip.paste.return_value = "original"

        injector = AsyncTextInjector()
        await injector.inject("test text")

        mock_pyperclip.copy.assert_any_call("test text")
        mock_pyautogui.hotkey.assert_called_once_with("ctrl", "v")


class TestFullPipeline:
    """Integration tests for the full pipeline."""

    @pytest.mark.asyncio
    @patch("blackwell_flow.text_injector.time.sleep")
    @patch("blackwell_flow.text_injector.pyautogui")
    @patch("blackwell_flow.text_injector.pyperclip")
    @patch("llama_cpp.Llama")
    @patch("faster_whisper.WhisperModel")
    async def test_transcribe_and_inject(
        self,
        mock_whisper_class: MagicMock,
        mock_llama_class: MagicMock,
        mock_pyperclip: MagicMock,
        mock_pyautogui: MagicMock,
        mock_sleep: MagicMock,
        mock_app_config: AppConfig,
        mock_whisper_model: MagicMock,
        mock_llm_model: MagicMock,
    ) -> None:
        """Test complete transcription and injection pipeline."""
        mock_whisper_class.return_value = mock_whisper_model
        mock_llama_class.return_value = mock_llm_model
        mock_pyperclip.paste.return_value = "original clipboard"

        # Initialize components
        inference = AsyncInferenceEngine(mock_app_config)
        await inference.load_models()

        injector = AsyncTextInjector()

        # Generate test audio
        audio = generate_speech_like_audio(duration=1.0)
        audio_bytes = audio_to_wav_bytes(audio)

        # Process through pipeline
        final_text, raw_transcript = await inference.process(
            audio_bytes,
            style="professional",
            enable_refinement=True,
        )

        # Inject result
        await injector.inject(final_text)

        # Verify injection happened
        mock_pyautogui.hotkey.assert_called_with("ctrl", "v")

    @pytest.mark.asyncio
    @patch("llama_cpp.Llama")
    @patch("faster_whisper.WhisperModel")
    async def test_pipeline_with_dictation_map(
        self,
        mock_whisper_class: MagicMock,
        mock_llama_class: MagicMock,
        mock_app_config: AppConfig,
    ) -> None:
        """Test pipeline applies dictation map."""
        # Setup mock to return text with terms to replace
        mock_segment = MagicMock()
        mock_segment.text = "I use pytorch and cuda for training."

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_class.return_value = mock_whisper

        # Make LLM return the input unchanged for this test
        mock_llm = MagicMock()
        mock_llm.return_value = {
            "choices": [{"text": "I use PyTorch and CUDA for training."}]
        }
        mock_llama_class.return_value = mock_llm

        inference = AsyncInferenceEngine(mock_app_config)
        await inference.load_models()

        # Set dictation map
        inference.set_dictation_map({
            "pytorch": "PyTorch",
            "cuda": "CUDA",
        })

        audio = generate_speech_like_audio(duration=1.0)
        audio_bytes = audio_to_wav_bytes(audio)

        final_text, raw_transcript = await inference.process(
            audio_bytes,
            style="professional",
            enable_refinement=True,
        )

        # Verify dictation map was applied to raw transcript
        assert "PyTorch" in raw_transcript
        assert "CUDA" in raw_transcript

    @pytest.mark.asyncio
    @patch("llama_cpp.Llama")
    @patch("faster_whisper.WhisperModel")
    async def test_pipeline_fallback_on_error(
        self,
        mock_whisper_class: MagicMock,
        mock_llama_class: MagicMock,
        mock_app_config: AppConfig,
        mock_whisper_model: MagicMock,
    ) -> None:
        """Test pipeline falls back to raw transcript on LLM error."""
        mock_whisper_class.return_value = mock_whisper_model

        # Make LLM raise an error
        mock_llm = MagicMock()
        mock_llm.side_effect = RuntimeError("LLM error")
        mock_llama_class.return_value = mock_llm

        mock_app_config.fallback_on_error = True

        inference = AsyncInferenceEngine(mock_app_config)
        await inference.load_models()

        audio = generate_speech_like_audio(duration=1.0)
        audio_bytes = audio_to_wav_bytes(audio)

        # Should not raise, should fall back
        final_text, raw_transcript = await inference.process(
            audio_bytes,
            style="professional",
            enable_refinement=True,
        )

        # Final should equal raw on fallback
        assert final_text == raw_transcript


class TestLatencyRequirements:
    """Tests to verify latency requirements are met."""

    @pytest.mark.asyncio
    @patch("blackwell_flow.text_injector.time.sleep")
    @patch("blackwell_flow.text_injector.pyautogui")
    @patch("blackwell_flow.text_injector.pyperclip")
    @patch("llama_cpp.Llama")
    @patch("faster_whisper.WhisperModel")
    async def test_pipeline_latency(
        self,
        mock_whisper_class: MagicMock,
        mock_llama_class: MagicMock,
        mock_pyperclip: MagicMock,
        mock_pyautogui: MagicMock,
        mock_sleep: MagicMock,
        mock_app_config: AppConfig,
        mock_whisper_model: MagicMock,
        mock_llm_model: MagicMock,
    ) -> None:
        """Test that mocked pipeline completes within reasonable time."""
        import time

        mock_whisper_class.return_value = mock_whisper_model
        mock_llama_class.return_value = mock_llm_model
        mock_pyperclip.paste.return_value = ""

        inference = AsyncInferenceEngine(mock_app_config)
        await inference.load_models()

        injector = AsyncTextInjector()

        audio = generate_speech_like_audio(duration=1.0)
        audio_bytes = audio_to_wav_bytes(audio)

        start_time = time.perf_counter()

        final_text, _ = await inference.process(
            audio_bytes,
            style="professional",
            enable_refinement=True,
        )
        await injector.inject(final_text)

        elapsed = time.perf_counter() - start_time

        # With mocks, this should be very fast
        # In production with real models, target is <400ms on RTX 5090
        assert elapsed < 1.0  # Generous limit for mocked tests
