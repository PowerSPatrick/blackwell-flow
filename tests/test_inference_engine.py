"""
Tests for inference engine.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from blackwell_flow.config import AppConfig, LLMConfig, WhisperConfig
from blackwell_flow.inference_engine import InferenceEngine, LLMEngine, WhisperEngine

# Check if ML dependencies are available
ml_deps_available = False
try:
    from faster_whisper import WhisperModel
    from llama_cpp import Llama
    ml_deps_available = True
except ImportError:
    pass


class TestWhisperEngine:
    """Tests for WhisperEngine."""

    def test_init(self, mock_whisper_config: WhisperConfig) -> None:
        """Test engine initialization."""
        engine = WhisperEngine(mock_whisper_config)

        assert engine.config == mock_whisper_config
        assert engine._model is None
        assert engine.is_loaded is False

    @pytest.mark.skipif(not ml_deps_available, reason="ML dependencies not installed")
    @patch("blackwell_flow.inference_engine.WhisperModel")
    def test_load(
        self,
        mock_whisper_model: MagicMock,
        mock_whisper_config: WhisperConfig,
    ) -> None:
        """Test model loading."""
        engine = WhisperEngine(mock_whisper_config)
        engine.load()

        mock_whisper_model.assert_called_once_with(
            mock_whisper_config.model_size,
            device=mock_whisper_config.device,
            device_index=mock_whisper_config.device_index,
            compute_type=mock_whisper_config.compute_type,
        )
        assert engine.is_loaded is True

    @pytest.mark.skipif(not ml_deps_available, reason="ML dependencies not installed")
    @patch("blackwell_flow.inference_engine.WhisperModel")
    def test_transcribe(
        self,
        mock_whisper_model_class: MagicMock,
        mock_whisper_config: WhisperConfig,
        simulated_audio: bytes,
        mock_whisper_model: MagicMock,
    ) -> None:
        """Test audio transcription."""
        mock_whisper_model_class.return_value = mock_whisper_model

        engine = WhisperEngine(mock_whisper_config)
        engine.load()

        transcript, language = engine.transcribe(simulated_audio)

        assert isinstance(transcript, str)
        assert len(transcript) > 0
        assert language == "en"

    def test_transcribe_without_load(
        self,
        mock_whisper_config: WhisperConfig,
        simulated_audio: bytes,
    ) -> None:
        """Test transcription without loading model raises error."""
        engine = WhisperEngine(mock_whisper_config)

        with pytest.raises(RuntimeError, match="not loaded"):
            engine.transcribe(simulated_audio)


class TestLLMEngine:
    """Tests for LLMEngine."""

    def test_init(self, mock_llm_config: LLMConfig) -> None:
        """Test engine initialization."""
        engine = LLMEngine(mock_llm_config)

        assert engine.config == mock_llm_config
        assert engine._model is None
        assert engine.is_loaded is False

    @pytest.mark.skipif(not ml_deps_available, reason="ML dependencies not installed")
    @patch("blackwell_flow.inference_engine.Llama")
    def test_load(
        self,
        mock_llama: MagicMock,
        mock_llm_config: LLMConfig,
    ) -> None:
        """Test model loading."""
        engine = LLMEngine(mock_llm_config)
        engine.load()

        mock_llama.assert_called_once_with(
            model_path=mock_llm_config.model_path,
            n_ctx=mock_llm_config.n_ctx,
            n_gpu_layers=mock_llm_config.n_gpu_layers,
            n_batch=mock_llm_config.n_batch,
            flash_attn=mock_llm_config.flash_attn,
            verbose=mock_llm_config.verbose,
        )
        assert engine.is_loaded is True

    @pytest.mark.skipif(not ml_deps_available, reason="ML dependencies not installed")
    @patch("blackwell_flow.inference_engine.Llama")
    def test_refine(
        self,
        mock_llama_class: MagicMock,
        mock_llm_config: LLMConfig,
        mock_llm_model: MagicMock,
    ) -> None:
        """Test text refinement."""
        mock_llama_class.return_value = mock_llm_model

        engine = LLMEngine(mock_llm_config)
        engine.load()

        text = "This is a test with um some filler words."
        prompt = "Clean up this text: " + text

        refined = engine.refine(text, prompt)

        assert isinstance(refined, str)
        mock_llm_model.assert_called_once()

    def test_refine_without_load(
        self,
        mock_llm_config: LLMConfig,
    ) -> None:
        """Test refinement without loading model raises error."""
        engine = LLMEngine(mock_llm_config)

        with pytest.raises(RuntimeError, match="not loaded"):
            engine.refine("text", "prompt")


class TestInferenceEngine:
    """Tests for combined InferenceEngine."""

    @pytest.mark.skipif(not ml_deps_available, reason="ML dependencies not installed")
    @patch("blackwell_flow.inference_engine.Llama")
    @patch("blackwell_flow.inference_engine.WhisperModel")
    def test_load_models(
        self,
        mock_whisper: MagicMock,
        mock_llama: MagicMock,
        mock_app_config: AppConfig,
    ) -> None:
        """Test loading both models."""
        engine = InferenceEngine(mock_app_config)
        engine.load_models()

        mock_whisper.assert_called_once()
        mock_llama.assert_called_once()

    def test_set_dictation_map(self, mock_app_config: AppConfig) -> None:
        """Test setting dictation map."""
        engine = InferenceEngine(mock_app_config)

        dictation_map = {"pytorch": "PyTorch", "cuda": "CUDA"}
        engine.set_dictation_map(dictation_map)

        assert engine._dictation_map == dictation_map

    def test_apply_dictation_map(self, mock_app_config: AppConfig) -> None:
        """Test applying dictation map to text."""
        engine = InferenceEngine(mock_app_config)
        engine.set_dictation_map({"pytorch": "PyTorch", "cuda": "CUDA"})

        text = "I use pytorch with cuda for training."
        result = engine._apply_dictation_map(text)

        assert result == "I use PyTorch with CUDA for training."

    @pytest.mark.skipif(not ml_deps_available, reason="ML dependencies not installed")
    @patch("blackwell_flow.inference_engine.Llama")
    @patch("blackwell_flow.inference_engine.WhisperModel")
    def test_process_full_pipeline(
        self,
        mock_whisper_class: MagicMock,
        mock_llama_class: MagicMock,
        mock_app_config: AppConfig,
        simulated_audio: bytes,
        mock_whisper_model: MagicMock,
        mock_llm_model: MagicMock,
    ) -> None:
        """Test full processing pipeline."""
        mock_whisper_class.return_value = mock_whisper_model
        mock_llama_class.return_value = mock_llm_model

        engine = InferenceEngine(mock_app_config)
        engine.load_models()

        final_text, raw_transcript = engine.process(
            simulated_audio,
            style="professional",
            enable_refinement=True,
        )

        assert isinstance(final_text, str)
        assert isinstance(raw_transcript, str)

    @pytest.mark.skipif(not ml_deps_available, reason="ML dependencies not installed")
    @patch("blackwell_flow.inference_engine.Llama")
    @patch("blackwell_flow.inference_engine.WhisperModel")
    def test_process_without_refinement(
        self,
        mock_whisper_class: MagicMock,
        mock_llama_class: MagicMock,
        mock_app_config: AppConfig,
        simulated_audio: bytes,
        mock_whisper_model: MagicMock,
        mock_llm_model: MagicMock,
    ) -> None:
        """Test processing without LLM refinement."""
        mock_whisper_class.return_value = mock_whisper_model
        mock_llama_class.return_value = mock_llm_model

        engine = InferenceEngine(mock_app_config)
        engine.load_models()

        final_text, raw_transcript = engine.process(
            simulated_audio,
            style="professional",
            enable_refinement=False,
        )

        # Without refinement, final should equal raw
        assert final_text == raw_transcript
        # LLM should not be called
        mock_llm_model.assert_not_called()

    @pytest.mark.skipif(not ml_deps_available, reason="ML dependencies not installed")
    @patch("blackwell_flow.inference_engine.Llama")
    @patch("blackwell_flow.inference_engine.WhisperModel")
    def test_process_fallback_on_llm_error(
        self,
        mock_whisper_class: MagicMock,
        mock_llama_class: MagicMock,
        mock_app_config: AppConfig,
        simulated_audio: bytes,
        mock_whisper_model: MagicMock,
    ) -> None:
        """Test fallback to raw transcript on LLM error."""
        mock_whisper_class.return_value = mock_whisper_model

        # Make LLM raise an error
        mock_llm = MagicMock()
        mock_llm.side_effect = RuntimeError("LLM error")
        mock_llama_class.return_value = mock_llm

        mock_app_config.fallback_on_error = True

        engine = InferenceEngine(mock_app_config)
        engine.load_models()

        final_text, raw_transcript = engine.process(
            simulated_audio,
            style="professional",
            enable_refinement=True,
        )

        # Should fall back to raw transcript
        assert final_text == raw_transcript
