"""
Tests for audio engine.
"""

from __future__ import annotations

import io
import sys
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from blackwell_flow.audio_engine import AudioRecorder, SileroVAD
from blackwell_flow.config import AudioConfig

from .conftest import audio_to_wav_bytes, generate_sine_wave, generate_speech_like_audio

# Check if torch is available
torch_available = False
try:
    import torch
    torch_available = True
except ImportError:
    pass


class TestSileroVAD:
    """Tests for Silero VAD wrapper."""

    def test_init(self) -> None:
        """Test VAD initialization."""
        vad = SileroVAD(threshold=0.6)

        assert vad.threshold == 0.6
        assert vad._model is None

    @pytest.mark.skipif(not torch_available, reason="torch not installed")
    @patch("torch.hub.load")
    def test_load(self, mock_hub_load: MagicMock) -> None:
        """Test VAD model loading."""
        mock_model = MagicMock()
        mock_utils = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        mock_hub_load.return_value = (mock_model, mock_utils)

        vad = SileroVAD()
        vad.load()

        mock_hub_load.assert_called_once()
        assert vad._model is not None

    @pytest.mark.skipif(not torch_available, reason="torch not installed")
    @patch("torch.hub.load")
    def test_is_speech_with_speech(self, mock_hub_load: MagicMock) -> None:
        """Test speech detection with speech audio."""
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(item=MagicMock(return_value=0.8))
        mock_utils = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        mock_hub_load.return_value = (mock_model, mock_utils)

        vad = SileroVAD(threshold=0.5)
        vad.load()

        audio = generate_speech_like_audio(duration=1.0)
        result = vad.is_speech(audio)

        assert result is True

    @pytest.mark.skipif(not torch_available, reason="torch not installed")
    @patch("torch.hub.load")
    def test_is_speech_with_silence(self, mock_hub_load: MagicMock) -> None:
        """Test speech detection with silence."""
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(item=MagicMock(return_value=0.1))
        mock_utils = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        mock_hub_load.return_value = (mock_model, mock_utils)

        vad = SileroVAD(threshold=0.5)
        vad.load()

        # Generate silence
        audio = np.zeros(16000, dtype=np.float32)
        result = vad.is_speech(audio)

        assert result is False


class TestAudioRecorder:
    """Tests for AudioRecorder."""

    def test_init(self, mock_audio_config: AudioConfig) -> None:
        """Test recorder initialization."""
        recorder = AudioRecorder(mock_audio_config)

        assert recorder.config == mock_audio_config
        assert recorder._recording is False
        assert recorder._stream is None

    @patch.object(SileroVAD, "load")
    def test_initialize(self, mock_vad_load: MagicMock, mock_audio_config: AudioConfig) -> None:
        """Test recorder initialization."""
        recorder = AudioRecorder(mock_audio_config)
        recorder.initialize()

        mock_vad_load.assert_called_once()

    def test_is_recording_property(self, mock_audio_config: AudioConfig) -> None:
        """Test is_recording property."""
        recorder = AudioRecorder(mock_audio_config)

        assert recorder.is_recording is False

    def test_to_wav_bytes(self, mock_audio_config: AudioConfig) -> None:
        """Test audio to WAV conversion."""
        recorder = AudioRecorder(mock_audio_config)

        # Generate test audio
        audio = generate_sine_wave(duration=0.5)

        # Convert to WAV
        wav_bytes = recorder._to_wav_bytes(audio)

        # Verify WAV format
        with io.BytesIO(wav_bytes) as buffer:
            with wave.open(buffer, "rb") as wav_file:
                assert wav_file.getnchannels() == 1
                assert wav_file.getsampwidth() == 2
                assert wav_file.getframerate() == 16000

    @patch.object(SileroVAD, "get_speech_segments")
    @patch.object(SileroVAD, "load")
    def test_apply_vad(
        self,
        mock_load: MagicMock,
        mock_segments: MagicMock,
        mock_audio_config: AudioConfig,
    ) -> None:
        """Test VAD application."""
        # Mock speech segments
        mock_segments.return_value = [
            {"start": 1000, "end": 5000},
            {"start": 6000, "end": 10000},
        ]

        recorder = AudioRecorder(mock_audio_config)
        recorder.initialize()

        # Generate test audio
        audio = generate_speech_like_audio(duration=1.0)

        # Apply VAD
        result = recorder._apply_vad(audio)

        # Result should contain only speech segments
        expected_length = (5000 - 1000) + (10000 - 6000)
        assert len(result) == expected_length

    @patch.object(SileroVAD, "get_speech_segments")
    @patch.object(SileroVAD, "load")
    def test_apply_vad_no_speech(
        self,
        mock_load: MagicMock,
        mock_segments: MagicMock,
        mock_audio_config: AudioConfig,
    ) -> None:
        """Test VAD with no speech detected."""
        mock_segments.return_value = []

        recorder = AudioRecorder(mock_audio_config)
        recorder.initialize()

        audio = np.zeros(16000, dtype=np.float32)
        result = recorder._apply_vad(audio)

        assert len(result) == 0


class TestAudioToWavBytes:
    """Tests for audio conversion utility."""

    def test_conversion(self) -> None:
        """Test basic audio to WAV conversion."""
        audio = generate_sine_wave(duration=1.0)
        wav_bytes = audio_to_wav_bytes(audio)

        # Verify it's valid WAV data
        with io.BytesIO(wav_bytes) as buffer:
            with wave.open(buffer, "rb") as wav_file:
                assert wav_file.getnchannels() == 1
                assert wav_file.getsampwidth() == 2
                assert wav_file.getframerate() == 16000
                assert wav_file.getnframes() == 16000

    def test_custom_sample_rate(self) -> None:
        """Test conversion with custom sample rate."""
        audio = generate_sine_wave(duration=1.0, sample_rate=48000)
        wav_bytes = audio_to_wav_bytes(audio, sample_rate=48000)

        with io.BytesIO(wav_bytes) as buffer:
            with wave.open(buffer, "rb") as wav_file:
                assert wav_file.getframerate() == 48000
