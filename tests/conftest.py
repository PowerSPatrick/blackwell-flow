"""
Pytest fixtures and configuration for Blackwell-Flow tests.
"""

from __future__ import annotations

import io
import wave
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from blackwell_flow.config import AppConfig, AudioConfig, HotkeyConfig, LLMConfig, UIConfig, WhisperConfig


@pytest.fixture
def mock_audio_config() -> AudioConfig:
    """Create a test audio configuration."""
    return AudioConfig(
        sample_rate=16000,
        channels=1,
        dtype="float32",
        blocksize=1024,
        vad_threshold=0.5,
        min_speech_duration_ms=100,
        max_speech_duration_s=10.0,
        speech_pad_ms=50,
    )


@pytest.fixture
def mock_whisper_config() -> WhisperConfig:
    """Create a test Whisper configuration."""
    return WhisperConfig(
        model_size="tiny",  # Use tiny for tests
        compute_type="float32",
        device="cpu",  # Use CPU for tests
        device_index=0,
        beam_size=1,
        vad_filter=False,
        language="en",
    )


@pytest.fixture
def mock_llm_config() -> LLMConfig:
    """Create a test LLM configuration."""
    return LLMConfig(
        model_path="models/test.gguf",
        n_ctx=512,
        n_gpu_layers=0,  # CPU for tests
        n_batch=128,
        flash_attn=False,
        verbose=False,
    )


@pytest.fixture
def mock_hotkey_config() -> HotkeyConfig:
    """Create a test hotkey configuration."""
    return HotkeyConfig(
        record_hotkey="ctrl+space",
        cycle_style_hotkey="ctrl+shift+s",
        quit_hotkey="ctrl+shift+q",
    )


@pytest.fixture
def mock_ui_config() -> UIConfig:
    """Create a test UI configuration."""
    return UIConfig(
        show_hud=False,  # Disable HUD for tests
        hud_position="top",
        hud_opacity=0.85,
        indicator_color="#FF4444",
        indicator_size=12,
    )


@pytest.fixture
def mock_app_config(
    mock_whisper_config: WhisperConfig,
    mock_llm_config: LLMConfig,
    mock_audio_config: AudioConfig,
    mock_hotkey_config: HotkeyConfig,
    mock_ui_config: UIConfig,
) -> AppConfig:
    """Create a test application configuration."""
    return AppConfig(
        whisper=mock_whisper_config,
        llm=mock_llm_config,
        audio=mock_audio_config,
        hotkeys=mock_hotkey_config,
        ui=mock_ui_config,
        style="professional",
        enable_refinement=True,
        fallback_on_error=True,
        log_level="DEBUG",
    )


def generate_sine_wave(
    frequency: float = 440.0,
    duration: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a sine wave for testing."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def generate_speech_like_audio(
    duration: float = 2.0,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Generate audio that resembles speech patterns for testing."""
    samples = int(sample_rate * duration)
    audio = np.zeros(samples, dtype=np.float32)

    # Mix of frequencies to simulate speech formants
    frequencies = [150, 300, 600, 1200, 2400]
    amplitudes = [0.3, 0.25, 0.2, 0.15, 0.1]

    t = np.linspace(0, duration, samples, dtype=np.float32)

    for freq, amp in zip(frequencies, amplitudes):
        audio += amp * np.sin(2 * np.pi * freq * t)

    # Add amplitude modulation to simulate speech rhythm
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    audio *= modulation

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8

    return audio.astype(np.float32)


def audio_to_wav_bytes(
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> bytes:
    """Convert float32 audio array to WAV bytes."""
    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)

    # Write to WAV in memory
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def simulated_audio() -> bytes:
    """Generate simulated speech audio as WAV bytes."""
    audio = generate_speech_like_audio(duration=2.0, sample_rate=16000)
    return audio_to_wav_bytes(audio, sample_rate=16000)


@pytest.fixture
def short_audio() -> bytes:
    """Generate short audio sample."""
    audio = generate_sine_wave(frequency=440.0, duration=0.5, sample_rate=16000)
    return audio_to_wav_bytes(audio, sample_rate=16000)


@pytest.fixture
def mock_whisper_model():
    """Create a mock Whisper model."""
    mock_model = MagicMock()

    # Mock transcribe output
    mock_segment = MagicMock()
    mock_segment.text = "This is a test transcription with um some filler words."

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99

    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    return mock_model


@pytest.fixture
def mock_llm_model():
    """Create a mock LLM model."""
    mock_model = MagicMock()

    # Mock completion output
    mock_model.return_value = {
        "choices": [
            {
                "text": "This is a test transcription with some filler words.",
            }
        ]
    }

    return mock_model
