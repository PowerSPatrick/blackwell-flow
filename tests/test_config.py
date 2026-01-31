"""
Tests for configuration management.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blackwell_flow.config import (
    AppConfig,
    AudioConfig,
    HotkeyConfig,
    LLMConfig,
    UIConfig,
    WhisperConfig,
    load_dictation_map,
)


class TestWhisperConfig:
    """Tests for WhisperConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = WhisperConfig()

        assert config.model_size == "large-v3-turbo"
        assert config.compute_type == "float16"
        assert config.device == "cuda"
        assert config.device_index == 0
        assert config.beam_size == 5
        assert config.vad_filter is True
        assert config.language is None

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = WhisperConfig(
            model_size="tiny",
            compute_type="float32",
            device="cpu",
            language="en",
        )

        assert config.model_size == "tiny"
        assert config.compute_type == "float32"
        assert config.device == "cpu"
        assert config.language == "en"


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LLMConfig()

        assert config.model_path == "models/Llama-3.1-8B-Q8_0.gguf"
        assert config.n_ctx == 4096
        assert config.n_gpu_layers == -1
        assert config.n_batch == 512
        assert config.flash_attn is True
        assert config.verbose is False

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = LLMConfig(
            model_path="models/custom.gguf",
            n_ctx=2048,
            n_gpu_layers=32,
        )

        assert config.model_path == "models/custom.gguf"
        assert config.n_ctx == 2048
        assert config.n_gpu_layers == 32


class TestAudioConfig:
    """Tests for AudioConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = AudioConfig()

        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.dtype == "float32"
        assert config.vad_threshold == 0.5

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = AudioConfig(
            sample_rate=48000,
            channels=2,
            vad_threshold=0.7,
        )

        assert config.sample_rate == 48000
        assert config.channels == 2
        assert config.vad_threshold == 0.7


class TestHotkeyConfig:
    """Tests for HotkeyConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = HotkeyConfig()

        assert config.record_hotkey == "ctrl+space"
        assert config.cycle_style_hotkey == "ctrl+shift+s"
        assert config.quit_hotkey == "ctrl+shift+q"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = HotkeyConfig(
            record_hotkey="caps_lock",
            cycle_style_hotkey="alt+s",
        )

        assert config.record_hotkey == "caps_lock"
        assert config.cycle_style_hotkey == "alt+s"


class TestUIConfig:
    """Tests for UIConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = UIConfig()

        assert config.show_hud is True
        assert config.hud_position == "top"
        assert config.hud_opacity == 0.85
        assert config.indicator_color == "#FF4444"
        assert config.indicator_size == 12


class TestAppConfig:
    """Tests for AppConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = AppConfig()

        assert config.style == "professional"
        assert config.enable_refinement is True
        assert config.fallback_on_error is True
        assert config.log_level == "INFO"

    def test_load_from_file(self, tmp_path: Path) -> None:
        """Test loading configuration from file."""
        config_data = {
            "style": "casual",
            "enable_refinement": False,
            "whisper": {
                "model_size": "tiny",
                "device": "cpu",
            },
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config = AppConfig.load(config_path)

        assert config.style == "casual"
        assert config.enable_refinement is False
        assert config.whisper.model_size == "tiny"
        assert config.whisper.device == "cpu"

    def test_load_missing_file(self) -> None:
        """Test loading from non-existent file returns defaults."""
        config = AppConfig.load(Path("nonexistent.json"))

        assert config.style == "professional"

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading configuration."""
        config = AppConfig(
            style="creative",
            enable_refinement=False,
        )

        config_path = tmp_path / "config.json"
        config.save(config_path)

        loaded = AppConfig.load(config_path)

        assert loaded.style == "creative"
        assert loaded.enable_refinement is False


class TestDictationMap:
    """Tests for dictation map loading."""

    def test_load_valid_file(self, tmp_path: Path) -> None:
        """Test loading valid dictation map."""
        map_data = {
            "pytorch": "PyTorch",
            "cuda": "CUDA",
        }

        map_path = tmp_path / "dictation_map.json"
        with open(map_path, "w") as f:
            json.dump(map_data, f)

        result = load_dictation_map(map_path)

        assert result == map_data

    def test_load_missing_file(self) -> None:
        """Test loading non-existent file returns empty dict."""
        result = load_dictation_map(Path("nonexistent.json"))

        assert result == {}
