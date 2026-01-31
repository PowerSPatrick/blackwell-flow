"""
Configuration management for Blackwell-Flow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


StyleType = Literal["professional", "casual", "creative", "bullet_points"]


class WhisperConfig(BaseModel):
    """Whisper STT configuration."""

    model_size: str = "large-v3-turbo"
    compute_type: str = "float16"
    device: str = "cuda"
    device_index: int = 0
    beam_size: int = 5
    vad_filter: bool = True
    language: str | None = None  # None for auto-detect


class LLMConfig(BaseModel):
    """LLM refiner configuration."""

    model_path: str = "models/Llama-3.1-8B-Q8_0.gguf"
    n_ctx: int = 4096
    n_gpu_layers: int = -1  # -1 for all layers on GPU
    n_batch: int = 512
    flash_attn: bool = True
    verbose: bool = False


class AudioConfig(BaseModel):
    """Audio capture configuration."""

    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    blocksize: int = 1024
    vad_threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = 30.0
    speech_pad_ms: int = 100


class HotkeyConfig(BaseModel):
    """Hotkey configuration."""

    record_hotkey: str = "ctrl+space"
    cycle_style_hotkey: str = "ctrl+shift+s"
    quit_hotkey: str = "ctrl+shift+q"


class UIConfig(BaseModel):
    """UI/HUD configuration."""

    show_hud: bool = True
    hud_position: Literal["top", "cursor"] = "top"
    hud_opacity: float = 0.85
    indicator_color: str = "#FF4444"
    indicator_size: int = 12


class AppConfig(BaseSettings):
    """Main application configuration."""

    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    hotkeys: HotkeyConfig = Field(default_factory=HotkeyConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    style: StyleType = "professional"
    enable_refinement: bool = True
    fallback_on_error: bool = True
    log_level: str = "INFO"

    dictation_map_path: Path = Path("dictation_map.json")

    class Config:
        env_prefix = "BLACKWELL_"
        env_nested_delimiter = "__"

    @classmethod
    def load(cls, config_path: Path | None = None) -> AppConfig:
        """Load configuration from file or defaults."""
        if config_path and config_path.exists():
            with open(config_path) as f:
                data = json.load(f)
            return cls(**data)
        return cls()

    def save(self, config_path: Path) -> None:
        """Save configuration to file."""
        with open(config_path, "w") as f:
            json.dump(self.model_dump(), f, indent=2, default=str)


def load_dictation_map(path: Path) -> dict[str, str]:
    """Load custom vocabulary mappings."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}
