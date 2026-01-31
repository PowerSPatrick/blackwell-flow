# Blackwell-Flow

A system-wide AI dictation utility providing instantaneous, context-aware speech-to-text with LLM refinement. Designed for NVIDIA RTX 5090 (Blackwell architecture) but compatible with any CUDA-capable GPU.

## Features

- **Hold-to-Talk (HTT)**: Capture audio only while the hotkey is held (default: `Ctrl+Space`)
- **Super Mode (LLM Refinement)**: Automatically refine transcripts using a local LLM
  - Remove disfluencies ("um", "uh", "like")
  - Correct technical jargon
  - Apply formatting based on context
- **Styles & Presets**: Quick-switch between "Professional," "Casual," "Creative," and "Bullet Points"
- **Custom Vocabulary**: User-editable `dictation_map.json` for specialized terms
- **Multi-Language Autodetect**: Seamless language switching
- **VRAM Resident**: Models stay loaded for <100ms cold-start inference

## Technical Stack

| Component | Technology |
|-----------|------------|
| Host Language | Python 3.12+ |
| STT Engine | faster-whisper (large-v3-turbo) |
| LLM Refiner | llama-cpp-python (Llama-3.1-8B Q8_0) |
| Compute | CUDA 12.8+ |
| Optimization | Flash Attention 2, FP8/FP16 |
| OS Integration | pynput, PyAutoGUI, pyperclip |
| GUI/HUD | PyQt6 (Frameless overlay) |

## Requirements

- NVIDIA GPU with CUDA 12.8+ support
- 16GB+ VRAM recommended (6GB Whisper + 10GB LLM)
- Python 3.12+
- Windows 10/11

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/blackwell-flow.git
cd blackwell-flow

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download models (first run will auto-download)
python -m blackwell_flow.app
```

## Usage

```bash
# Start the application
python -m blackwell_flow.app

# Or with specific options
python -m blackwell_flow.app --style professional --hotkey "ctrl+space"
```

### Hotkeys

| Hotkey | Action |
|--------|--------|
| `Ctrl+Space` (hold) | Record audio |
| `Ctrl+Shift+S` | Cycle styles |
| `Ctrl+Shift+Q` | Quit application |

### Styles

- **Professional**: Formal, clear language suitable for business
- **Casual**: Relaxed, conversational tone
- **Creative**: Expressive, varied vocabulary
- **Bullet Points**: Structured list format

## Configuration

Edit `config.json` to customize:

```json
{
  "hotkey": "ctrl+space",
  "style": "professional",
  "whisper_model": "large-v3-turbo",
  "llm_model": "Llama-3.1-8B-Q8_0",
  "sample_rate": 16000,
  "vad_threshold": 0.5
}
```

## Architecture

```
blackwell_flow/
├── __init__.py
├── app.py              # Main entry point & asyncio loop
├── audio_engine.py     # Audio capture, VAD, buffering
├── inference_engine.py # Whisper STT + LLM refinement
├── ui_manager.py       # PyQt6 system tray & HUD
├── text_injector.py    # Clipboard & paste injection
├── config.py           # Configuration management
└── styles.py           # Style presets
```

## Performance Targets

- Total latency (Release to Paste): <400ms on RTX 5090
- VRAM allocation: ~16GB total
- Cold-start inference: <100ms

## License

MIT License - See LICENSE file for details.
