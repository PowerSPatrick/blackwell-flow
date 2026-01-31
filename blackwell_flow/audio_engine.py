"""
Audio capture and Voice Activity Detection (VAD) for Blackwell-Flow.
"""

from __future__ import annotations

import asyncio
import io
import threading
import wave
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd
import structlog

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .config import AudioConfig

logger = structlog.get_logger(__name__)


class SileroVAD:
    """Silero Voice Activity Detection wrapper."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load the Silero VAD model."""
        import torch

        with self._lock:
            if self._model is None:
                logger.info("Loading Silero VAD model")
                self._model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    trust_repo=True,
                )
                self._get_speech_timestamps = utils[0]
                self._model.eval()
                logger.info("Silero VAD model loaded")

    def is_speech(self, audio: NDArray[np.float32], sample_rate: int = 16000) -> bool:
        """Check if audio chunk contains speech."""
        import torch

        if self._model is None:
            self.load()

        with self._lock:
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # Ensure correct sample rate
            if sample_rate != 16000:
                import torchaudio.functional as F
                audio_tensor = F.resample(audio_tensor, sample_rate, 16000)

            speech_prob = self._model(audio_tensor.squeeze(), 16000).item()
            return speech_prob > self.threshold

    def get_speech_segments(
        self,
        audio: NDArray[np.float32],
        sample_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        speech_pad_ms: int = 100,
    ) -> list[dict[str, int]]:
        """Get speech segments from audio."""
        import torch

        if self._model is None:
            self.load()

        with self._lock:
            audio_tensor = torch.from_numpy(audio).float()

            segments = self._get_speech_timestamps(
                audio_tensor,
                self._model,
                sampling_rate=sample_rate,
                min_speech_duration_ms=min_speech_duration_ms,
                speech_pad_ms=speech_pad_ms,
            )
            return segments


class AudioRecorder:
    """Handles audio recording with VAD support."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self.vad = SileroVAD(threshold=config.vad_threshold)

        self._recording = False
        self._audio_buffer: list[NDArray[np.float32]] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._callback: Callable[[bytes], None] | None = None

    def initialize(self) -> None:
        """Initialize the audio subsystem and VAD."""
        logger.info("Initializing audio engine", sample_rate=self.config.sample_rate)
        self.vad.load()
        logger.info("Audio engine initialized")

    def _audio_callback(
        self,
        indata: NDArray[np.float32],
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback for audio stream."""
        if status:
            logger.warning("Audio callback status", status=str(status))

        if self._recording:
            with self._lock:
                self._audio_buffer.append(indata.copy())

    def start_recording(self) -> None:
        """Start recording audio."""
        if self._recording:
            return

        with self._lock:
            self._audio_buffer.clear()
            self._recording = True

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=self.config.blocksize,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("Recording started")

    def stop_recording(self) -> bytes:
        """Stop recording and return audio data as WAV bytes."""
        if not self._recording:
            return b""

        self._recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            if not self._audio_buffer:
                logger.warning("No audio captured")
                return b""

            # Concatenate all audio chunks
            audio_data = np.concatenate(self._audio_buffer, axis=0)
            self._audio_buffer.clear()

        # Apply VAD to trim silence
        audio_data = self._apply_vad(audio_data)

        if len(audio_data) == 0:
            logger.warning("No speech detected in recording")
            return b""

        # Convert to WAV bytes
        wav_bytes = self._to_wav_bytes(audio_data)

        duration_s = len(audio_data) / self.config.sample_rate
        logger.info("Recording stopped", duration_s=round(duration_s, 2))

        return wav_bytes

    def _apply_vad(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply VAD to extract speech segments."""
        # Flatten if needed
        if audio.ndim > 1:
            audio = audio.flatten()

        segments = self.vad.get_speech_segments(
            audio,
            sample_rate=self.config.sample_rate,
            min_speech_duration_ms=self.config.min_speech_duration_ms,
            speech_pad_ms=self.config.speech_pad_ms,
        )

        if not segments:
            return np.array([], dtype=np.float32)

        # Concatenate all speech segments
        speech_chunks = []
        for seg in segments:
            start = seg["start"]
            end = seg["end"]
            speech_chunks.append(audio[start:end])

        return np.concatenate(speech_chunks) if speech_chunks else np.array([], dtype=np.float32)

    def _to_wav_bytes(self, audio: NDArray[np.float32]) -> bytes:
        """Convert float32 audio to WAV bytes."""
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)

        # Write to WAV in memory
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(self.config.channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.config.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        buffer.seek(0)
        return buffer.read()

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording


class AsyncAudioRecorder:
    """Async wrapper for AudioRecorder."""

    def __init__(self, config: AudioConfig) -> None:
        self._recorder = AudioRecorder(config)
        self._loop: asyncio.AbstractEventLoop | None = None

    async def initialize(self) -> None:
        """Initialize the audio engine."""
        self._loop = asyncio.get_event_loop()
        await self._loop.run_in_executor(None, self._recorder.initialize)

    async def start_recording(self) -> None:
        """Start recording audio."""
        if self._loop is None:
            self._loop = asyncio.get_event_loop()
        await self._loop.run_in_executor(None, self._recorder.start_recording)

    async def stop_recording(self) -> bytes:
        """Stop recording and return audio data."""
        if self._loop is None:
            self._loop = asyncio.get_event_loop()
        return await self._loop.run_in_executor(None, self._recorder.stop_recording)

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recorder.is_recording
