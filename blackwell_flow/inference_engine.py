"""
Inference engine for Whisper STT and LLM refinement.
"""

from __future__ import annotations

import asyncio
import io
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from .config import AppConfig, LLMConfig, StyleType, WhisperConfig

logger = structlog.get_logger(__name__)


class WhisperEngine:
    """Faster-Whisper STT engine."""

    def __init__(self, config: WhisperConfig) -> None:
        self.config = config
        self._model = None

    def load(self) -> None:
        """Load the Whisper model into VRAM."""
        from faster_whisper import WhisperModel

        logger.info(
            "Loading Whisper model",
            model=self.config.model_size,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

        start_time = time.perf_counter()

        self._model = WhisperModel(
            self.config.model_size,
            device=self.config.device,
            device_index=self.config.device_index,
            compute_type=self.config.compute_type,
        )

        load_time = time.perf_counter() - start_time
        logger.info("Whisper model loaded", load_time_s=round(load_time, 2))

    def transcribe(self, audio_bytes: bytes) -> tuple[str, str | None]:
        """
        Transcribe audio bytes to text.

        Returns:
            Tuple of (transcript, detected_language)
        """
        if self._model is None:
            raise RuntimeError("Whisper model not loaded. Call load() first.")

        # Convert WAV bytes to numpy array
        import wave

        with io.BytesIO(audio_bytes) as buffer:
            with wave.open(buffer, "rb") as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        start_time = time.perf_counter()

        segments, info = self._model.transcribe(
            audio,
            beam_size=self.config.beam_size,
            vad_filter=self.config.vad_filter,
            language=self.config.language,
        )

        # Collect all segments
        transcript = " ".join(segment.text.strip() for segment in segments)

        transcribe_time = time.perf_counter() - start_time
        logger.info(
            "Transcription complete",
            duration_s=round(transcribe_time, 3),
            language=info.language,
            language_probability=round(info.language_probability, 2),
        )

        return transcript, info.language

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None


class LLMEngine:
    """Llama.cpp LLM engine for text refinement."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._model = None

    def load(self) -> None:
        """Load the LLM into VRAM."""
        from llama_cpp import Llama

        logger.info(
            "Loading LLM model",
            model_path=self.config.model_path,
            n_gpu_layers=self.config.n_gpu_layers,
        )

        start_time = time.perf_counter()

        self._model = Llama(
            model_path=self.config.model_path,
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.n_gpu_layers,
            n_batch=self.config.n_batch,
            flash_attn=self.config.flash_attn,
            verbose=self.config.verbose,
        )

        load_time = time.perf_counter() - start_time
        logger.info("LLM model loaded", load_time_s=round(load_time, 2))

    def refine(self, text: str, prompt: str, max_tokens: int = 1024) -> str:
        """
        Refine text using the LLM.

        Args:
            text: The raw transcript to refine
            prompt: The full system + user prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Refined text
        """
        if self._model is None:
            raise RuntimeError("LLM model not loaded. Call load() first.")

        start_time = time.perf_counter()

        output = self._model(
            prompt,
            max_tokens=max_tokens,
            stop=["\n\n", "Transcript:", "User:"],
            echo=False,
        )

        refined_text = output["choices"][0]["text"].strip()

        refine_time = time.perf_counter() - start_time
        logger.info(
            "Refinement complete",
            duration_s=round(refine_time, 3),
            input_length=len(text),
            output_length=len(refined_text),
        )

        return refined_text

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None


class InferenceEngine:
    """Combined inference engine managing Whisper and LLM."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.whisper = WhisperEngine(config.whisper)
        self.llm = LLMEngine(config.llm)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="inference")
        self._dictation_map: dict[str, str] = {}

    def load_models(self) -> None:
        """Load both models into VRAM."""
        logger.info("Loading inference models")
        start_time = time.perf_counter()

        # Load models (could be parallelized if needed)
        self.whisper.load()
        self.llm.load()

        total_time = time.perf_counter() - start_time
        logger.info("All models loaded", total_time_s=round(total_time, 2))

    def set_dictation_map(self, dictation_map: dict[str, str]) -> None:
        """Set custom vocabulary mappings."""
        self._dictation_map = dictation_map

    def _apply_dictation_map(self, text: str) -> str:
        """Apply custom vocabulary replacements."""
        for pattern, replacement in self._dictation_map.items():
            text = text.replace(pattern, replacement)
        return text

    def process(
        self,
        audio_bytes: bytes,
        style: StyleType,
        enable_refinement: bool = True,
    ) -> tuple[str, str]:
        """
        Process audio through the full pipeline.

        Args:
            audio_bytes: WAV audio data
            style: The style preset to use for refinement
            enable_refinement: Whether to run LLM refinement

        Returns:
            Tuple of (final_text, raw_transcript)
        """
        from .styles import get_refinement_prompt

        # Step 1: Transcribe
        raw_transcript, language = self.whisper.transcribe(audio_bytes)

        if not raw_transcript.strip():
            return "", ""

        # Apply dictation map
        raw_transcript = self._apply_dictation_map(raw_transcript)

        # Step 2: Refine (if enabled)
        if enable_refinement and self.llm.is_loaded:
            try:
                prompt = get_refinement_prompt(style, raw_transcript)
                refined_text = self.llm.refine(raw_transcript, prompt)
                return refined_text, raw_transcript
            except Exception as e:
                logger.error("LLM refinement failed, using raw transcript", error=str(e))
                if self.config.fallback_on_error:
                    return raw_transcript, raw_transcript
                raise

        return raw_transcript, raw_transcript


class AsyncInferenceEngine:
    """Async wrapper for InferenceEngine."""

    def __init__(self, config: AppConfig) -> None:
        self._engine = InferenceEngine(config)
        self._loop: asyncio.AbstractEventLoop | None = None

    async def load_models(self) -> None:
        """Load models asynchronously."""
        self._loop = asyncio.get_event_loop()
        await self._loop.run_in_executor(None, self._engine.load_models)

    def set_dictation_map(self, dictation_map: dict[str, str]) -> None:
        """Set custom vocabulary mappings."""
        self._engine.set_dictation_map(dictation_map)

    async def process(
        self,
        audio_bytes: bytes,
        style: StyleType,
        enable_refinement: bool = True,
    ) -> tuple[str, str]:
        """Process audio through the inference pipeline."""
        if self._loop is None:
            self._loop = asyncio.get_event_loop()

        return await self._loop.run_in_executor(
            None,
            self._engine.process,
            audio_bytes,
            style,
            enable_refinement,
        )

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._engine.whisper.is_loaded and self._engine.llm.is_loaded
