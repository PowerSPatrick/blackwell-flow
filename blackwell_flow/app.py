"""
Main application entry point for Blackwell-Flow.

Coordinates audio capture, inference, UI, and text injection.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from .audio_engine import AsyncAudioRecorder
from .config import AppConfig, load_dictation_map
from .hotkey_manager import AsyncHotkeyManager
from .inference_engine import AsyncInferenceEngine
from .styles import cycle_style
from .text_injector import AsyncTextInjector
from .ui_manager import UIManager

if TYPE_CHECKING:
    from .config import StyleType

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class BlackwellFlow:
    """Main Blackwell-Flow application."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

        # Components
        self._audio: AsyncAudioRecorder | None = None
        self._inference: AsyncInferenceEngine | None = None
        self._hotkeys: AsyncHotkeyManager | None = None
        self._injector: AsyncTextInjector | None = None
        self._ui: UIManager | None = None

        # State
        self._current_style: StyleType = config.style
        self._running = False
        self._processing = False

        # Event loop integration
        self._loop: asyncio.AbstractEventLoop | None = None

    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing Blackwell-Flow")
        start_time = time.perf_counter()

        # Initialize UI first (needed for Qt event loop)
        self._ui = UIManager(self.config.ui)
        self._ui.set_callbacks(
            on_style_change=self._on_style_change,
            on_quit=self._request_quit,
        )
        self._ui.initialize()
        self._ui.set_style(self._current_style)

        # Initialize audio engine
        self._audio = AsyncAudioRecorder(self.config.audio)
        await self._audio.initialize()

        # Initialize inference engine
        self._inference = AsyncInferenceEngine(self.config)
        await self._inference.load_models()

        # Load dictation map
        dictation_map = load_dictation_map(self.config.dictation_map_path)
        if dictation_map:
            self._inference.set_dictation_map(dictation_map)
            logger.info("Loaded dictation map", entries=len(dictation_map))

        # Initialize text injector
        self._injector = AsyncTextInjector()

        # Initialize hotkey manager
        self._hotkeys = AsyncHotkeyManager(self.config.hotkeys)
        self._hotkeys.set_callbacks(
            on_record_start=self._on_record_start,
            on_record_stop=self._on_record_stop,
            on_cycle_style=self._on_cycle_style,
            on_quit=self._request_quit,
        )

        total_time = time.perf_counter() - start_time
        logger.info("Blackwell-Flow initialized", total_time_s=round(total_time, 2))

        if self._ui:
            self._ui.show_notification(
                "Blackwell-Flow",
                f"Ready! Press {self.config.hotkeys.record_hotkey} to record.",
            )

    def _on_record_start(self) -> None:
        """Handle record hotkey press."""
        if self._processing:
            return

        logger.debug("Starting recording")

        if self._ui:
            self._ui.set_recording(True)

        if self._audio:
            asyncio.run_coroutine_threadsafe(
                self._audio.start_recording(),
                self._loop,
            )

    def _on_record_stop(self) -> None:
        """Handle record hotkey release."""
        if self._processing:
            return

        logger.debug("Stopping recording")

        if self._ui:
            self._ui.set_recording(False)

        # Process the recording
        asyncio.run_coroutine_threadsafe(
            self._process_recording(),
            self._loop,
        )

    async def _process_recording(self) -> None:
        """Process the recorded audio."""
        if self._audio is None or self._inference is None or self._injector is None:
            return

        self._processing = True
        pipeline_start = time.perf_counter()

        try:
            # Get audio data
            audio_bytes = await self._audio.stop_recording()

            if not audio_bytes:
                logger.warning("No audio captured")
                return

            # Process through inference pipeline
            final_text, raw_transcript = await self._inference.process(
                audio_bytes,
                self._current_style,
                enable_refinement=self.config.enable_refinement,
            )

            if not final_text:
                logger.warning("No text transcribed")
                if self._ui:
                    self._ui.show_status("No speech detected")
                return

            # Inject text
            await self._injector.inject(final_text)

            # Calculate total latency
            total_latency = time.perf_counter() - pipeline_start
            logger.info(
                "Pipeline complete",
                total_latency_ms=round(total_latency * 1000, 1),
                text_length=len(final_text),
            )

            # Show success status
            if self._ui:
                latency_ms = round(total_latency * 1000)
                self._ui.show_status(f"Done ({latency_ms}ms)", duration_ms=1000)

        except Exception as e:
            logger.error("Pipeline failed", error=str(e))
            if self._ui:
                self._ui.show_status("Error processing audio")
        finally:
            self._processing = False

    def _on_style_change(self, style: StyleType) -> None:
        """Handle style change from UI."""
        self._current_style = style
        logger.info("Style changed", style=style)

    def _on_cycle_style(self) -> None:
        """Handle style cycle hotkey."""
        self._current_style = cycle_style(self._current_style)

        if self._ui:
            self._ui.set_style(self._current_style)
            self._ui.show_status(f"Style: {self._current_style.replace('_', ' ').title()}")

        logger.info("Style cycled", style=self._current_style)

    def _request_quit(self) -> None:
        """Request application quit."""
        logger.info("Quit requested")
        self._running = False

        if self._ui:
            self._ui.quit()

    async def run(self) -> None:
        """Run the main application loop."""
        self._running = True
        self._loop = asyncio.get_event_loop()

        # Start hotkey listener
        if self._hotkeys:
            self._hotkeys.start()

        logger.info("Blackwell-Flow running")

        try:
            # Run Qt event loop in executor
            if self._ui:
                # Process Qt events periodically
                while self._running:
                    from PyQt6.QtWidgets import QApplication

                    app = QApplication.instance()
                    if app:
                        app.processEvents()
                    await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        """Shutdown all components."""
        logger.info("Shutting down Blackwell-Flow")

        if self._hotkeys:
            self._hotkeys.stop()

        logger.info("Blackwell-Flow stopped")


async def main_async(config: AppConfig) -> int:
    """Async main entry point."""
    app = BlackwellFlow(config)

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        app._request_quit()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    try:
        await app.initialize()
        await app.run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        logger.error("Fatal error", error=str(e))
        return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Blackwell-Flow - AI Dictation Utility")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--style",
        choices=["professional", "casual", "creative", "bullet_points"],
        default=None,
        help="Initial style preset",
    )
    parser.add_argument(
        "--hotkey",
        type=str,
        default=None,
        help="Record hotkey (e.g., 'ctrl+space')",
    )
    parser.add_argument(
        "--no-refinement",
        action="store_true",
        help="Disable LLM refinement (raw Whisper output only)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Load configuration
    config = AppConfig.load(args.config)

    # Apply CLI overrides
    if args.style:
        config.style = args.style
    if args.hotkey:
        config.hotkeys.record_hotkey = args.hotkey
    if args.no_refinement:
        config.enable_refinement = False
    if args.debug:
        config.log_level = "DEBUG"

    # Run the application
    try:
        return asyncio.run(main_async(config))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
