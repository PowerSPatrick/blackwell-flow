"""
Text injection utilities for Blackwell-Flow.

Handles clipboard management and text pasting into active applications.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import pyautogui
import pyperclip
import structlog

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


class TextInjector:
    """Handles text injection via clipboard paste."""

    def __init__(self, paste_delay: float = 0.05) -> None:
        """
        Initialize the text injector.

        Args:
            paste_delay: Delay in seconds before pasting (allows window focus)
        """
        self.paste_delay = paste_delay
        self._original_clipboard: str | None = None

    def _save_clipboard(self) -> None:
        """Save the current clipboard contents."""
        try:
            self._original_clipboard = pyperclip.paste()
        except Exception:
            self._original_clipboard = None

    def _restore_clipboard(self) -> None:
        """Restore the original clipboard contents."""
        if self._original_clipboard is not None:
            try:
                # Small delay before restoring to ensure paste completes
                time.sleep(0.1)
                pyperclip.copy(self._original_clipboard)
            except Exception as e:
                logger.warning("Failed to restore clipboard", error=str(e))
            finally:
                self._original_clipboard = None

    def inject(self, text: str, restore_clipboard: bool = True) -> None:
        """
        Inject text into the active application.

        Args:
            text: The text to inject
            restore_clipboard: Whether to restore original clipboard after paste
        """
        if not text:
            logger.warning("No text to inject")
            return

        start_time = time.perf_counter()

        # Save current clipboard
        if restore_clipboard:
            self._save_clipboard()

        try:
            # Copy text to clipboard
            pyperclip.copy(text)

            # Small delay for clipboard to update
            time.sleep(self.paste_delay)

            # Simulate Ctrl+V paste
            pyautogui.hotkey("ctrl", "v")

            inject_time = time.perf_counter() - start_time
            logger.info(
                "Text injected",
                length=len(text),
                duration_ms=round(inject_time * 1000, 1),
            )
        finally:
            # Restore original clipboard
            if restore_clipboard:
                self._restore_clipboard()

    def inject_raw(self, text: str) -> None:
        """
        Inject text by typing it character by character.

        This is slower but works in applications that don't support paste.
        Use sparingly for special cases.

        Args:
            text: The text to type
        """
        if not text:
            return

        start_time = time.perf_counter()

        # Type the text (slower but more compatible)
        pyautogui.typewrite(text, interval=0.01)

        inject_time = time.perf_counter() - start_time
        logger.info(
            "Text typed",
            length=len(text),
            duration_ms=round(inject_time * 1000, 1),
        )


class AsyncTextInjector:
    """Async wrapper for TextInjector."""

    def __init__(self, paste_delay: float = 0.05) -> None:
        self._injector = TextInjector(paste_delay)
        self._loop: asyncio.AbstractEventLoop | None = None

    async def inject(self, text: str, restore_clipboard: bool = True) -> None:
        """Inject text into the active application."""
        if self._loop is None:
            self._loop = asyncio.get_event_loop()

        await self._loop.run_in_executor(
            None,
            self._injector.inject,
            text,
            restore_clipboard,
        )

    async def inject_raw(self, text: str) -> None:
        """Inject text by typing it character by character."""
        if self._loop is None:
            self._loop = asyncio.get_event_loop()

        await self._loop.run_in_executor(None, self._injector.inject_raw, text)
