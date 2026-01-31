"""
Global hotkey management for Blackwell-Flow.

Uses pynput for system-wide hotkey detection.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import structlog
from pynput import keyboard
from pynput.keyboard import Key, KeyCode

if TYPE_CHECKING:
    from .config import HotkeyConfig

logger = structlog.get_logger(__name__)


def parse_hotkey(hotkey_str: str) -> set[Key | KeyCode]:
    """Parse a hotkey string into a set of keys."""
    keys: set[Key | KeyCode] = set()

    parts = hotkey_str.lower().replace("+", " ").split()

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Map common key names
        key_map = {
            "ctrl": Key.ctrl_l,
            "control": Key.ctrl_l,
            "alt": Key.alt_l,
            "shift": Key.shift_l,
            "space": Key.space,
            "caps": Key.caps_lock,
            "capslock": Key.caps_lock,
            "caps_lock": Key.caps_lock,
            "tab": Key.tab,
            "enter": Key.enter,
            "return": Key.enter,
            "esc": Key.esc,
            "escape": Key.esc,
        }

        if part in key_map:
            keys.add(key_map[part])
        elif len(part) == 1:
            keys.add(KeyCode.from_char(part))
        else:
            # Try to find as Key attribute
            try:
                keys.add(getattr(Key, part))
            except AttributeError:
                logger.warning(f"Unknown key: {part}")

    return keys


class HotkeyManager:
    """Manages global hotkeys using pynput."""

    def __init__(self, config: HotkeyConfig) -> None:
        self.config = config

        self._record_keys = parse_hotkey(config.record_hotkey)
        self._cycle_keys = parse_hotkey(config.cycle_style_hotkey)
        self._quit_keys = parse_hotkey(config.quit_hotkey)

        self._pressed_keys: set[Key | KeyCode] = set()
        self._listener: keyboard.Listener | None = None
        self._listener_thread: threading.Thread | None = None

        self._on_record_start: Callable[[], None] | None = None
        self._on_record_stop: Callable[[], None] | None = None
        self._on_cycle_style: Callable[[], None] | None = None
        self._on_quit: Callable[[], None] | None = None

        self._recording = False
        self._lock = threading.Lock()

    def set_callbacks(
        self,
        on_record_start: Callable[[], None] | None = None,
        on_record_stop: Callable[[], None] | None = None,
        on_cycle_style: Callable[[], None] | None = None,
        on_quit: Callable[[], None] | None = None,
    ) -> None:
        """Set hotkey callbacks."""
        self._on_record_start = on_record_start
        self._on_record_stop = on_record_stop
        self._on_cycle_style = on_cycle_style
        self._on_quit = on_quit

    def _normalize_key(self, key: Key | KeyCode | None) -> Key | KeyCode | None:
        """Normalize key to handle left/right variants."""
        if key is None:
            return None

        # Normalize left/right modifier keys
        if isinstance(key, Key):
            if key in (Key.ctrl_l, Key.ctrl_r):
                return Key.ctrl_l
            if key in (Key.alt_l, Key.alt_r):
                return Key.alt_l
            if key in (Key.shift_l, Key.shift_r):
                return Key.shift_l

        return key

    def _on_press(self, key: Key | KeyCode | None) -> None:
        """Handle key press."""
        key = self._normalize_key(key)
        if key is None:
            return

        with self._lock:
            self._pressed_keys.add(key)

            # Check for record hotkey (hold-to-talk)
            if self._record_keys <= self._pressed_keys and not self._recording:
                self._recording = True
                logger.debug("Record hotkey pressed")
                if self._on_record_start:
                    self._on_record_start()

            # Check for cycle style hotkey
            if self._cycle_keys <= self._pressed_keys:
                logger.debug("Cycle style hotkey pressed")
                if self._on_cycle_style:
                    self._on_cycle_style()

            # Check for quit hotkey
            if self._quit_keys <= self._pressed_keys:
                logger.debug("Quit hotkey pressed")
                if self._on_quit:
                    self._on_quit()

    def _on_release(self, key: Key | KeyCode | None) -> None:
        """Handle key release."""
        key = self._normalize_key(key)
        if key is None:
            return

        with self._lock:
            # Check if any record key was released
            if self._recording and key in self._record_keys:
                self._recording = False
                logger.debug("Record hotkey released")
                if self._on_record_stop:
                    self._on_record_stop()

            self._pressed_keys.discard(key)

    def start(self) -> None:
        """Start the hotkey listener."""
        if self._listener is not None:
            return

        logger.info(
            "Starting hotkey listener",
            record_hotkey=self.config.record_hotkey,
            cycle_hotkey=self.config.cycle_style_hotkey,
            quit_hotkey=self.config.quit_hotkey,
        )

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def stop(self) -> None:
        """Stop the hotkey listener."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
            logger.info("Hotkey listener stopped")

    @property
    def is_recording(self) -> bool:
        """Check if recording hotkey is held."""
        with self._lock:
            return self._recording


class AsyncHotkeyManager:
    """Async wrapper for HotkeyManager."""

    def __init__(self, config: HotkeyConfig) -> None:
        self._manager = HotkeyManager(config)
        self._loop: asyncio.AbstractEventLoop | None = None

        # Async event for signaling
        self._record_start_event: asyncio.Event | None = None
        self._record_stop_event: asyncio.Event | None = None

    def set_callbacks(
        self,
        on_record_start: Callable[[], None] | None = None,
        on_record_stop: Callable[[], None] | None = None,
        on_cycle_style: Callable[[], None] | None = None,
        on_quit: Callable[[], None] | None = None,
    ) -> None:
        """Set hotkey callbacks."""
        self._manager.set_callbacks(
            on_record_start=on_record_start,
            on_record_stop=on_record_stop,
            on_cycle_style=on_cycle_style,
            on_quit=on_quit,
        )

    def start(self) -> None:
        """Start the hotkey listener."""
        self._manager.start()

    def stop(self) -> None:
        """Stop the hotkey listener."""
        self._manager.stop()

    @property
    def is_recording(self) -> bool:
        """Check if recording hotkey is held."""
        return self._manager.is_recording
