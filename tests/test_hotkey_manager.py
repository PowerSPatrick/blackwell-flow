"""
Tests for hotkey manager.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pynput.keyboard import Key, KeyCode

from blackwell_flow.config import HotkeyConfig
from blackwell_flow.hotkey_manager import HotkeyManager, parse_hotkey


class TestParseHotkey:
    """Tests for hotkey parsing."""

    def test_parse_ctrl_space(self) -> None:
        """Test parsing ctrl+space."""
        keys = parse_hotkey("ctrl+space")

        assert Key.ctrl_l in keys
        assert Key.space in keys
        assert len(keys) == 2

    def test_parse_ctrl_shift_s(self) -> None:
        """Test parsing ctrl+shift+s."""
        keys = parse_hotkey("ctrl+shift+s")

        assert Key.ctrl_l in keys
        assert Key.shift_l in keys
        assert KeyCode.from_char("s") in keys
        assert len(keys) == 3

    def test_parse_caps_lock(self) -> None:
        """Test parsing caps_lock."""
        keys = parse_hotkey("caps_lock")

        assert Key.caps_lock in keys
        assert len(keys) == 1

    def test_parse_with_spaces(self) -> None:
        """Test parsing with spaces instead of plus."""
        keys = parse_hotkey("ctrl space")

        assert Key.ctrl_l in keys
        assert Key.space in keys

    def test_parse_alt_key(self) -> None:
        """Test parsing alt key."""
        keys = parse_hotkey("alt+a")

        assert Key.alt_l in keys
        assert KeyCode.from_char("a") in keys

    def test_parse_unknown_key(self) -> None:
        """Test parsing unknown key logs warning."""
        # Should not raise, just skip unknown key
        keys = parse_hotkey("ctrl+unknownkey")

        assert Key.ctrl_l in keys


class TestHotkeyManager:
    """Tests for HotkeyManager."""

    def test_init(self, mock_hotkey_config: HotkeyConfig) -> None:
        """Test manager initialization."""
        manager = HotkeyManager(mock_hotkey_config)

        assert manager.config == mock_hotkey_config
        assert manager._recording is False
        assert manager._listener is None

    def test_set_callbacks(self, mock_hotkey_config: HotkeyConfig) -> None:
        """Test setting callbacks."""
        manager = HotkeyManager(mock_hotkey_config)

        start_callback = MagicMock()
        stop_callback = MagicMock()
        cycle_callback = MagicMock()
        quit_callback = MagicMock()

        manager.set_callbacks(
            on_record_start=start_callback,
            on_record_stop=stop_callback,
            on_cycle_style=cycle_callback,
            on_quit=quit_callback,
        )

        assert manager._on_record_start == start_callback
        assert manager._on_record_stop == stop_callback
        assert manager._on_cycle_style == cycle_callback
        assert manager._on_quit == quit_callback

    def test_normalize_key(self, mock_hotkey_config: HotkeyConfig) -> None:
        """Test key normalization."""
        manager = HotkeyManager(mock_hotkey_config)

        # Left/right modifiers should normalize to left
        assert manager._normalize_key(Key.ctrl_r) == Key.ctrl_l
        assert manager._normalize_key(Key.alt_r) == Key.alt_l
        assert manager._normalize_key(Key.shift_r) == Key.shift_l

        # Regular keys should pass through
        assert manager._normalize_key(Key.space) == Key.space
        assert manager._normalize_key(KeyCode.from_char("a")) == KeyCode.from_char("a")

        # None should return None
        assert manager._normalize_key(None) is None

    def test_on_press_record_start(self, mock_hotkey_config: HotkeyConfig) -> None:
        """Test record start callback on hotkey press."""
        manager = HotkeyManager(mock_hotkey_config)

        start_callback = MagicMock()
        manager.set_callbacks(on_record_start=start_callback)

        # Simulate pressing ctrl+space
        manager._on_press(Key.ctrl_l)
        start_callback.assert_not_called()  # Not all keys pressed yet

        manager._on_press(Key.space)
        start_callback.assert_called_once()  # Now should trigger

    def test_on_release_record_stop(self, mock_hotkey_config: HotkeyConfig) -> None:
        """Test record stop callback on hotkey release."""
        manager = HotkeyManager(mock_hotkey_config)

        stop_callback = MagicMock()
        manager.set_callbacks(on_record_stop=stop_callback)

        # Start recording first
        manager._on_press(Key.ctrl_l)
        manager._on_press(Key.space)

        # Release space
        manager._on_release(Key.space)
        stop_callback.assert_called_once()

    def test_on_press_cycle_style(self, mock_hotkey_config: HotkeyConfig) -> None:
        """Test cycle style callback."""
        manager = HotkeyManager(mock_hotkey_config)

        cycle_callback = MagicMock()
        manager.set_callbacks(on_cycle_style=cycle_callback)

        # Simulate pressing ctrl+shift+s
        manager._on_press(Key.ctrl_l)
        manager._on_press(Key.shift_l)
        manager._on_press(KeyCode.from_char("s"))

        cycle_callback.assert_called_once()

    def test_on_press_quit(self, mock_hotkey_config: HotkeyConfig) -> None:
        """Test quit callback."""
        manager = HotkeyManager(mock_hotkey_config)

        quit_callback = MagicMock()
        manager.set_callbacks(on_quit=quit_callback)

        # Simulate pressing ctrl+shift+q
        manager._on_press(Key.ctrl_l)
        manager._on_press(Key.shift_l)
        manager._on_press(KeyCode.from_char("q"))

        quit_callback.assert_called_once()

    def test_is_recording_property(self, mock_hotkey_config: HotkeyConfig) -> None:
        """Test is_recording property."""
        manager = HotkeyManager(mock_hotkey_config)

        assert manager.is_recording is False

        # Simulate starting recording
        manager._on_press(Key.ctrl_l)
        manager._on_press(Key.space)

        assert manager.is_recording is True

        # Simulate stopping recording
        manager._on_release(Key.space)

        assert manager.is_recording is False

    @patch("blackwell_flow.hotkey_manager.keyboard.Listener")
    def test_start(
        self,
        mock_listener_class: MagicMock,
        mock_hotkey_config: HotkeyConfig,
    ) -> None:
        """Test starting the listener."""
        mock_listener = MagicMock()
        mock_listener_class.return_value = mock_listener

        manager = HotkeyManager(mock_hotkey_config)
        manager.start()

        mock_listener_class.assert_called_once()
        mock_listener.start.assert_called_once()

    @patch("blackwell_flow.hotkey_manager.keyboard.Listener")
    def test_stop(
        self,
        mock_listener_class: MagicMock,
        mock_hotkey_config: HotkeyConfig,
    ) -> None:
        """Test stopping the listener."""
        mock_listener = MagicMock()
        mock_listener_class.return_value = mock_listener

        manager = HotkeyManager(mock_hotkey_config)
        manager.start()
        manager.stop()

        mock_listener.stop.assert_called_once()
        assert manager._listener is None

    def test_double_start(self, mock_hotkey_config: HotkeyConfig) -> None:
        """Test that double start is a no-op."""
        with patch("blackwell_flow.hotkey_manager.keyboard.Listener") as mock_listener_class:
            mock_listener = MagicMock()
            mock_listener_class.return_value = mock_listener

            manager = HotkeyManager(mock_hotkey_config)
            manager.start()
            manager.start()  # Second start should be ignored

            # Listener should only be created once
            assert mock_listener_class.call_count == 1
