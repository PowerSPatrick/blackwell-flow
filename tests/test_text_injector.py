"""
Tests for text injection.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from blackwell_flow.text_injector import TextInjector


class TestTextInjector:
    """Tests for TextInjector."""

    def test_init(self) -> None:
        """Test injector initialization."""
        injector = TextInjector(paste_delay=0.1)

        assert injector.paste_delay == 0.1
        assert injector._original_clipboard is None

    @patch("blackwell_flow.text_injector.pyperclip")
    def test_save_clipboard(self, mock_pyperclip: MagicMock) -> None:
        """Test saving clipboard contents."""
        mock_pyperclip.paste.return_value = "original content"

        injector = TextInjector()
        injector._save_clipboard()

        assert injector._original_clipboard == "original content"
        mock_pyperclip.paste.assert_called_once()

    @patch("blackwell_flow.text_injector.time.sleep")
    @patch("blackwell_flow.text_injector.pyperclip")
    def test_restore_clipboard(
        self,
        mock_pyperclip: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """Test restoring clipboard contents."""
        injector = TextInjector()
        injector._original_clipboard = "original content"

        injector._restore_clipboard()

        mock_pyperclip.copy.assert_called_once_with("original content")
        assert injector._original_clipboard is None

    @patch("blackwell_flow.text_injector.time.sleep")
    @patch("blackwell_flow.text_injector.pyautogui")
    @patch("blackwell_flow.text_injector.pyperclip")
    def test_inject(
        self,
        mock_pyperclip: MagicMock,
        mock_pyautogui: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """Test text injection."""
        mock_pyperclip.paste.return_value = "original"

        injector = TextInjector(paste_delay=0.01)
        injector.inject("test text", restore_clipboard=True)

        # Verify clipboard operations
        mock_pyperclip.paste.assert_called_once()  # Save original
        mock_pyperclip.copy.assert_any_call("test text")  # Set new text
        mock_pyperclip.copy.assert_any_call("original")  # Restore original

        # Verify paste
        mock_pyautogui.hotkey.assert_called_once_with("ctrl", "v")

    @patch("blackwell_flow.text_injector.time.sleep")
    @patch("blackwell_flow.text_injector.pyautogui")
    @patch("blackwell_flow.text_injector.pyperclip")
    def test_inject_without_restore(
        self,
        mock_pyperclip: MagicMock,
        mock_pyautogui: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """Test injection without restoring clipboard."""
        injector = TextInjector()
        injector.inject("test text", restore_clipboard=False)

        # Should only copy once (the new text)
        mock_pyperclip.copy.assert_called_once_with("test text")
        mock_pyperclip.paste.assert_not_called()

    @patch("blackwell_flow.text_injector.pyautogui")
    @patch("blackwell_flow.text_injector.pyperclip")
    def test_inject_empty_text(
        self,
        mock_pyperclip: MagicMock,
        mock_pyautogui: MagicMock,
    ) -> None:
        """Test injection with empty text does nothing."""
        injector = TextInjector()
        injector.inject("")

        mock_pyperclip.copy.assert_not_called()
        mock_pyautogui.hotkey.assert_not_called()

    @patch("blackwell_flow.text_injector.pyautogui")
    def test_inject_raw(self, mock_pyautogui: MagicMock) -> None:
        """Test raw text injection by typing."""
        injector = TextInjector()
        injector.inject_raw("test")

        mock_pyautogui.typewrite.assert_called_once_with("test", interval=0.01)

    @patch("blackwell_flow.text_injector.pyautogui")
    def test_inject_raw_empty_text(self, mock_pyautogui: MagicMock) -> None:
        """Test raw injection with empty text does nothing."""
        injector = TextInjector()
        injector.inject_raw("")

        mock_pyautogui.typewrite.assert_not_called()
