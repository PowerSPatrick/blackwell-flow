"""
UI Manager for Blackwell-Flow.

Provides system tray icon and recording HUD overlay.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

import structlog
from PyQt6.QtCore import QPoint, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QCursor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMenu,
    QSystemTrayIcon,
    QWidget,
)

if TYPE_CHECKING:
    from .config import StyleType, UIConfig

logger = structlog.get_logger(__name__)


class RecordingIndicator(QWidget):
    """A small recording indicator that follows the cursor or stays at top."""

    def __init__(
        self,
        position: str = "top",
        color: str = "#FF4444",
        size: int = 12,
        opacity: float = 0.85,
    ) -> None:
        super().__init__()

        self.position = position
        self.indicator_color = QColor(color)
        self.indicator_size = size

        # Window setup
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self.setWindowOpacity(opacity)

        # Size based on indicator
        self.setFixedSize(size + 10, size + 10)

        # Timer for cursor following (if cursor mode)
        self._follow_timer = QTimer(self)
        self._follow_timer.timeout.connect(self._update_position)

        # Pulse animation
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse)
        self._pulse_state = 0

    def paintEvent(self, event) -> None:
        """Paint the recording indicator."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate pulse effect
        pulse_factor = 0.2 * (1 + self._pulse_state / 10)
        adjusted_size = int(self.indicator_size * (1 - pulse_factor * 0.2))

        # Draw outer glow
        glow_color = QColor(self.indicator_color)
        glow_color.setAlpha(100)
        painter.setBrush(glow_color)
        painter.setPen(Qt.PenStyle.NoPen)

        center = self.rect().center()
        painter.drawEllipse(center, adjusted_size + 2, adjusted_size + 2)

        # Draw main indicator
        painter.setBrush(self.indicator_color)
        painter.drawEllipse(center, adjusted_size // 2, adjusted_size // 2)

    def _pulse(self) -> None:
        """Animate the pulse effect."""
        self._pulse_state = (self._pulse_state + 1) % 20
        self.update()

    def _update_position(self) -> None:
        """Update position to follow cursor."""
        if self.position == "cursor":
            cursor_pos = QCursor.pos()
            self.move(cursor_pos.x() + 20, cursor_pos.y() + 20)
        else:
            # Top center of screen
            screen = QApplication.primaryScreen()
            if screen:
                geometry = screen.geometry()
                self.move(geometry.width() // 2 - self.width() // 2, 10)

    def show_recording(self) -> None:
        """Show the recording indicator."""
        self._update_position()
        self.show()

        if self.position == "cursor":
            self._follow_timer.start(50)

        self._pulse_timer.start(50)
        logger.debug("Recording indicator shown")

    def hide_recording(self) -> None:
        """Hide the recording indicator."""
        self._follow_timer.stop()
        self._pulse_timer.stop()
        self.hide()
        logger.debug("Recording indicator hidden")


class StatusHUD(QWidget):
    """A translucent status HUD for showing messages."""

    def __init__(self, opacity: float = 0.85) -> None:
        super().__init__()

        # Window setup
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self.setWindowOpacity(opacity)
        self.setMinimumWidth(200)

        # Label for status text
        self._label = QLabel(self)
        self._label.setStyleSheet("""
            QLabel {
                background-color: rgba(30, 30, 30, 200);
                color: white;
                padding: 8px 16px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Auto-hide timer
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)

    def show_message(self, message: str, duration_ms: int = 2000) -> None:
        """Show a status message."""
        self._label.setText(message)
        self._label.adjustSize()
        self.adjustSize()

        # Position at top center
        screen = QApplication.primaryScreen()
        if screen:
            geometry = screen.geometry()
            self.move(geometry.width() // 2 - self.width() // 2, 50)

        self.show()

        if duration_ms > 0:
            self._hide_timer.start(duration_ms)


class SystemTrayManager:
    """Manages the system tray icon and menu."""

    style_changed = None  # Will be set up as signal
    quit_requested = None

    def __init__(
        self,
        config: UIConfig,
        on_style_change: Callable[[StyleType], None] | None = None,
        on_quit: Callable[[], None] | None = None,
    ) -> None:
        self.config = config
        self._on_style_change = on_style_change
        self._on_quit = on_quit

        self._app: QApplication | None = None
        self._tray_icon: QSystemTrayIcon | None = None
        self._recording_indicator: RecordingIndicator | None = None
        self._status_hud: StatusHUD | None = None
        self._current_style: StyleType = "professional"

    def initialize(self, app: QApplication) -> None:
        """Initialize the system tray and HUD."""
        self._app = app

        # Create tray icon
        self._tray_icon = QSystemTrayIcon(app)
        self._tray_icon.setIcon(self._create_icon())
        self._tray_icon.setToolTip("Blackwell-Flow - Ready")

        # Create context menu
        menu = QMenu()

        # Style submenu
        style_menu = menu.addMenu("Style")
        for style in ["professional", "casual", "creative", "bullet_points"]:
            action = style_menu.addAction(style.replace("_", " ").title())
            action.triggered.connect(lambda checked, s=style: self._set_style(s))

        menu.addSeparator()

        # Quit action
        quit_action = menu.addAction("Quit")
        quit_action.triggered.connect(self._quit)

        self._tray_icon.setContextMenu(menu)
        self._tray_icon.show()

        # Create recording indicator
        self._recording_indicator = RecordingIndicator(
            position=self.config.hud_position,
            color=self.config.indicator_color,
            size=self.config.indicator_size,
            opacity=self.config.hud_opacity,
        )

        # Create status HUD
        self._status_hud = StatusHUD(opacity=self.config.hud_opacity)

        logger.info("System tray initialized")

    def _create_icon(self, recording: bool = False) -> QIcon:
        """Create the tray icon."""
        size = 64
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw microphone icon
        if recording:
            color = QColor("#FF4444")
        else:
            color = QColor("#4CAF50")

        painter.setBrush(color)
        painter.setPen(QPen(color, 2))

        # Simple microphone shape
        center = size // 2
        painter.drawEllipse(center - 10, 10, 20, 30)
        painter.drawRoundedRect(center - 15, 35, 30, 10, 3, 3)
        painter.drawLine(center, 45, center, 55)
        painter.drawLine(center - 10, 55, center + 10, 55)

        painter.end()

        return QIcon(pixmap)

    def _set_style(self, style: StyleType) -> None:
        """Set the current style."""
        self._current_style = style
        if self._on_style_change:
            self._on_style_change(style)
        self.show_status(f"Style: {style.replace('_', ' ').title()}")

    def _quit(self) -> None:
        """Handle quit request."""
        if self._on_quit:
            self._on_quit()
        if self._app:
            self._app.quit()

    def set_recording(self, recording: bool) -> None:
        """Update the UI to reflect recording state."""
        if self._tray_icon:
            self._tray_icon.setIcon(self._create_icon(recording))
            self._tray_icon.setToolTip(
                "Blackwell-Flow - Recording..." if recording else "Blackwell-Flow - Ready"
            )

        if self._recording_indicator and self.config.show_hud:
            if recording:
                self._recording_indicator.show_recording()
            else:
                self._recording_indicator.hide_recording()

    def show_status(self, message: str, duration_ms: int = 2000) -> None:
        """Show a status message in the HUD."""
        if self._status_hud and self.config.show_hud:
            self._status_hud.show_message(message, duration_ms)

    def show_notification(self, title: str, message: str) -> None:
        """Show a system notification."""
        if self._tray_icon:
            self._tray_icon.showMessage(title, message, QSystemTrayIcon.MessageIcon.Information)

    @property
    def current_style(self) -> StyleType:
        """Get the current style."""
        return self._current_style

    def set_current_style(self, style: StyleType) -> None:
        """Set the current style without triggering callback."""
        self._current_style = style


class UIManager:
    """Main UI manager coordinating all UI components."""

    def __init__(self, config: UIConfig) -> None:
        self.config = config
        self._app: QApplication | None = None
        self._tray_manager: SystemTrayManager | None = None
        self._style_callback: Callable[[StyleType], None] | None = None
        self._quit_callback: Callable[[], None] | None = None

    def set_callbacks(
        self,
        on_style_change: Callable[[StyleType], None] | None = None,
        on_quit: Callable[[], None] | None = None,
    ) -> None:
        """Set UI callbacks."""
        self._style_callback = on_style_change
        self._quit_callback = on_quit

    def initialize(self) -> QApplication:
        """Initialize the Qt application and UI components."""
        # Create or get existing QApplication
        self._app = QApplication.instance()
        if self._app is None:
            self._app = QApplication([])

        self._app.setQuitOnLastWindowClosed(False)

        # Initialize system tray
        self._tray_manager = SystemTrayManager(
            self.config,
            on_style_change=self._style_callback,
            on_quit=self._quit_callback,
        )
        self._tray_manager.initialize(self._app)

        logger.info("UI Manager initialized")
        return self._app

    def set_recording(self, recording: bool) -> None:
        """Update recording state in UI."""
        if self._tray_manager:
            self._tray_manager.set_recording(recording)

    def show_status(self, message: str, duration_ms: int = 2000) -> None:
        """Show a status message."""
        if self._tray_manager:
            self._tray_manager.show_status(message, duration_ms)

    def show_notification(self, title: str, message: str) -> None:
        """Show a system notification."""
        if self._tray_manager:
            self._tray_manager.show_notification(title, message)

    @property
    def current_style(self) -> StyleType:
        """Get the current style."""
        if self._tray_manager:
            return self._tray_manager.current_style
        return "professional"

    def set_style(self, style: StyleType) -> None:
        """Set the current style."""
        if self._tray_manager:
            self._tray_manager.set_current_style(style)

    def run(self) -> int:
        """Run the Qt event loop."""
        if self._app:
            return self._app.exec()
        return 1

    def quit(self) -> None:
        """Quit the application."""
        if self._app:
            self._app.quit()
