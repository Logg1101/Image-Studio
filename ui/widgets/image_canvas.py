from typing import Optional
from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QSizePolicy
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap

class ImageCanvasWidget(QFrame):
    """High-resolution image viewport with save, remix, and send actions."""
    image_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("cardPanel")
        self.setMinimumSize(360, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.image_label = QLabel("Output Canvas (Ready)")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: #090C12; border: 1px dashed #252C3A; border-radius: 10px; color: #8993A7; font-size: 13px; font-weight: 500;")
        layout.addWidget(self.image_label)

        self.current_image_path: Optional[str] = None

    def set_image(self, image_path: str):
        self.current_image_path = image_path
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.image_label.setStyleSheet("background: #090C12; border: 1px solid #252C3A; border-radius: 10px;")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_image_path:
            self.set_image(self.current_image_path)
