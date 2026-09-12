from typing import Optional
from pathlib import Path
from PIL import Image

from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QDoubleSpinBox, QPushButton, QFileDialog
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap

class PuLIDPanelWidget(QGroupBox):
    """
    Dedicated PuLID Face Identification control widget.
    Provides face image loading, toggle, and identity strength weighting.
    """
    def __init__(self, parent=None):
        super().__init__("PULID FACE IDENTIFIER", parent)
        self.face_image_path: Optional[str] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(6)

        # Enable Checkbox & Strength
        top_row = QHBoxLayout()
        self.enable_check = QCheckBox("Enable PuLID Face ID")
        self.enable_check.setChecked(False)

        self.strength_spin = QDoubleSpinBox()
        self.strength_spin.setRange(0.0, 1.5)
        self.strength_spin.setSingleStep(0.05)
        self.strength_spin.setValue(0.80)
        self.strength_spin.setFixedWidth(70)

        top_row.addWidget(self.enable_check)
        top_row.addStretch()
        top_row.addWidget(QLabel("Strength:"))
        top_row.addWidget(self.strength_spin)
        layout.addLayout(top_row)

        # Face Image Preview Box
        self.face_preview = QLabel("Click to Select Face Reference Image")
        self.face_preview.setAlignment(Qt.AlignCenter)
        self.face_preview.setFixedHeight(90)
        self.face_preview.setStyleSheet("background: #070D19; border: 1px dashed #1E293B; border-radius: 6px; color: #64748B; font-size: 11px;")
        layout.addWidget(self.face_preview)

        btn_row = QHBoxLayout()
        self.load_btn = QPushButton("Select Face Image")
        self.clear_btn = QPushButton("Clear")
        btn_row.addWidget(self.load_btn)
        btn_row.addWidget(self.clear_btn)
        layout.addLayout(btn_row)

        self.load_btn.clicked.connect(self._select_image)
        self.face_preview.mousePressEvent = lambda e: self._select_image()
        self.clear_btn.clicked.connect(self.clear)

    def _select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Face Reference Image", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            self.set_face_image(path)

    def set_face_image(self, path: str):
        self.face_image_path = path
        pix = QPixmap(path)
        if not pix.isNull():
            self.face_preview.setPixmap(pix.scaled(self.face_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.face_preview.setStyleSheet("background: #070D19; border: 1px solid #38BDF8; border-radius: 6px;")
            self.enable_check.setChecked(True)

    def clear(self):
        self.face_image_path = None
        self.face_preview.clear()
        self.face_preview.setText("Click to Select Face Reference Image")
        self.face_preview.setStyleSheet("background: #070D19; border: 1px dashed #1E293B; border-radius: 6px; color: #64748B; font-size: 11px;")
        self.enable_check.setChecked(False)

    def is_enabled(self) -> bool:
        return self.enable_check.isChecked() and self.face_image_path is not None

    def get_strength(self) -> float:
        return float(self.strength_spin.value())

    def get_pil_image(self) -> Optional[Image.Image]:
        if self.is_enabled() and self.face_image_path:
            return Image.open(self.face_image_path).convert("RGB")
        return None
