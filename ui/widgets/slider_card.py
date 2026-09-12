from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton
)
from PySide6.QtCore import Qt, Signal

class SliderCardWidget(QWidget):
    """
    Sleek parameter slider component matching the reference screenshot.
    Top row: Purple pill badge title on left, Value box + reset button on right.
    Bottom row: Min label, purple track slider, Max label.
    """
    valueChanged = Signal(int)

    def __init__(
        self,
        title: str,
        min_val: int,
        max_val: int,
        default_val: int,
        step: int = 1,
        parent=None
    ):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.default_val = default_val

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(3)

        # Top Row: Title Badge (Left) + Value Box & Reset (Right)
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(6)

        self.title_badge = QLabel(title)
        self.title_badge.setStyleSheet(
            "background: #151A24; border: 1px solid #252C3A; color: #8993A7; "
            "border-radius: 4px; padding: 2px 6px; font-weight: 700; font-size: 10px; text-transform: uppercase;"
        )
        top_row.addWidget(self.title_badge)
        top_row.addStretch()

        self.val_label = QLabel(str(default_val))
        self.val_label.setFixedWidth(50)
        self.val_label.setAlignment(Qt.AlignCenter)
        self.val_label.setStyleSheet(
            "background: #151A24; border: 1px solid #252C3A; border-radius: 5px;"
            "color: #FFFFFF; font-weight: bold; font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 2px 4px;"
        )
        top_row.addWidget(self.val_label)

        self.reset_btn = QPushButton("↺")
        self.reset_btn.setFixedSize(22, 22)
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; color: #8993A7; font-size: 13px; }"
            "QPushButton:hover { color: #7C6CFF; }"
        )
        top_row.addWidget(self.reset_btn)
        layout.addLayout(top_row)

        # Bottom Row: Min label, Slider, Max label
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(6)

        min_lbl = QLabel(str(min_val))
        min_lbl.setStyleSheet("color: #8993A7; font-size: 10px; font-family: 'JetBrains Mono', monospace;")
        bottom_row.addWidget(min_lbl)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(min_val, max_val)
        self.slider.setSingleStep(step)
        self.slider.setValue(default_val)
        self.slider.setStyleSheet(
            "QSlider::groove:horizontal { height: 4px; background: #151A24; border-radius: 2px; border: 1px solid #252C3A; }"
            "QSlider::sub-page:horizontal { background: #7C6CFF; border-radius: 2px; }"
            "QSlider::handle:horizontal { width: 14px; height: 14px; margin: -5px 0; background: #FFFFFF; border: 2px solid #7C6CFF; border-radius: 7px; }"
            "QSlider::handle:horizontal:hover { background: #FFFFFF; border-color: #8E80FF; }"
        )
        bottom_row.addWidget(self.slider, stretch=1)

        max_lbl = QLabel(str(max_val))
        max_lbl.setStyleSheet("color: #8993A7; font-size: 10px; font-family: 'JetBrains Mono', monospace;")
        bottom_row.addWidget(max_lbl)

        layout.addLayout(bottom_row)

        # Bind events
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.reset_btn.clicked.connect(self.reset)

    def _on_slider_changed(self, val: int):
        self.val_label.setText(str(val))
        self.valueChanged.emit(val)

    def value(self) -> int:
        return self.slider.value()

    def getValue(self) -> int:
        return self.slider.value()

    def setValue(self, val: int):
        self.slider.setValue(val)
        self.val_label.setText(str(val))

    def reset(self):
        self.setValue(self.default_val)
