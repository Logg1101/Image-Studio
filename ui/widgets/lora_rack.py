from pathlib import Path
from typing import Dict, List, Any, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QPushButton, QFrame, QCheckBox, QSlider, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from ui.widgets.lora_selector_dialog import LoRASelectorPopup

class LoRACardWidget(QFrame):
    """
    Individual LoRA card in the modern stack.
    Displays cleaned name, full filename tooltip, enable toggle,
    synchronized slider + spinner weight controls, and remove action.
    """
    removed = Signal(object)
    changed = Signal()

    def __init__(self, display_name: str, file_path: Path, initial_weight: float = 0.80, parent=None):
        super().__init__(parent)
        self.display_name = display_name
        self.file_path = file_path
        self.setObjectName("cardPanel")
        self.setStyleSheet(
            "QFrame#cardPanel { background-color: #151A24; border: 1px solid #252C3A; border-radius: 9px; padding: 4px 6px; } "
            "QFrame#cardPanel:hover { border-color: #353F54; }"
        )
        self.setToolTip(f"File: {file_path.name}\nPath: {file_path.resolve()}")

        main_lay = QVBoxLayout(self)
        main_lay.setContentsMargins(6, 5, 6, 5)
        main_lay.setSpacing(4)

        # Top Row: Grip + Checkbox/Name + Weight Spinner + Remove
        top_row = QHBoxLayout()
        top_row.setSpacing(6)

        grip = QLabel("⋮⋮")
        grip.setStyleSheet("color: #4B5563; font-weight: bold; font-size: 11px; background: transparent; border: none;")
        top_row.addWidget(grip)

        self.enable_chk = QCheckBox()
        self.enable_chk.setChecked(True)
        self.enable_chk.setToolTip("Enable/Disable this LoRA")
        top_row.addWidget(self.enable_chk)

        name_lbl = QLabel(f"🧩 {self.display_name}")
        name_lbl.setStyleSheet("color: #E8ECF4; font-weight: 700; font-size: 11px; background: transparent; border: none;")
        top_row.addWidget(name_lbl, stretch=1)

        self.spin = QDoubleSpinBox()
        self.spin.setRange(-2.0, 2.0)
        self.spin.setSingleStep(0.05)
        self.spin.setValue(initial_weight)
        self.spin.setFixedWidth(58)
        self.spin.setToolTip("LoRA Weight / Multiplier (-2.0 to 2.0)")
        top_row.addWidget(self.spin)

        self.del_btn = QPushButton("✕")
        self.del_btn.setFixedSize(20, 20)
        self.del_btn.setCursor(Qt.PointingHandCursor)
        self.del_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; color: #8993A7; font-size: 11px; font-weight: bold; } "
            "QPushButton:hover { color: #FF5C6C; }"
        )
        top_row.addWidget(self.del_btn)
        main_lay.addLayout(top_row)

        # Bottom Row: Synchronized Mini Slider
        slider_row = QHBoxLayout()
        slider_row.setSpacing(4)
        slider_row.setContentsMargins(18, 0, 4, 0)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(-200, 200)
        self.slider.setValue(int(initial_weight * 100))
        self.slider.setFixedHeight(12)
        slider_row.addWidget(self.slider)
        main_lay.addLayout(slider_row)

        # Sync events
        self.spin.valueChanged.connect(self._on_spin_changed)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.enable_chk.toggled.connect(self._on_enable_toggled)
        self.del_btn.clicked.connect(lambda: self.removed.emit(self))

    def _on_spin_changed(self, val: float):
        self.slider.blockSignals(True)
        self.slider.setValue(int(val * 100))
        self.slider.blockSignals(False)
        self.changed.emit()

    def _on_slider_changed(self, val: int):
        self.spin.blockSignals(True)
        self.spin.setValue(val / 100.0)
        self.spin.blockSignals(False)
        self.changed.emit()

    def _on_enable_toggled(self, checked: bool):
        self.spin.setEnabled(checked)
        self.slider.setEnabled(checked)
        self.changed.emit()

    def is_active(self) -> bool:
        return self.enable_chk.isChecked()

    def get_weight(self) -> float:
        return float(self.spin.value())

    def get_path_str(self) -> str:
        return str(self.file_path.resolve())


class LoRARackWidget(QWidget):
    """
    Modern LoRA rack supporting searchable selection popups,
    card stack visualization, and dynamic weighting.
    """
    loras_changed = Signal()

    def __init__(self, available_loras: Dict[str, Path], parent=None):
        super().__init__(parent)
        self.available_loras = available_loras
        self.cards: List[LoRACardWidget] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Header Row
        hdr_row = QHBoxLayout()
        hdr = QLabel("🎨 LORA RACK / ADAPTERS")
        hdr.setStyleSheet("color: #8993A7; font-weight: 800; font-size: 11px; letter-spacing: 0.5px;")
        hdr_row.addWidget(hdr)
        hdr_row.addStretch()

        self.add_btn = QPushButton("+ Add LoRA")
        self.add_btn.setStyleSheet(
            "QPushButton { background: #151A24; border: 1px solid #252C3A; border-radius: 6px; color: #4F9CFF; font-weight: 700; font-size: 11px; padding: 4px 10px; }"
            "QPushButton:hover { background: #1D2433; border-color: #7C6CFF; color: #FFFFFF; }"
        )
        self.add_btn.clicked.connect(self._open_selector)
        hdr_row.addWidget(self.add_btn)
        layout.addLayout(hdr_row)

        # Cards Container
        self.stack_layout = QVBoxLayout()
        self.stack_layout.setSpacing(4)
        layout.addLayout(self.stack_layout)

        # Empty State Label
        self.empty_lbl = QLabel("No active LoRAs. Click '+ Add LoRA' to attach.")
        self.empty_lbl.setStyleSheet("color: #555E70; font-size: 10px; padding: 6px 2px; font-style: italic;")
        layout.addWidget(self.empty_lbl)

        self._update_empty_state()

    def set_available_loras(self, available_loras: Dict[str, Path]):
        self.available_loras = available_loras

    def _open_selector(self):
        popup = LoRASelectorPopup(self.available_loras, parent=self)
        popup.lora_selected.connect(self.add_lora)
        popup.show_at(self.add_btn)

    def add_lora(self, display_name: str, file_path: Path, weight: float = 0.80):
        # Check if already added
        path_str = str(file_path.resolve())
        for c in self.cards:
            if c.get_path_str() == path_str:
                c.spin.setValue(weight)
                c.enable_chk.setChecked(True)
                return

        card = LoRACardWidget(display_name, file_path, initial_weight=weight)
        card.removed.connect(self._remove_card)
        card.changed.connect(self.loras_changed.emit)

        self.cards.append(card)
        self.stack_layout.addWidget(card)
        self._update_empty_state()
        self.loras_changed.emit()

    def _remove_card(self, card: LoRACardWidget):
        if card in self.cards:
            self.cards.remove(card)
            self.stack_layout.removeWidget(card)
            card.deleteLater()
            self._update_empty_state()
            self.loras_changed.emit()

    def _update_empty_state(self):
        self.empty_lbl.setVisible(len(self.cards) == 0)

    def get_active_loras(self) -> Dict[str, float]:
        """Returns {file_path_str: weight} for all enabled LoRAs."""
        active = {}
        for c in self.cards:
            if c.is_active():
                active[c.get_path_str()] = c.get_weight()
        return active

    def clear(self):
        for c in list(self.cards):
            self._remove_card(c)
