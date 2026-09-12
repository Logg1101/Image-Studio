from pathlib import Path
from typing import Dict, Optional
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QListWidget,
    QListWidgetItem, QLabel, QPushButton, QFrame, QWidget, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, Signal, QSize, QPoint
from PySide6.QtGui import QColor, QFont, QKeyEvent

class LoRASelectorPopup(QDialog):
    """
    Wide, floating searchable selector popover unconstrained by narrow sidebars.
    Supports instant filtering, full filename visibility, keyboard navigation,
    and clean selection.
    """
    lora_selected = Signal(str, Path)  # (display_name, file_path)

    def __init__(self, available_loras: Dict[str, Path], parent=None):
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.available_loras = available_loras
        self.setFixedWidth(460)
        self.setFixedHeight(380)

        # Outer Container with Border & Shadow
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(6, 6, 6, 6)

        container = QFrame()
        container.setStyleSheet(
            "background-color: #0F131B; border: 1px solid #252C3A; border-radius: 12px;"
        )
        lay = QVBoxLayout(container)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # Header Title
        hdr_row = QHBoxLayout()
        title_lbl = QLabel("🧩 Select LoRA Adapter")
        title_lbl.setStyleSheet("font-weight: 800; font-size: 12px; color: #E8ECF4; border: none; background: transparent;")
        hdr_row.addWidget(title_lbl)
        hdr_row.addStretch()

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(20, 20)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("background: transparent; border: none; color: #8993A7; font-weight: bold; font-size: 11px;")
        close_btn.clicked.connect(self.reject)
        hdr_row.addWidget(close_btn)
        lay.addLayout(hdr_row)

        # Search Bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Type to search LoRAs...")
        self.search_input.setStyleSheet(
            "background-color: #151A24; border: 1px solid #252C3A; border-radius: 8px; "
            "padding: 8px 12px; color: #E8ECF4; font-size: 12px;"
        )
        lay.addWidget(self.search_input)

        # Results List
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            "QListWidget { background-color: #151A24; border: 1px solid #252C3A; border-radius: 8px; padding: 4px; color: #E8ECF4; } "
            "QListWidget::item { border-radius: 6px; padding: 6px 8px; margin-bottom: 2px; } "
            "QListWidget::item:selected { background-color: #1A2234; border: 1px solid #7C6CFF; color: #FFFFFF; } "
            "QListWidget::item:hover:!selected { background-color: #1D2433; }"
        )
        lay.addWidget(self.list_widget)

        # Footer Hint
        hint_lbl = QLabel("↑↓ Navigate  •  Enter Select  •  Esc Close")
        hint_lbl.setStyleSheet("color: #8993A7; font-size: 10px; font-family: 'JetBrains Mono', monospace; border: none; background: transparent;")
        hint_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(hint_lbl)

        outer_layout.addWidget(container)

        # Drop shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(0, 6)
        container.setGraphicsEffect(shadow)

        # Signals
        self.search_input.textChanged.connect(self._filter_list)
        self.list_widget.itemDoubleClicked.connect(self._on_item_chosen)
        self.search_input.returnPressed.connect(self._on_enter_pressed)

        self._populate()

    def _clean_name(self, raw_name: str) -> str:
        name = raw_name
        for ext in [".safetensors", ".pt", ".bin", ".ckpt"]:
            if name.lower().endswith(ext):
                name = name[:-len(ext)]
                break
        return name

    def _populate(self):
        self.list_widget.clear()
        query = self.search_input.text().strip().lower()

        for raw_name, path in sorted(self.available_loras.items(), key=lambda x: x[0].lower()):
            clean = self._clean_name(raw_name)
            filename = path.name

            if query and (query not in clean.lower() and query not in filename.lower()):
                continue

            item = QListWidgetItem()
            # Custom widget for rich item display
            w = QWidget()
            w_lay = QVBoxLayout(w)
            w_lay.setContentsMargins(2, 2, 2, 2)
            w_lay.setSpacing(1)

            top_line = QHBoxLayout()
            top_line.setSpacing(6)
            icon_lbl = QLabel("🧩")
            icon_lbl.setStyleSheet("font-size: 12px; background: transparent; border: none;")
            name_lbl = QLabel(clean)
            name_lbl.setStyleSheet("font-weight: 700; font-size: 12px; color: #E8ECF4; background: transparent; border: none;")
            top_line.addWidget(icon_lbl)
            top_line.addWidget(name_lbl)
            top_line.addStretch()
            w_lay.addLayout(top_line)

            fn_lbl = QLabel(filename)
            fn_lbl.setStyleSheet("font-size: 10px; color: #8993A7; font-family: 'JetBrains Mono', monospace; background: transparent; border: none;")
            w_lay.addWidget(fn_lbl)

            item.setSizeHint(QSize(400, 44))
            item.setData(Qt.UserRole, (clean, path))
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, w)

        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    def _filter_list(self):
        self._populate()

    def _on_enter_pressed(self):
        cur = self.list_widget.currentItem()
        if cur:
            self._on_item_chosen(cur)

    def _on_item_chosen(self, item: QListWidgetItem):
        data = item.data(Qt.UserRole)
        if data:
            clean_name, path = data
            self.lora_selected.emit(clean_name, path)
            self.accept()

    def show_at(self, target_widget: QWidget):
        # Calculate optimal popover position
        geo = target_widget.rect()
        global_pos = target_widget.mapToGlobal(geo.bottomLeft())
        self.move(global_pos.x(), global_pos.y() + 4)
        self.search_input.setText("")
        self.search_input.setFocus()
        self.exec()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Down:
            idx = self.list_widget.currentRow()
            if idx < self.list_widget.count() - 1:
                self.list_widget.setCurrentRow(idx + 1)
            event.accept()
        elif event.key() == Qt.Key_Up:
            idx = self.list_widget.currentRow()
            if idx > 0:
                self.list_widget.setCurrentRow(idx - 1)
            event.accept()
        elif event.key() == Qt.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(event)
