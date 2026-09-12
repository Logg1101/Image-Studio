import os
import json
import datetime
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
from PIL import Image

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QLineEdit,
    QPushButton, QComboBox, QGroupBox, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QFrame, QScrollArea, QSizePolicy, QGridLayout
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QPixmap, QIcon, QFont

import config.paths as paths
from core.types import sanitize_seed

class StructuredInspectorWidget(QWidget):
    """
    Categorized Parameter Inspector:
    Empty state when nothing selected, and structured metadata sections
    (Image, Model, Prompts, Sampling, Adapters) when selected.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_lay = QVBoxLayout(self)
        self.main_lay.setContentsMargins(0, 0, 0, 0)
        self.main_lay.setSpacing(8)

        # Empty State
        self.empty_widget = QFrame()
        self.empty_widget.setObjectName("cardPanel")
        self.empty_widget.setStyleSheet(
            "background-color: #0F131B; border: 1px dashed #252C3A; border-radius: 12px; padding: 24px;"
        )
        empty_lay = QVBoxLayout(self.empty_widget)
        empty_lay.setAlignment(Qt.AlignCenter)
        empty_lay.setSpacing(8)

        e_icon = QLabel("📁")
        e_icon.setStyleSheet("font-size: 32px; background: transparent; border: none;")
        e_icon.setAlignment(Qt.AlignCenter)
        e_lbl = QLabel("Select an image to inspect generation metadata")
        e_lbl.setStyleSheet("color: #8993A7; font-size: 11px; font-weight: 600; background: transparent; border: none;")
        e_lbl.setAlignment(Qt.AlignCenter)
        empty_lay.addWidget(e_icon)
        empty_lay.addWidget(e_lbl)
        self.main_lay.addWidget(self.empty_widget)

        # Active Inspector Content (Scrollable)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_content = QWidget()
        self.inspect_lay = QVBoxLayout(self.scroll_content)
        self.inspect_lay.setContentsMargins(0, 0, 4, 0)
        self.inspect_lay.setSpacing(8)

        # Image Preview Card
        self.preview_lbl = QLabel()
        self.preview_lbl.setAlignment(Qt.AlignCenter)
        self.preview_lbl.setMinimumHeight(200)
        self.preview_lbl.setMaximumHeight(260)
        self.preview_lbl.setStyleSheet(
            "background: #151A24; border: 1px solid #252C3A; border-radius: 10px; padding: 4px;"
        )
        self.inspect_lay.addWidget(self.preview_lbl)

        # 1. Image Info Section
        self.img_box = self._create_section("📷 IMAGE DETAILS")
        self.img_info_lbl = QLabel()
        self.img_info_lbl.setStyleSheet("font-size: 11px; color: #E8ECF4; font-family: 'JetBrains Mono', monospace;")
        self.img_box.layout().addWidget(self.img_info_lbl)
        self.inspect_lay.addWidget(self.img_box)

        # 2. Model Info Section
        self.model_box = self._create_section("🧠 MODEL & CHECKPOINT")
        self.model_info_lbl = QLabel()
        self.model_info_lbl.setStyleSheet("font-size: 11px; color: #35D6C5; font-weight: bold;")
        self.model_box.layout().addWidget(self.model_info_lbl)
        self.inspect_lay.addWidget(self.model_box)

        # 3. Prompts Section
        self.prompt_box = self._create_section("✍️ PROMPT & CONDITIONING")
        self.prompt_val = QLabel()
        self.prompt_val.setWordWrap(True)
        self.prompt_val.setStyleSheet("font-size: 11px; color: #E8ECF4; padding: 2px;")
        self.neg_prompt_val = QLabel()
        self.neg_prompt_val.setWordWrap(True)
        self.neg_prompt_val.setStyleSheet("font-size: 11px; color: #8993A7; padding: 2px;")
        self.prompt_box.layout().addWidget(QLabel("Prompt:"))
        self.prompt_box.layout().addWidget(self.prompt_val)
        self.prompt_box.layout().addWidget(QLabel("Negative:"))
        self.prompt_box.layout().addWidget(self.neg_prompt_val)
        self.inspect_lay.addWidget(self.prompt_box)

        # 4. Sampling Section
        self.sampling_box = self._create_section("🎛️ SAMPLING PARAMETERS")
        self.sampling_grid = QGridLayout()
        self.sampling_grid.setSpacing(4)
        self.sampling_box.layout().addLayout(self.sampling_grid)
        self.inspect_lay.addWidget(self.sampling_box)

        # 5. Adapters Section
        self.adapter_box = self._create_section("🎨 ADAPTERS & LORAS")
        self.adapter_info_lbl = QLabel()
        self.adapter_info_lbl.setWordWrap(True)
        self.adapter_info_lbl.setStyleSheet("font-size: 11px; color: #E8ECF4; font-family: 'JetBrains Mono', monospace;")
        self.adapter_box.layout().addWidget(self.adapter_info_lbl)
        self.inspect_lay.addWidget(self.adapter_box)

        self.inspect_lay.addStretch()
        self.scroll_area.setWidget(self.scroll_content)
        self.main_lay.addWidget(self.scroll_area)

        self.set_empty()

    def _create_section(self, title: str) -> QFrame:
        f = QFrame()
        f.setObjectName("cardPanel")
        f.setStyleSheet(
            "QFrame#cardPanel { background-color: #0F131B; border: 1px solid #252C3A; border-radius: 9px; padding: 6px; }"
        )
        l = QVBoxLayout(f)
        l.setContentsMargins(6, 6, 6, 6)
        l.setSpacing(4)
        hdr = QLabel(title)
        hdr.setStyleSheet("color: #8993A7; font-weight: 800; font-size: 10px; letter-spacing: 0.5px;")
        l.addWidget(hdr)
        return f

    def set_empty(self):
        self.empty_widget.setVisible(True)
        self.scroll_area.setVisible(False)

    def display_metadata(self, img_path: str, meta: Optional[Dict[str, Any]]):
        self.empty_widget.setVisible(False)
        self.scroll_area.setVisible(True)

        # Preview pixmap
        pix = QPixmap(img_path)
        if not pix.isNull():
            scaled = pix.scaled(280, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_lbl.setPixmap(scaled)

        p = Path(img_path)
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M:%S")
        fsize = os.path.getsize(p) / (1024 * 1024)
        dims = f"{pix.width()}x{pix.height()}" if not pix.isNull() else "Unknown"

        self.img_info_lbl.setText(f"File: {p.name}\nResolution: {dims} | Size: {fsize:.2f} MB\nCreated: {mtime}")

        if not meta:
            self.model_info_lbl.setText("No embedded generation parameters found.")
            self.prompt_val.setText("N/A")
            self.neg_prompt_val.setText("N/A")
            self.adapter_info_lbl.setText("None")
            return

        # Model
        model_name = meta.get("model", {}).get("id") if isinstance(meta.get("model"), dict) else str(meta.get("model", "Default SDXL"))
        arch = meta.get("model", {}).get("architecture", "sdxl") if isinstance(meta.get("model"), dict) else "sdxl"
        self.model_info_lbl.setText(f"● {model_name} ({arch.upper()})")

        # Prompts
        self.prompt_val.setText(meta.get("prompt", "(Empty)"))
        self.neg_prompt_val.setText(meta.get("negative_prompt", "(None)"))

        # Clear and repopulate Sampling Grid
        while self.sampling_grid.count():
            item = self.sampling_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        params = [
            ("Steps", str(meta.get("steps", 28))),
            ("CFG Scale", f"{meta.get('guidance_scale', 7.0):.1f}"),
            ("Sampler", str(meta.get("sampler", "Euler a"))),
            ("Schedule", str(meta.get("scheduler", "Normal"))),
            ("Seed", str(meta.get("seed", "Random"))),
            ("Denoise", f"{meta.get('denoising_strength', 1.0):.2f}" if "denoising_strength" in meta else "1.00"),
        ]
        for i, (k, v) in enumerate(params):
            row, col = i // 2, (i % 2) * 2
            k_lbl = QLabel(f"{k}:")
            k_lbl.setStyleSheet("color: #8993A7; font-size: 10px; font-weight: 600;")
            v_lbl = QLabel(v)
            v_lbl.setStyleSheet("color: #E8ECF4; font-size: 11px; font-weight: bold; font-family: 'JetBrains Mono', monospace;")
            self.sampling_grid.addWidget(k_lbl, row, col)
            self.sampling_grid.addWidget(v_lbl, row, col + 1)

        # Adapters
        loras = meta.get("loras", {})
        adapter_lines = []
        if isinstance(loras, dict) and loras:
            for l_path, l_wt in loras.items():
                adapter_lines.append(f"• LoRA: {Path(l_path).stem} (Weight: {l_wt:.2f})")
        if meta.get("ip_adapter_name") and meta.get("ip_adapter_name") != "None":
            adapter_lines.append(f"• IP-Adapter: {meta.get('ip_adapter_name')} (Scale: {meta.get('ip_adapter_scale', 0.6):.2f})")
        if meta.get("pulid_enabled"):
            adapter_lines.append(f"• PuLID FaceID: Enabled (Strength: {meta.get('pulid_strength', 0.8):.2f})")
        if meta.get("controlnet_type") and meta.get("controlnet_type") != "None":
            adapter_lines.append(f"• ControlNet: {meta.get('controlnet_type')} (Weight: {meta.get('controlnet_scale', 0.8):.2f})")

        self.adapter_info_lbl.setText("\n".join(adapter_lines) if adapter_lines else "No adapters utilized.")


class CreativeVaultPage(QWidget):
    """
    Creative Vault: Visual Asset Library.
    Left: Search/Filters & Thumbnail-prioritized gallery grid.
    Right: Structured Parameter Inspector with quick action reuse.
    """
    send_to_t2i_requested = Signal(dict)
    send_to_editor_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.all_images: List[Path] = []
        self.current_img_path: Optional[str] = None
        self.current_meta: Optional[dict] = None
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        # ==========================================================
        # LEFT SECTION: Gallery & Search/Filter Controls
        # ==========================================================
        left_box = QFrame()
        left_box.setObjectName("workspacePanel")
        left_lay = QVBoxLayout(left_box)
        left_lay.setContentsMargins(10, 10, 10, 10)
        left_lay.setSpacing(8)

        # Header Row
        hdr_row = QHBoxLayout()
        hdr_badge = QLabel("📁 CREATIVE VAULT — ASSET LIBRARY")
        hdr_badge.setStyleSheet(
            "background: #151A24; border: 1px solid #252C3A; color: #4F9CFF; "
            "border-radius: 6px; padding: 4px 10px; font-weight: 800; font-size: 11px;"
        )
        hdr_row.addWidget(hdr_badge)
        hdr_row.addStretch()

        self.refresh_btn = QPushButton("🔄 Refresh Vault")
        self.refresh_btn.setStyleSheet("font-size: 11px; padding: 4px 10px;")
        hdr_row.addWidget(self.refresh_btn)
        left_lay.addLayout(hdr_row)

        # Search Bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Search prompts, characters, models, or filenames...")
        left_lay.addWidget(self.search_input)

        # Filter Row
        filter_row = QHBoxLayout()
        filter_row.setSpacing(8)

        self.proj_filter = QComboBox()
        self.proj_filter.addItem("All Projects")

        self.char_filter = QComboBox()
        self.char_filter.addItem("All Characters")

        filter_row.addWidget(QLabel("Project:"))
        filter_row.addWidget(self.proj_filter, stretch=1)
        filter_row.addWidget(QLabel("Character:"))
        filter_row.addWidget(self.char_filter, stretch=1)
        left_lay.addLayout(filter_row)

        # Visual Thumbnail Grid
        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(120, 120))
        self.list_widget.setViewMode(QListWidget.IconMode)
        self.list_widget.setResizeMode(QListWidget.Adjust)
        self.list_widget.setSpacing(12)
        self.list_widget.setStyleSheet(
            "QListWidget { background-color: #090C12; border: 1px solid #252C3A; border-radius: 10px; padding: 8px; } "
            "QListWidget::item { background: #151A24; border: 1px solid #252C3A; border-radius: 8px; padding: 6px; } "
            "QListWidget::item:selected { background: #1A2234; border: 1px solid #7C6CFF; } "
            "QListWidget::item:hover:!selected { background: #1D2433; border-color: #353F54; }"
        )
        left_lay.addWidget(self.list_widget, stretch=1)

        layout.addWidget(left_box, stretch=3)

        # ==========================================================
        # RIGHT SECTION: Structured Parameter Inspector
        # ==========================================================
        right_box = QFrame()
        right_box.setObjectName("sidebarPanel")
        right_box.setFixedWidth(380)
        right_lay = QVBoxLayout(right_box)
        right_lay.setContentsMargins(10, 10, 10, 10)
        right_lay.setSpacing(8)

        insp_header = QLabel("🔍 PARAMETER INSPECTOR")
        insp_header.setStyleSheet("color: #8993A7; font-weight: 800; font-size: 11px; letter-spacing: 0.5px;")
        right_lay.addWidget(insp_header)

        self.inspector = StructuredInspectorWidget()
        right_lay.addWidget(self.inspector, stretch=1)

        # Action Buttons
        btn_row = QVBoxLayout()
        btn_row.setSpacing(6)
        self.reuse_btn = QPushButton("⚡ Send Parameters to Text Studio")
        self.reuse_btn.setObjectName("btnPrimary")
        self.send_editor_btn = QPushButton("↗️ Send to Image Editor")
        self.send_editor_btn.setObjectName("btnSecondary")
        self.open_folder_btn = QPushButton("📂 Open Containing Folder")

        btn_row.addWidget(self.reuse_btn)
        btn_row.addWidget(self.send_editor_btn)
        btn_row.addWidget(self.open_folder_btn)
        right_lay.addLayout(btn_row)

        layout.addWidget(right_box, stretch=0)

        # Event connections
        self.refresh_btn.clicked.connect(self.scan_and_reload)
        self.search_input.textChanged.connect(self._apply_filters)
        self.proj_filter.currentIndexChanged.connect(self._apply_filters)
        self.char_filter.currentIndexChanged.connect(self._apply_filters)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.reuse_btn.clicked.connect(self._reuse_parameters)
        self.send_editor_btn.clicked.connect(self._send_to_editor)
        self.open_folder_btn.clicked.connect(self._open_folder)

        self.scan_and_reload()

    def scan_and_reload(self):
        self.list_widget.clear()
        self.all_images.clear()

        # Scan recursively in outputs/
        if paths.OUTPUTS_DIR.exists():
            for p in paths.OUTPUTS_DIR.rglob("*.*"):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    self.all_images.append(p)

        self.all_images.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        # Update Project & Character filters
        projects = set()
        characters = set()
        for p in self.all_images:
            parts = p.parts
            if "Projects" in parts:
                idx = parts.index("Projects")
                if idx + 1 < len(parts):
                    projects.add(parts[idx + 1])
            stem_parts = p.stem.split("_")
            if stem_parts and len(stem_parts[0]) > 1:
                characters.add(stem_parts[0])

        self.proj_filter.blockSignals(True)
        self.proj_filter.clear()
        self.proj_filter.addItem("All Projects")
        for pr in sorted(projects):
            self.proj_filter.addItem(pr)
        self.proj_filter.blockSignals(False)

        self.char_filter.blockSignals(True)
        self.char_filter.clear()
        self.char_filter.addItem("All Characters")
        for ch in sorted(characters):
            self.char_filter.addItem(ch)
        self.char_filter.blockSignals(False)

        self._apply_filters()

    def _apply_filters(self):
        self.list_widget.clear()
        search_query = self.search_input.text().strip().lower()
        selected_proj = self.proj_filter.currentText()
        selected_char = self.char_filter.currentText()

        count = 0
        for img_path in self.all_images:
            if count >= 80:
                break

            # Project filter
            if selected_proj != "All Projects" and selected_proj not in img_path.parts:
                continue

            # Character filter
            if selected_char != "All Characters" and not img_path.name.startswith(selected_char):
                continue

            # Search query
            if search_query and (search_query not in img_path.name.lower()):
                continue

            # Formulate Clean Display Label (e.g. Belfast | 08/30 · 16:34)
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(img_path))
            time_str = mtime.strftime("%m/%d · %H:%M")
            display_title = img_path.stem
            if len(display_title) > 16:
                display_title = display_title[:14] + "..."
            item_text = f"{display_title}\n{time_str}"

            item = QListWidgetItem(item_text)
            item.setIcon(QIcon(str(img_path)))
            item.setData(Qt.UserRole, str(img_path))
            item.setToolTip(f"Filename: {img_path.name}\nPath: {img_path.resolve()}\nCreated: {mtime.strftime('%Y-%m-%d %H:%M')}")
            item.setTextAlignment(Qt.AlignCenter)
            self.list_widget.addItem(item)
            count += 1

    def _on_item_clicked(self, item: QListWidgetItem):
        self.current_img_path = item.data(Qt.UserRole)
        if not self.current_img_path or not os.path.exists(self.current_img_path):
            self.inspector.set_empty()
            return

        try:
            with Image.open(self.current_img_path) as img:
                raw_meta = img.info.get("parameters", "")
                if raw_meta:
                    self.current_meta = json.loads(raw_meta)
                else:
                    self.current_meta = None
                self.inspector.display_metadata(self.current_img_path, self.current_meta)
        except Exception as e:
            self.current_meta = None
            self.inspector.display_metadata(self.current_img_path, None)

    def _reuse_parameters(self):
        if self.current_meta:
            self.send_to_t2i_requested.emit(self.current_meta)

    def _send_to_editor(self):
        if self.current_img_path:
            self.send_to_editor_requested.emit(self.current_img_path)

    def _open_folder(self):
        if self.current_img_path:
            folder = Path(self.current_img_path).parent
            if folder.exists():
                subprocess.Popen(f'explorer "{folder.resolve()}"')
