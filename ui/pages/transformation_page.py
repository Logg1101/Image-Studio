import os
import random
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton,
    QDoubleSpinBox, QComboBox, QGroupBox, QProgressBar, QFileDialog,
    QMessageBox, QFrame, QScrollArea, QCheckBox, QSlider, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap

from ui.state import model_manager
from adapters.registry import adapter_registry
from core.types import GenerationRequest, GenerationResult, MAX_SEED, sanitize_seed
from core.project_manager import ProjectManager
from ui.controllers.generation_controller import GenerationWorker
from ui.widgets.image_canvas import ImageCanvasWidget
from ui.widgets.reference_asset_manager import ReferenceAssetManagerWidget, ReferenceAsset

class CollapsibleSection(QFrame):
    """Clean collapsible section header with expand/collapse toggle."""
    def __init__(self, title: str, is_open: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("cardPanel")
        self.setStyleSheet(
            "QFrame#cardPanel { background-color: #0F131B; border: 1px solid #252C3A; border-radius: 10px; padding: 4px; } "
        )
        self.main_lay = QVBoxLayout(self)
        self.main_lay.setContentsMargins(6, 6, 6, 6)
        self.main_lay.setSpacing(6)

        # Header bar
        self.header_btn = QPushButton(f"{'▼' if is_open else '▶'}  {title}")
        self.header_btn.setCursor(Qt.PointingHandCursor)
        self.header_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; text-align: left; "
            "color: #E8ECF4; font-weight: 800; font-size: 11px; letter-spacing: 0.5px; padding: 4px 2px; } "
            "QPushButton:hover { color: #7C6CFF; }"
        )
        self.main_lay.addWidget(self.header_btn)

        # Content container
        self.content_widget = QWidget()
        self.content_lay = QVBoxLayout(self.content_widget)
        self.content_lay.setContentsMargins(4, 2, 4, 4)
        self.content_lay.setSpacing(6)
        self.main_lay.addWidget(self.content_widget)

        self.is_open = is_open
        self.content_widget.setVisible(is_open)
        self.header_btn.clicked.connect(self.toggle)
        self.title_text = title

    def toggle(self):
        self.is_open = not self.is_open
        self.content_widget.setVisible(self.is_open)
        self.header_btn.setText(f"{'▼' if self.is_open else '▶'}  {self.title_text}")


class TransformationPage(QWidget):
    """
    Adapter Lab Redesign:
    Mental Model: REFERENCE ASSETS -> ADAPTERS (IP-Adapter, PuLID, ControlNet) -> GENERATION/TRANSFORMATION
    """
    send_to_editor_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: Optional[GenerationWorker] = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(10)

        # ==========================================================
        # LEFT CONTROL COLUMN (Scrollable, 460-490px width)
        # ==========================================================
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(470)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(4, 4, 8, 4)
        scroll_layout.setSpacing(10)

        # ----------------------------------------------------------
        # 1. UNIFIED REFERENCE ASSET SHELF
        # ----------------------------------------------------------
        self.ref_manager = ReferenceAssetManagerWidget()
        scroll_layout.addWidget(self.ref_manager)

        # ----------------------------------------------------------
        # 2. ADAPTERS (Collapsible Sections)
        # ----------------------------------------------------------
        adapters_header = QLabel("⚡ ADAPTERS CONFIGURATION")
        adapters_header.setStyleSheet("color: #8993A7; font-weight: 800; font-size: 11px; letter-spacing: 0.6px; margin-top: 4px;")
        scroll_layout.addWidget(adapters_header)

        # --- A. IP-ADAPTER ---
        self.ip_section = CollapsibleSection("IP-ADAPTER STYLE & CONCEPT TRANSFER", is_open=True)
        ip_content = self.ip_section.content_lay

        ip_m_row = QHBoxLayout()
        ip_m_row.addWidget(QLabel("Module / Weight:"))
        self.ip_combo = QComboBox()
        self.ip_combo.addItem("None")
        for ip in sorted(adapter_registry.scan_ip_adapters().keys()):
            self.ip_combo.addItem(ip)
        ip_m_row.addWidget(self.ip_combo, stretch=1)
        ip_content.addLayout(ip_m_row)

        ip_scale_row = QHBoxLayout()
        ip_scale_row.addWidget(QLabel("Scale / Weight:"))
        self.ip_scale_spin = QDoubleSpinBox()
        self.ip_scale_spin.setRange(0.0, 1.5)
        self.ip_scale_spin.setSingleStep(0.05)
        self.ip_scale_spin.setValue(0.60)
        self.ip_scale_spin.setFixedWidth(64)
        ip_scale_row.addWidget(self.ip_scale_spin)

        self.ip_ref_status = QLabel("Reference: Auto (from Reference Assets)")
        self.ip_ref_status.setStyleSheet("color: #35D6C5; font-size: 10px; font-style: italic;")
        ip_scale_row.addStretch()
        ip_scale_row.addWidget(self.ip_ref_status)
        ip_content.addLayout(ip_scale_row)
        scroll_layout.addWidget(self.ip_section)

        # --- B. PULID (FACE ID) ---
        self.pulid_section = CollapsibleSection("PULID HIGH-FIDELITY FACE ID", is_open=True)
        pulid_content = self.pulid_section.content_lay

        pulid_top = QHBoxLayout()
        self.pulid_chk = QCheckBox("Enable PuLID Face Identity")
        self.pulid_chk.setChecked(False)
        self.pulid_chk.setStyleSheet("font-weight: 700; color: #E8ECF4;")
        pulid_top.addWidget(self.pulid_chk)

        pulid_top.addStretch()
        pulid_top.addWidget(QLabel("Strength:"))
        self.pulid_strength_spin = QDoubleSpinBox()
        self.pulid_strength_spin.setRange(0.0, 1.5)
        self.pulid_strength_spin.setSingleStep(0.05)
        self.pulid_strength_spin.setValue(0.80)
        self.pulid_strength_spin.setFixedWidth(64)
        pulid_top.addWidget(self.pulid_strength_spin)
        pulid_content.addLayout(pulid_top)

        self.pulid_ref_status = QLabel("Reference: Auto (Face asset from Reference Shelf)")
        self.pulid_ref_status.setStyleSheet("color: #35D6C5; font-size: 10px; font-style: italic;")
        pulid_content.addWidget(self.pulid_ref_status)
        scroll_layout.addWidget(self.pulid_section)

        # --- C. CONTROLNET GUIDANCE ---
        self.cnet_section = CollapsibleSection("CONTROLNET STRUCTURAL GUIDANCE", is_open=False)
        cnet_content = self.cnet_section.content_lay

        cn_m_row = QHBoxLayout()
        cn_m_row.addWidget(QLabel("Module:"))
        self.cnet_combo = QComboBox()
        self.cnet_combo.addItem("None")
        for c in sorted(adapter_registry.scan_controlnets().keys()):
            self.cnet_combo.addItem(c)
        cn_m_row.addWidget(self.cnet_combo, stretch=1)
        cnet_content.addLayout(cn_m_row)

        cn_scale_row = QHBoxLayout()
        cn_scale_row.addWidget(QLabel("Strength:"))
        self.cnet_scale_spin = QDoubleSpinBox()
        self.cnet_scale_spin.setRange(0.0, 2.0)
        self.cnet_scale_spin.setSingleStep(0.05)
        self.cnet_scale_spin.setValue(0.80)
        self.cnet_scale_spin.setFixedWidth(64)
        cn_scale_row.addWidget(self.cnet_scale_spin)

        self.cnet_ref_status = QLabel("Reference: Auto (from Reference Assets)")
        self.cnet_ref_status.setStyleSheet("color: #35D6C5; font-size: 10px; font-style: italic;")
        cn_scale_row.addStretch()
        cn_scale_row.addWidget(self.cnet_ref_status)
        cnet_content.addLayout(cn_scale_row)
        scroll_layout.addWidget(self.cnet_section)

        # ----------------------------------------------------------
        # 3. GENERATION & TRANSFORMATION PARAMETERS
        # ----------------------------------------------------------
        gen_header = QLabel("🎛️ GENERATION & PROMPTS")
        gen_header.setStyleSheet("color: #8993A7; font-weight: 800; font-size: 11px; letter-spacing: 0.6px; margin-top: 4px;")
        scroll_layout.addWidget(gen_header)

        # Img2Img Denoising Slider Card
        denoise_card = QFrame()
        denoise_card.setObjectName("cardPanel")
        denoise_lay = QVBoxLayout(denoise_card)
        denoise_lay.setContentsMargins(8, 6, 8, 6)
        denoise_lay.setSpacing(4)

        d_hdr_row = QHBoxLayout()
        d_title = QLabel("Denoising Strength (0.0=Original, 1.0=Full Dream)")
        d_title.setStyleSheet("color: #8993A7; font-size: 10px; font-weight: 600; text-transform: uppercase;")
        self.denoise_spin = QDoubleSpinBox()
        self.denoise_spin.setRange(0.0, 1.0)
        self.denoise_spin.setSingleStep(0.05)
        self.denoise_spin.setValue(0.65)
        self.denoise_spin.setFixedWidth(64)
        d_hdr_row.addWidget(d_title)
        d_hdr_row.addStretch()
        d_hdr_row.addWidget(self.denoise_spin)
        denoise_lay.addLayout(d_hdr_row)

        self.denoise_slider = QSlider(Qt.Horizontal)
        self.denoise_slider.setRange(0, 100)
        self.denoise_slider.setValue(65)
        denoise_lay.addWidget(self.denoise_slider)
        scroll_layout.addWidget(denoise_card)

        self.denoise_spin.valueChanged.connect(lambda v: self.denoise_slider.setValue(int(v * 100)))
        self.denoise_slider.valueChanged.connect(lambda v: self.denoise_spin.setValue(v / 100.0))

        # Transformation Prompt
        p_box = QGroupBox("TRANSFORMATION PROMPT")
        p_lay = QVBoxLayout(p_box)
        p_lay.setContentsMargins(8, 8, 8, 6)
        p_lay.setSpacing(4)
        self.prompt_text = QTextEdit()
        self.prompt_text.setPlaceholderText("Describe modifications, style additions, lighting...")
        self.prompt_text.setMinimumHeight(65)
        self.prompt_text.setMaximumHeight(90)
        p_lay.addWidget(self.prompt_text)

        p_lay.addWidget(QLabel("NEGATIVE PROMPT (OPTIONAL):"))
        self.neg_prompt_text = QTextEdit()
        self.neg_prompt_text.setText("blurry, low quality, distorted, bad anatomy, deformed")
        self.neg_prompt_text.setFixedHeight(45)
        p_lay.addWidget(self.neg_prompt_text)
        scroll_layout.addWidget(p_box)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, stretch=0)

        # ==========================================================
        # RIGHT WORKSPACE / CENTERPIECE PREVIEW CANVAS
        # ==========================================================
        right_panel = QFrame()
        right_panel.setObjectName("workspacePanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)

        # Output Canvas Header
        canvas_hdr = QHBoxLayout()
        badge = QLabel("🪄 Adapter Lab Live Canvas")
        badge.setStyleSheet(
            "background: #151A24; border: 1px solid #252C3A; color: #4F9CFF; "
            "border-radius: 6px; padding: 4px 10px; font-weight: 800; font-size: 11px;"
        )
        canvas_hdr.addWidget(badge)
        canvas_hdr.addStretch()

        self.save_btn = QPushButton("💾 Save Image As...")
        self.save_btn.setStyleSheet("font-size: 11px; padding: 4px 10px;")
        canvas_hdr.addWidget(self.save_btn)
        right_layout.addLayout(canvas_hdr)

        # Centerpiece Canvas
        self.canvas = ImageCanvasWidget()
        right_layout.addWidget(self.canvas, stretch=1)

        # Status & Progress Telemetry
        self.telemetry_label = QLabel("Adapter Lab | Ready to transform")
        self.telemetry_label.setAlignment(Qt.AlignCenter)
        self.telemetry_label.setStyleSheet(
            "background: #151A24; border: 1px solid #252C3A; border-radius: 6px; "
            "color: #10B981; font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 6px 12px;"
        )
        right_layout.addWidget(self.telemetry_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        # Primary Action Bar
        action_bar = QHBoxLayout()
        action_bar.setSpacing(8)

        self.gen_btn = QPushButton("⚡ RUN TRANSFORMATION")
        self.gen_btn.setObjectName("btnPrimary")
        self.stop_btn = QPushButton("🛑 STOP")
        self.stop_btn.setObjectName("btnStop")
        self.stop_btn.setEnabled(False)

        action_bar.addWidget(self.gen_btn, stretch=3)
        action_bar.addWidget(self.stop_btn, stretch=1)
        right_layout.addLayout(action_bar)

        # Secondary Actions
        sub_bar = QHBoxLayout()
        self.send_editor_btn = QPushButton("↗️ Send to Image Editor")
        self.send_editor_btn.setObjectName("btnSecondary")
        sub_bar.addWidget(self.send_editor_btn)
        sub_bar.addStretch()
        right_layout.addLayout(sub_bar)

        main_layout.addWidget(right_panel, stretch=1)

        # Connect events
        self.gen_btn.clicked.connect(self._start_generation)
        self.stop_btn.clicked.connect(self._cancel_generation)
        self.save_btn.clicked.connect(self._save_image)
        self.send_editor_btn.clicked.connect(self._send_to_editor)
        self.ref_manager.references_changed.connect(self._on_references_updated)

        self._on_references_updated()

    def _on_references_updated(self):
        # Update auto-detection status labels
        src_asset = self.ref_manager.get_source_asset()
        ip_asset = self.ref_manager.get_ip_adapter_asset()
        pulid_asset = self.ref_manager.get_pulid_asset()
        cnet_asset = self.ref_manager.get_controlnet_asset()

        if ip_asset:
            self.ip_ref_status.setText(f"✓ Bound: Ref '{ip_asset.display_name}'")
            self.ip_ref_status.setStyleSheet("color: #10B981; font-size: 10px; font-weight: 600;")
        else:
            self.ip_ref_status.setText("No reference tagged for IP-Adapter")
            self.ip_ref_status.setStyleSheet("color: #8993A7; font-size: 10px; font-style: italic;")

        if pulid_asset:
            self.pulid_ref_status.setText(f"✓ Bound: Ref '{pulid_asset.display_name}'")
            self.pulid_ref_status.setStyleSheet("color: #10B981; font-size: 10px; font-weight: 600;")
        else:
            self.pulid_ref_status.setText("No reference tagged for PuLID")
            self.pulid_ref_status.setStyleSheet("color: #8993A7; font-size: 10px; font-style: italic;")

        if cnet_asset:
            self.cnet_ref_status.setText(f"✓ Bound: Ref '{cnet_asset.display_name}'")
            self.cnet_ref_status.setStyleSheet("color: #10B981; font-size: 10px; font-weight: 600;")
        else:
            self.cnet_ref_status.setText("No reference tagged for ControlNet")
            self.cnet_ref_status.setStyleSheet("color: #8993A7; font-size: 10px; font-style: italic;")

    def load_source_image(self, path: str):
        """Cross-tab entrypoint: imports image into unified reference shelf."""
        self.ref_manager.add_reference(path, consumers={"source", "ip_adapter"})
        self._on_references_updated()

    def _start_generation(self):
        # 1. Resolve source image from unified references
        src_pil = self.ref_manager.get_source_pil_image()
        if not src_pil:
            QMessageBox.warning(
                self, "Reference Required",
                "Please add a reference image in the Reference Assets shelf tagged as 'Source (Img2Img)'."
            )
            return

        models = list(model_manager.available_models.values())
        if not models:
            QMessageBox.warning(self, "Error", "No models available.")
            return
        model_info = models[0]

        # 2. Resolve adapter assets
        pulid_on = self.pulid_chk.isChecked()
        pulid_img = self.ref_manager.get_pulid_pil_image() if pulid_on else None
        pulid_str = float(self.pulid_strength_spin.value())

        cnet_type = self.cnet_combo.currentText() if self.cnet_combo.currentText() != "None" else None
        cnet_img = self.ref_manager.get_controlnet_pil_image() if cnet_type else None
        if cnet_type and not cnet_img:
            # Fallback to source image if not explicitly tagged
            cnet_img = src_pil

        ip_name = self.ip_combo.currentText() if self.ip_combo.currentText() != "None" else None
        ip_img = self.ref_manager.get_ip_adapter_pil_image() if ip_name else None
        if ip_name and not ip_img:
            ip_img = src_pil

        req = GenerationRequest(
            model=model_info,
            prompt=self.prompt_text.toPlainText().strip() or "masterpiece, high quality, highly detailed",
            negative_prompt=self.neg_prompt_text.toPlainText().strip(),
            width=src_pil.width,
            height=src_pil.height,
            steps=28,
            guidance_scale=7.0,
            sampler="Euler a",
            scheduler="Normal",
            seed=random.randint(0, MAX_SEED),
            init_image=src_pil,
            denoising_strength=float(self.denoise_spin.value()),
            pulid_enabled=pulid_on and pulid_img is not None,
            pulid_image=pulid_img,
            pulid_strength=pulid_str,
            controlnet_type=cnet_type,
            controlnet_image=cnet_img,
            controlnet_scale=float(self.cnet_scale_spin.value()),
            ip_adapter_name=ip_name,
            ip_adapter_image=ip_img,
            ip_adapter_scale=float(self.ip_scale_spin.value())
        )

        self.progress_bar.setVisible(True)
        self.gen_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.telemetry_label.setText("Preparing pipeline & loading adapters...")

        self.worker = GenerationWorker(req)
        self.worker.progress_changed.connect(self._on_progress)
        self.worker.finished_success.connect(self._on_success)
        self.worker.finished_error.connect(self._on_error)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.start()

    def _on_progress(self, step: int, total: int, desc: str):
        pct = int((step / max(total, 1)) * 100)
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"{desc} ({pct}%)")
        self.telemetry_label.setText(f"Rendering: Step {step}/{total} ({pct}%)")

    def _on_success(self, res: GenerationResult):
        self.canvas.set_image(res.image_path)
        self._reset_ui()
        self.telemetry_label.setText(f"✓ Transformation completed in {res.generation_time_ms / 1000:.2f}s")
        self.progress_bar.setVisible(False)

    def _on_error(self, err: str):
        QMessageBox.critical(self, "Transformation Error", err)
        self._reset_ui()
        self.telemetry_label.setText(f"⚠️ Error: {err}")
        self.progress_bar.setVisible(False)

    def _on_cancelled(self):
        self._reset_ui()
        self.telemetry_label.setText("Transformation cancelled by user.")
        self.progress_bar.setVisible(False)

    def _cancel_generation(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()

    def _reset_ui(self):
        self.gen_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _save_image(self):
        if not self.canvas.current_image_path:
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Save Output Image", "transformed_output.png", "PNG Images (*.png)")
        if dest:
            import shutil
            shutil.copy(self.canvas.current_image_path, dest)

    def _send_to_editor(self):
        if self.canvas.current_image_path:
            self.send_to_editor_requested.emit(self.canvas.current_image_path)
