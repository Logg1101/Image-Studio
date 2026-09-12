import os
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QLineEdit,
    QPushButton, QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox,
    QProgressBar, QFileDialog, QMessageBox, QFrame, QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QTimer

from ui.state import model_manager
from adapters.registry import adapter_registry
from core.types import GenerationRequest, GenerationResult, MAX_SEED, sanitize_seed
from core.prompt_enhancer import AIPromptEnhancer
from engines.sdxl.schedulers import AVAILABLE_SAMPLERS, AVAILABLE_SCHEDULERS
from ui.controllers.generation_controller import GenerationWorker
from ui.widgets.slider_card import SliderCardWidget
from ui.widgets.lora_rack import LoRARackWidget
from ui.widgets.image_canvas import ImageCanvasWidget

class TextStudioPage(QWidget):
    """
    Professional Text-to-Image Studio.
    Left: Scrollable configuration sidebar (Checkpoint, Dimensions, Sampling, LoRA Stack, Upscaling).
    Right: Prompt workstation, centerpiece canvas, and real-time generation telemetry.
    """
    send_to_editor_requested = Signal(str)
    send_to_i2i_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: Optional[GenerationWorker] = None
        self.neg_prompt_visible = False
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(10)

        # ==========================================================
        # 1. LEFT SIDEBAR PANEL (Scrollable independently, ~350px)
        # ==========================================================
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(350)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        sidebar_content = QWidget()
        sidebar_lay = QVBoxLayout(sidebar_content)
        sidebar_lay.setContentsMargins(4, 4, 8, 4)
        sidebar_lay.setSpacing(10)

        # Section 1: ACTIVE CHECKPOINT
        ckpt_group = QFrame()
        ckpt_group.setObjectName("cardPanel")
        ckpt_lay = QVBoxLayout(ckpt_group)
        ckpt_lay.setContentsMargins(8, 8, 8, 8)
        ckpt_lay.setSpacing(6)

        ckpt_header = QHBoxLayout()
        ckpt_lbl = QLabel("🧠 ACTIVE CHECKPOINT")
        ckpt_lbl.setStyleSheet("color: #8993A7; font-weight: 800; font-size: 11px; letter-spacing: 0.5px;")
        self.status_dot = QLabel("● Ready")
        self.status_dot.setStyleSheet("color: #10B981; font-size: 10px; font-weight: bold;")
        ckpt_header.addWidget(ckpt_lbl)
        ckpt_header.addStretch()
        ckpt_header.addWidget(self.status_dot)
        ckpt_lay.addLayout(ckpt_header)

        ckpt_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.refresh_models_btn = QPushButton("🔄")
        self.refresh_models_btn.setFixedSize(30, 30)
        self.refresh_models_btn.setToolTip("Rescan Checkpoints")
        ckpt_row.addWidget(self.model_combo, stretch=1)
        ckpt_row.addWidget(self.refresh_models_btn)
        ckpt_lay.addLayout(ckpt_row)
        sidebar_lay.addWidget(ckpt_group)

        # Section 2: ASPECT RATIO & DIMENSIONS
        ar_group = QFrame()
        ar_group.setObjectName("cardPanel")
        ar_lay = QVBoxLayout(ar_group)
        ar_lay.setContentsMargins(8, 8, 8, 8)
        ar_lay.setSpacing(6)

        ar_header = QLabel("📐 ASPECT RATIO & RESOLUTION")
        ar_header.setStyleSheet("color: #8993A7; font-weight: 800; font-size: 11px; letter-spacing: 0.5px;")
        ar_lay.addWidget(ar_header)

        ar_btn_row = QHBoxLayout()
        ar_btn_row.setSpacing(4)
        self.ar_buttons = {}
        ratios = [("1:1", 1024, 1024), ("16:9", 1344, 768), ("9:16", 768, 1344), ("4:3", 1152, 896), ("3:4", 896, 1152)]
        for label, w, h in ratios:
            btn = QPushButton(label)
            btn.setObjectName("aspectBtn")
            btn.setCheckable(True)
            if label == "9:16":
                btn.setChecked(True)
            btn.clicked.connect(lambda _, w=w, h=h, l=label: self._on_aspect_clicked(l, w, h))
            ar_btn_row.addWidget(btn)
            self.ar_buttons[label] = btn
        ar_lay.addLayout(ar_btn_row)

        self.width_card = SliderCardWidget("Width (px)", 512, 2048, 768, step=64)
        self.height_card = SliderCardWidget("Height (px)", 512, 2048, 1344, step=64)
        ar_lay.addWidget(self.width_card)
        ar_lay.addWidget(self.height_card)
        sidebar_lay.addWidget(ar_group)

        # Section 3: SAMPLING PARAMETERS
        samp_group = QFrame()
        samp_group.setObjectName("cardPanel")
        samp_lay = QVBoxLayout(samp_group)
        samp_lay.setContentsMargins(8, 8, 8, 8)
        samp_lay.setSpacing(6)

        samp_header = QLabel("🎛️ SAMPLING PARAMETERS")
        samp_header.setStyleSheet("color: #8993A7; font-weight: 800; font-size: 11px; letter-spacing: 0.5px;")
        samp_lay.addWidget(samp_header)

        self.steps_card = SliderCardWidget("Sampling Steps (SDXL: 25-35)", 10, 60, 30)
        self.cfg_card = SliderCardWidget("CFG Guidance (SDXL: 5-8)", 1, 15, 7)
        samp_lay.addWidget(self.steps_card)
        samp_lay.addWidget(self.cfg_card)

        samp_sched_row = QHBoxLayout()
        s_box = QVBoxLayout()
        s_box.setSpacing(2)
        s_lbl = QLabel("Sampler:")
        s_lbl.setStyleSheet("color: #8993A7; font-size: 10px; font-weight: 600;")
        self.sampler_combo = QComboBox()
        self.sampler_combo.addItems(AVAILABLE_SAMPLERS)
        s_box.addWidget(s_lbl)
        s_box.addWidget(self.sampler_combo)
        samp_sched_row.addLayout(s_box)

        sc_box = QVBoxLayout()
        sc_box.setSpacing(2)
        sc_lbl = QLabel("Schedule:")
        sc_lbl.setStyleSheet("color: #8993A7; font-size: 10px; font-weight: 600;")
        self.sched_combo = QComboBox()
        self.sched_combo.addItems(AVAILABLE_SCHEDULERS)
        sc_box.addWidget(sc_lbl)
        sc_box.addWidget(self.sched_combo)
        samp_sched_row.addLayout(sc_box)
        samp_lay.addLayout(samp_sched_row)
        sidebar_lay.addWidget(samp_group)

        # Section 4: LORA RACK / ADAPTERS
        self.lora_rack = LoRARackWidget(available_loras=adapter_registry.scan_loras())
        sidebar_lay.addWidget(self.lora_rack)

        # Section 5: HIGH-RES UPSCALER
        up_group = QFrame()
        up_group.setObjectName("cardPanel")
        up_lay = QVBoxLayout(up_group)
        up_lay.setContentsMargins(8, 8, 8, 8)
        up_lay.setSpacing(6)

        up_header = QLabel("🔍 HIGH-RES UPSCALER")
        up_header.setStyleSheet("color: #8993A7; font-weight: 800; font-size: 11px; letter-spacing: 0.5px;")
        up_lay.addWidget(up_header)

        up_row = QHBoxLayout()
        up_m_box = QVBoxLayout()
        up_m_box.setSpacing(2)
        up_m_lbl = QLabel("Method:")
        up_m_lbl.setStyleSheet("color: #8993A7; font-size: 10px; font-weight: 600;")
        self.upscale_combo = QComboBox()
        self.upscale_combo.addItems(["None", "4x-RealCUGAN", "Tiled SDXL", "Lanczos", "Bicubic"])
        up_m_box.addWidget(up_m_lbl)
        up_m_box.addWidget(self.upscale_combo)
        up_row.addLayout(up_m_box)

        up_s_box = QVBoxLayout()
        up_s_box.setSpacing(2)
        up_s_lbl = QLabel("Scale:")
        up_s_lbl.setStyleSheet("color: #8993A7; font-size: 10px; font-weight: 600;")
        self.scale_factor_combo = QComboBox()
        self.scale_factor_combo.addItems(["1.5x", "2.0x", "3.0x", "4.0x"])
        self.scale_factor_combo.setCurrentIndex(1)
        up_s_box.addWidget(up_s_lbl)
        up_s_box.addWidget(self.scale_factor_combo)
        up_row.addLayout(up_s_box)
        up_lay.addLayout(up_row)
        sidebar_lay.addWidget(up_group)

        sidebar_lay.addStretch()
        scroll_area.setWidget(sidebar_content)
        main_layout.addWidget(scroll_area, stretch=0)

        # ==========================================================
        # 2. RIGHT WORKSPACE PANEL (Prompt Editor & Canvas)
        # ==========================================================
        workspace = QFrame()
        workspace.setObjectName("workspacePanel")
        workspace_lay = QVBoxLayout(workspace)
        workspace_lay.setContentsMargins(10, 10, 10, 10)
        workspace_lay.setSpacing(8)

        # Inner Two-Column: Prompt & Action vs Live Output Canvas
        inner_row = QHBoxLayout()
        inner_row.setSpacing(12)

        # --- LEFT INNER: Prompt Editor ---
        left_inner = QVBoxLayout()
        left_inner.setSpacing(8)

        # Prompt Header
        p_hdr_row = QHBoxLayout()
        p_title = QLabel("✍️ PROMPT WORKSPACE")
        p_title.setStyleSheet("font-weight: 800; font-size: 11px; color: #E8ECF4; letter-spacing: 0.5px;")
        self.token_label = QLabel("~0 tokens (0 chars)")
        self.token_label.setStyleSheet(
            "background: #151A24; border: 1px solid #252C3A; border-radius: 4px; "
            "padding: 2px 8px; font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #8993A7;"
        )
        p_hdr_row.addWidget(p_title)
        p_hdr_row.addStretch()
        p_hdr_row.addWidget(self.token_label)
        left_inner.addLayout(p_hdr_row)

        # Prompt Box
        self.prompt_text = QTextEdit()
        self.prompt_text.setPlaceholderText("Describe your image: character, subject, style, lighting, composition...")
        self.prompt_text.setText("1girl, masterpiece, highly detailed, Belfast, Azur Lane, white long hair, blue eyes, maid headdress, chain choker, maid uniform, elegant, cinematic lighting")
        self.prompt_text.setMinimumHeight(130)
        left_inner.addWidget(self.prompt_text)

        # Tools Row
        tools_row = QHBoxLayout()
        tools_row.setSpacing(6)
        self.enhance_btn = QPushButton("✨ AI Enhance")
        self.rand_prompt_btn = QPushButton("🎲 Randomize")
        self.clear_prompt_btn = QPushButton("🧹 Clear")

        for b in (self.enhance_btn, self.rand_prompt_btn, self.clear_prompt_btn):
            b.setStyleSheet("font-size: 11px; padding: 5px 10px;")
        tools_row.addWidget(self.enhance_btn)
        tools_row.addWidget(self.rand_prompt_btn)
        tools_row.addWidget(self.clear_prompt_btn)
        tools_row.addStretch()
        left_inner.addLayout(tools_row)

        # Negative Prompt Collapsible
        self.neg_toggle_btn = QPushButton("Negative Prompt  ▼")
        self.neg_toggle_btn.setStyleSheet(
            "text-align: left; background: #151A24; border: 1px solid #252C3A; "
            "border-radius: 6px; padding: 6px 10px; color: #8993A7; font-weight: 600;"
        )
        left_inner.addWidget(self.neg_toggle_btn)

        self.neg_prompt_text = QTextEdit()
        self.neg_prompt_text.setPlaceholderText("Elements to suppress (e.g. blurry, low quality, bad anatomy)...")
        self.neg_prompt_text.setText("blurry, low quality, distorted, bad anatomy, deformed, artifacts, ugly")
        self.neg_prompt_text.setFixedHeight(60)
        self.neg_prompt_text.setVisible(False)
        left_inner.addWidget(self.neg_prompt_text)

        left_inner.addStretch()

        # Primary Action Bar
        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.gen_btn = QPushButton("⚡ Generate Image")
        self.gen_btn.setObjectName("btnGenerate")
        self.stop_btn = QPushButton("🛑 Stop")
        self.stop_btn.setObjectName("btnStop")
        self.stop_btn.setEnabled(False)

        action_row.addWidget(self.gen_btn, stretch=3)
        action_row.addWidget(self.stop_btn, stretch=1)
        left_inner.addLayout(action_row)

        inner_row.addLayout(left_inner, stretch=3)

        # --- RIGHT INNER: Output Canvas ---
        right_inner = QVBoxLayout()
        right_inner.setSpacing(6)

        out_hdr = QHBoxLayout()
        out_badge = QLabel("🖼️ Render Output Canvas")
        out_badge.setStyleSheet(
            "background: #151A24; border: 1px solid #252C3A; color: #4F9CFF; "
            "border-radius: 6px; padding: 4px 10px; font-weight: 800; font-size: 11px;"
        )
        out_hdr.addWidget(out_badge)
        out_hdr.addStretch()

        self.save_icon_btn = QPushButton("💾 Save Image")
        self.save_icon_btn.setStyleSheet("font-size: 11px; padding: 4px 10px;")
        out_hdr.addWidget(self.save_icon_btn)
        right_inner.addLayout(out_hdr)

        self.canvas = ImageCanvasWidget()
        right_inner.addWidget(self.canvas, stretch=1)

        self.status_bar_lbl = QLabel("✓ Ready to generate")
        self.status_bar_lbl.setAlignment(Qt.AlignCenter)
        self.status_bar_lbl.setStyleSheet(
            "background: #151A24; border: 1px solid #252C3A; border-radius: 6px; "
            "color: #10B981; font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 6px 12px;"
        )
        right_inner.addWidget(self.status_bar_lbl)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setVisible(False)
        right_inner.addWidget(self.progress_bar)

        self.send_i2i_btn = QPushButton("↗️ Send to Img2Img / Adapter Lab")
        self.send_i2i_btn.setObjectName("btnSecondary")
        right_inner.addWidget(self.send_i2i_btn)

        inner_row.addLayout(right_inner, stretch=3)
        workspace_lay.addLayout(inner_row)

        main_layout.addWidget(workspace, stretch=1)

        # Event connections
        self._refresh_models()
        self.refresh_models_btn.clicked.connect(self._refresh_models)
        self.prompt_text.textChanged.connect(self._update_tokens)
        self.enhance_btn.clicked.connect(self._enhance_prompt)
        self.rand_prompt_btn.clicked.connect(self._randomize_prompt)
        self.clear_prompt_btn.clicked.connect(self.prompt_text.clear)
        self.neg_toggle_btn.clicked.connect(self._toggle_neg_prompt)
        self.gen_btn.clicked.connect(self._start_generation)
        self.stop_btn.clicked.connect(self._cancel_generation)
        self.save_icon_btn.clicked.connect(self._save_image)
        self.send_i2i_btn.clicked.connect(self._send_to_i2i)
        self._update_tokens()

    def _refresh_models(self):
        self.model_combo.clear()
        for name in model_manager.available_models.keys():
            self.model_combo.addItem(name)
        self.lora_rack.set_available_loras(adapter_registry.scan_loras())

    def _on_aspect_clicked(self, label: str, w: int, h: int):
        for l, btn in self.ar_buttons.items():
            btn.setChecked(l == label)
        self.width_card.setValue(w)
        self.height_card.setValue(h)

    def _update_tokens(self):
        txt = self.prompt_text.toPlainText()
        chars = len(txt)
        tokens = int(len(txt.split()) * 1.3)
        self.token_label.setText(f"~{tokens} tokens ({chars} chars)")

    def _enhance_prompt(self):
        cur = self.prompt_text.toPlainText().strip()
        enhanced = AIPromptEnhancer.enhance(cur)
        self.prompt_text.setText(enhanced)

    def _randomize_prompt(self):
        random_ideas = [
            "cyberpunk android samurai standing in neon rain, intricate glowing tattoos, reflective puddle, volumetric fog",
            "majestic ancient dragon perched atop obsidian mountain peaks, aurora borealis sky, highly detailed fantasy digital painting",
            "cosy anime coffee shop on a rainy autumn evening, warm glowing lanterns, steam rising from mug, soft bokeh"
        ]
        self.prompt_text.setText(random.choice(random_ideas))

    def _toggle_neg_prompt(self):
        self.neg_prompt_visible = not self.neg_prompt_visible
        self.neg_prompt_text.setVisible(self.neg_prompt_visible)
        self.neg_toggle_btn.setText("Negative Prompt  ▲" if self.neg_prompt_visible else "Negative Prompt  ▼")

    def _start_generation(self):
        selected_model_name = self.model_combo.currentText()
        if not selected_model_name or selected_model_name not in model_manager.available_models:
            QMessageBox.warning(self, "Model Required", "Please select an available checkpoint.")
            return

        model_info = model_manager.available_models[selected_model_name]
        loras = self.lora_rack.get_active_loras()
        prompt = self.prompt_text.toPlainText().strip() or "masterpiece, high quality"
        neg_prompt = self.neg_prompt_text.toPlainText().strip() if self.neg_prompt_visible else None

        req = GenerationRequest(
            model=model_info,
            prompt=prompt,
            negative_prompt=neg_prompt,
            width=self.width_card.getValue(),
            height=self.height_card.getValue(),
            steps=self.steps_card.getValue(),
            guidance_scale=float(self.cfg_card.getValue()),
            sampler=self.sampler_combo.currentText(),
            scheduler=self.sched_combo.currentText(),
            seed=random.randint(0, MAX_SEED),
            loras=loras,
            upscale_method=self.upscale_combo.currentText(),
            upscale_factor=float(self.scale_factor_combo.currentText().replace("x", ""))
        )

        self.progress_bar.setVisible(True)
        self.gen_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_bar_lbl.setText("⚙️ Initializing pipeline and weights...")

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
        self.status_bar_lbl.setText(f"Rendering: Step {step}/{total} ({pct}%)")

    def _on_success(self, res: GenerationResult):
        self.canvas.set_image(res.image_path)
        self._reset_ui()
        self.status_bar_lbl.setText(f"✓ Render complete in {res.generation_time_ms / 1000:.2f}s")
        self.progress_bar.setVisible(False)

    def _on_error(self, err: str):
        QMessageBox.critical(self, "Generation Error", err)
        self._reset_ui()
        self.status_bar_lbl.setText(f"⚠️ Error: {err}")
        self.progress_bar.setVisible(False)

    def _on_cancelled(self):
        self._reset_ui()
        self.status_bar_lbl.setText("Generation cancelled by user.")
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
        dest, _ = QFileDialog.getSaveFileName(self, "Save Output Image", "output.png", "PNG Images (*.png)")
        if dest:
            import shutil
            shutil.copy(self.canvas.current_image_path, dest)

    def _send_to_i2i(self):
        if self.canvas.current_image_path:
            self.send_to_i2i_requested.emit(self.canvas.current_image_path)
