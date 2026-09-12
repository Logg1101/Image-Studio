import os
from typing import Optional
from pathlib import Path
from PIL import Image, ImageEnhance

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
    QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox, QFileDialog,
    QTabWidget, QMessageBox, QProgressBar
)
from PySide6.QtCore import Qt, Signal

from ui.widgets.drawing_canvas import DrawingCanvasWidget
from engines.restoration import RestorationEngine
from engines.supir_enhancer import SupirEngine
import config.paths as paths

class ImageEditorPage(QWidget):
    send_to_t2i_requested = Signal(str)
    send_to_i2i_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.restoration_engine = RestorationEngine()
        self.supir_engine = SupirEngine()
        self.current_image_path: Optional[str] = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(10)

        # LEFT PANEL: Editing Tools (460px width, no scroll needed)
        left_panel = QWidget()
        left_panel.setFixedWidth(460)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # Top File Actions
        file_box = QGroupBox("IMAGE INPUT")
        f_layout = QHBoxLayout(file_box)
        self.load_btn = QPushButton("Open Image...")
        self.clear_btn = QPushButton("Reset Canvas")
        f_layout.addWidget(self.load_btn)
        f_layout.addWidget(self.clear_btn)
        left_layout.addWidget(file_box)

        # Tool Tabs
        self.tool_tabs = QTabWidget()

        # TAB 1: Watermark & Object Erase
        erase_tab = QWidget()
        e_layout = QVBoxLayout(erase_tab)
        e_layout.setSpacing(8)

        brush_row = QHBoxLayout()
        brush_row.addWidget(QLabel("Brush Size:"))
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(4, 128)
        self.brush_spin.setValue(28)
        brush_row.addWidget(self.brush_spin)
        e_layout.addLayout(brush_row)

        self.clear_mask_btn = QPushButton("Clear Inpaint Mask")
        e_layout.addWidget(self.clear_mask_btn)

        self.erase_btn = QPushButton("Erase Object / Watermark")
        self.erase_btn.setObjectName("btnAccent")
        e_layout.addWidget(self.erase_btn)
        e_layout.addStretch()
        self.tool_tabs.addTab(erase_tab, "Erase / Inpaint")

        # TAB 2: Background Removal
        bg_tab = QWidget()
        bg_layout = QVBoxLayout(bg_tab)
        bg_layout.setSpacing(8)
        bg_layout.addWidget(QLabel("Strips image background cleanly using U-2-Net."))
        self.remove_bg_btn = QPushButton("Remove Background")
        self.remove_bg_btn.setObjectName("btnAccent")
        bg_layout.addWidget(self.remove_bg_btn)
        bg_layout.addStretch()
        self.tool_tabs.addTab(bg_tab, "Background")

        # TAB 3: Color & Enhancement Settings
        adj_tab = QWidget()
        adj_layout = QVBoxLayout(adj_tab)
        adj_layout.setSpacing(6)

        # Sliders: Brightness, Contrast, Saturation, Sharpness
        self.bright_slider = self._create_slider_row(adj_layout, "Brightness (1.0 = Normal):", 1.0)
        self.contrast_slider = self._create_slider_row(adj_layout, "Contrast (1.0 = Normal):", 1.0)
        self.sat_slider = self._create_slider_row(adj_layout, "Color Saturation:", 1.0)
        self.sharp_slider = self._create_slider_row(adj_layout, "Sharpness Level:", 1.0)

        self.apply_adj_btn = QPushButton("Apply Color Adjustments")
        self.apply_adj_btn.setObjectName("btnAccent")
        adj_layout.addWidget(self.apply_adj_btn)
        adj_layout.addStretch()
        self.tool_tabs.addTab(adj_tab, "Color & Enhance")

        # TAB 4: Upscaler
        up_tab = QWidget()
        up_layout = QVBoxLayout(up_tab)
        up_layout.setSpacing(6)

        fast_row = QHBoxLayout()
        fast_row.addWidget(QLabel("Fast Upscale Factor:"))
        self.fast_scale_combo = QComboBox()
        self.fast_scale_combo.addItems(["1.5x", "2.0x", "3.0x", "4.0x"])
        self.fast_scale_combo.setCurrentIndex(1)
        fast_row.addWidget(self.fast_scale_combo)
        up_layout.addLayout(fast_row)

        self.fast_upscale_btn = QPushButton("Run High-Res Upscale")
        self.fast_upscale_btn.setObjectName("btnAccent")
        up_layout.addWidget(self.fast_upscale_btn)
        up_layout.addStretch()
        self.tool_tabs.addTab(up_tab, "Upscaler")

        left_layout.addWidget(self.tool_tabs)

        # Bottom Actions
        actions_box = QGroupBox("OUTPUT ACTIONS")
        act_layout = QVBoxLayout(actions_box)
        self.save_btn = QPushButton("Save Edited Image As...")
        self.send_t2i_btn = QPushButton("Send to Text Studio")
        self.send_i2i_btn = QPushButton("Send to Transformation Lab")
        act_layout.addWidget(self.save_btn)
        act_layout.addWidget(self.send_t2i_btn)
        act_layout.addWidget(self.send_i2i_btn)
        left_layout.addWidget(actions_box)

        main_layout.addWidget(left_panel, stretch=0)

        # RIGHT PANEL: Interactive Drawing Canvas
        self.canvas = DrawingCanvasWidget()
        main_layout.addWidget(self.canvas, stretch=1)

        # Event bindings
        self.load_btn.clicked.connect(self._open_image)
        self.clear_btn.clicked.connect(self._reset_canvas)
        self.brush_spin.valueChanged.connect(self.canvas.set_brush_size)
        self.clear_mask_btn.clicked.connect(self.canvas.clear_mask)
        self.erase_btn.clicked.connect(self._run_erase)
        self.remove_bg_btn.clicked.connect(self._run_remove_bg)
        self.apply_adj_btn.clicked.connect(self._run_adjustments)
        self.fast_upscale_btn.clicked.connect(self._run_fast_upscale)
        self.save_btn.clicked.connect(self._save_image)
        self.send_t2i_btn.clicked.connect(self._send_to_t2i)
        self.send_i2i_btn.clicked.connect(self._send_to_i2i)

    def _create_slider_row(self, layout: QVBoxLayout, label_text: str, default_val: float) -> QDoubleSpinBox:
        row = QHBoxLayout()
        row.addWidget(QLabel(label_text))
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 3.0)
        spin.setSingleStep(0.05)
        spin.setValue(default_val)
        spin.setFixedWidth(70)
        row.addWidget(spin)
        layout.addLayout(row)
        return spin

    def load_image(self, path: str):
        self.current_image_path = path
        self.canvas.set_image(path)

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image to Edit", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            self.load_image(path)

    def _reset_canvas(self):
        if self.current_image_path:
            self.canvas.set_image(self.current_image_path)

    def _run_erase(self):
        base_img, mask_img = self.canvas.get_pil_image_and_mask()
        if not base_img:
            QMessageBox.warning(self, "Warning", "Please open an image first.")
            return

        try:
            result = self.restoration_engine.remove_object(base_img, mask_img)
            out_path = str(paths.IMAGES_DIR / f"edited_{int(os.path.getmtime(self.current_image_path) if self.current_image_path else 100)}.png")
            result.save(out_path)
            self.load_image(out_path)
            QMessageBox.information(self, "Success", "Object / watermark erased successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Inpainting failed: {e}")

    def _run_remove_bg(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "Please open an image first.")
            return
        try:
            img = Image.open(self.current_image_path).convert("RGBA")
            result = self.restoration_engine.remove_background(img)
            out_path = str(paths.IMAGES_DIR / "nobg_output.png")
            result.save(out_path)
            self.load_image(out_path)
            QMessageBox.information(self, "Success", "Background removed successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Background removal failed: {e}")

    def _run_adjustments(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "Please open an image first.")
            return
        try:
            img = Image.open(self.current_image_path).convert("RGB")
            # Brightness
            if self.bright_slider.value() != 1.0:
                img = ImageEnhance.Brightness(img).enhance(self.bright_slider.value())
            # Contrast
            if self.contrast_slider.value() != 1.0:
                img = ImageEnhance.Contrast(img).enhance(self.contrast_slider.value())
            # Saturation (Color)
            if self.sat_slider.value() != 1.0:
                img = ImageEnhance.Color(img).enhance(self.sat_slider.value())
            # Sharpness
            if self.sharp_slider.value() != 1.0:
                img = ImageEnhance.Sharpness(img).enhance(self.sharp_slider.value())

            out_path = str(paths.IMAGES_DIR / "enhanced_output.png")
            img.save(out_path)
            self.load_image(out_path)
            QMessageBox.information(self, "Success", "Color adjustments applied.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Adjustments failed: {e}")

    def _run_fast_upscale(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "Please open an image first.")
            return
        factor_str = self.fast_scale_combo.currentText().replace("x", "")
        factor = float(factor_str)
        try:
            img = Image.open(self.current_image_path).convert("RGB")
            new_w = int(img.width * factor)
            new_h = int(img.height * factor)
            upscaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            out_path = str(paths.IMAGES_DIR / f"upscaled_{factor_str}x.png")
            upscaled.save(out_path)
            self.load_image(out_path)
            QMessageBox.information(self, "Success", f"Image upscaled {factor_str}x to {new_w}x{new_h}.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Upscaling failed: {e}")

    def _save_image(self):
        if not self.current_image_path:
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Save Edited Image", "edited_output.png", "PNG Images (*.png)")
        if dest:
            import shutil
            shutil.copy(self.current_image_path, dest)

    def _send_to_t2i(self):
        if self.current_image_path:
            self.send_to_t2i_requested.emit(self.current_image_path)

    def _send_to_i2i(self):
        if self.current_image_path:
            self.send_to_i2i_requested.emit(self.current_image_path)
