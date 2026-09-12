import psutil
import torch
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame
from PySide6.QtCore import Qt, QTimer

class HeaderTelemetryWidget(QWidget):
    """
    Header bar with ImageStudio branding and quiet, semantic system telemetry.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 4, 8, 8)
        main_layout.setSpacing(12)

        # Left Branding
        brand_layout = QHBoxLayout()
        brand_layout.setSpacing(8)

        icon_lbl = QLabel("⚡")
        icon_lbl.setStyleSheet("font-size: 20px; color: #F59E0B;")
        brand_layout.addWidget(icon_lbl)

        text_box = QVBoxLayout()
        text_box.setSpacing(1)
        title_lbl = QLabel("ImageStudio")
        title_lbl.setStyleSheet("font-size: 15px; font-weight: 800; color: #E8ECF4; letter-spacing: 0.5px;")
        sub_lbl = QLabel("Professional AI Creative Workstation")
        sub_lbl.setStyleSheet("font-size: 10px; font-weight: 500; color: #8993A7; letter-spacing: 0.2px;")
        text_box.addWidget(title_lbl)
        text_box.addWidget(sub_lbl)
        brand_layout.addLayout(text_box)

        main_layout.addLayout(brand_layout)
        main_layout.addStretch()

        # Right Telemetry Container
        telemetry_frame = QFrame()
        telemetry_frame.setStyleSheet(
            "background: #151A24; border: 1px solid #252C3A; border-radius: 10px; padding: 2px 6px;"
        )
        telemetry_layout = QHBoxLayout(telemetry_frame)
        telemetry_layout.setContentsMargins(6, 2, 6, 2)
        telemetry_layout.setSpacing(10)

        self.gpu_pill = QLabel("● GPU")
        self.vram_pill = QLabel("VRAM: --/-- GB")
        self.ram_pill = QLabel("RAM: --/-- GB")
        self.cpu_pill = QLabel("CPU: --%")

        pill_base = "font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; background: transparent; border: none;"
        self.gpu_pill.setStyleSheet(pill_base + " color: #10B981;")
        self.vram_pill.setStyleSheet(pill_base + " color: #4F9CFF;")
        self.ram_pill.setStyleSheet(pill_base + " color: #8993A7;")
        self.cpu_pill.setStyleSheet(pill_base + " color: #8993A7;")

        telemetry_layout.addWidget(self.gpu_pill)
        telemetry_layout.addWidget(self.vram_pill)
        telemetry_layout.addWidget(self.ram_pill)
        telemetry_layout.addWidget(self.cpu_pill)

        main_layout.addWidget(telemetry_frame)

        # Telemetry update timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(2000)
        self.update_stats()

    def update_stats(self):
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        self.cpu_pill.setText(f"CPU: {cpu:.0f}%")
        self.ram_pill.setText(f"RAM: {ram.used / (1024**3):.1f}/{ram.total / (1024**3):.1f} GB")

        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "")
            alloc = torch.cuda.memory_allocated(0) / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.gpu_pill.setText(f"● {dev_name}")
            self.vram_pill.setText(f"VRAM: {alloc:.1f}/{total:.1f} GB")
        else:
            self.gpu_pill.setText("● CPU Mode")
            self.vram_pill.setText("VRAM: N/A")
