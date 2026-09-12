import psutil
import torch
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PySide6.QtCore import QTimer

class HardwareMonitorWidget(QWidget):
    """Real-time hardware status indicator showing GPU, VRAM, RAM, and CPU usage."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(8)

        self.gpu_label = QLabel("GPU: Detecting...")
        self.vram_label = QLabel("VRAM: --/-- GB")
        self.ram_label = QLabel("RAM: --/-- GB")
        self.cpu_label = QLabel("CPU: --%")

        for lbl in [self.gpu_label, self.vram_label, self.ram_label, self.cpu_label]:
            lbl.setStyleSheet("background: #151D2C; border: 1px solid #243046; border-radius: 6px; padding: 4px 8px; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #E2E8F0;")
            layout.addWidget(lbl)

        self.vram_label.setStyleSheet(self.vram_label.styleSheet() + " color: #38BDF8; font-weight: bold;")
        self.gpu_label.setStyleSheet(self.gpu_label.styleSheet() + " color: #10B981; font-weight: bold;")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(2000)
        self.update_stats()

    def update_stats(self):
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        self.cpu_label.setText(f"CPU: {cpu:.0f}%")
        self.ram_label.setText(f"RAM: {ram.used / (1024**3):.1f}/{ram.total / (1024**3):.1f} GB")

        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "")
            alloc = torch.cuda.memory_allocated(0) / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.gpu_label.setText(f"● {dev_name}")
            self.vram_label.setText(f"VRAM: {alloc:.1f}/{total:.1f} GB")
        else:
            self.gpu_label.setText("CPU Mode")
            self.vram_label.setText("VRAM: N/A")
