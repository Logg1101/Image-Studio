import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

import config.paths
from core.device import get_device_info
from ui.main_window import ImageStudioMainWindow

def main():
    # 1. Initialize filesystem directories
    config.paths.ensure_directories()

    # 2. Hardware check & console banner
    device_info = get_device_info()
    print("==================================================")
    print("  ImageStudio Professional AI Workstation Starting")
    print(f"  GPU: {device_info.name} ({device_info.vram_gb:.2f} GB VRAM)")
    print("==================================================")

    # 3. Create Qt Application
    app = QApplication(sys.argv)
    app.setApplicationName("ImageStudio")
    app.setOrganizationName("ImageStudio")

    # 4. Load Cyber-Dark Workstation Stylesheet
    qss_path = Path("ui/themes/cyber_dark.qss")
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    # 5. Launch Main Workstation Window
    window = ImageStudioMainWindow()
    window.show()

    # 6. Start Qt Event Loop
    sys.exit(app.exec())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: Application failed to start.\n{e}", file=sys.stderr)
        sys.exit(1)
