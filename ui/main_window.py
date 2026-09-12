import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTabWidget, QStatusBar, QMessageBox, QApplication
)
from PySide6.QtCore import Qt

from ui.widgets.header_widget import HeaderTelemetryWidget
from ui.pages.text_studio_page import TextStudioPage
from ui.pages.multi_character_page import MultiCharacterPage
from ui.pages.transformation_page import TransformationPage
from ui.pages.image_editor_page import ImageEditorPage
from ui.pages.creative_vault_page import CreativeVaultPage
from ui.pages.project_usage_page import ProjectUsagePage
from core.types import sanitize_seed

class ImageStudioMainWindow(QMainWindow):
    """
    Production-grade AI Image Workstation Shell matching the reference screenshot.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ImageStudio — Generative AI & Transformation Suite")
        self.resize(1440, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # 1. Top Header with Logo & Telemetry Pills
        self.header_bar = HeaderTelemetryWidget()
        main_layout.addWidget(self.header_bar)

        # 2. Main Workspace Tabs
        self.tabs = QTabWidget()
        self.t2i_page = TextStudioPage()
        self.multi_char_page = MultiCharacterPage()
        self.adapter_page = TransformationPage()
        self.vault_page = CreativeVaultPage()
        self.alchemy_page = ImageEditorPage()
        self.usage_page = ProjectUsagePage()

        self.tabs.addTab(self.t2i_page, "⚡ Text Studio")
        self.tabs.addTab(self.multi_char_page, "👥 Multi-Character")
        self.tabs.addTab(self.adapter_page, "🪄 Adapter Lab")
        self.tabs.addTab(self.vault_page, "📁 Creative Vault")
        self.tabs.addTab(self.alchemy_page, "⚗️ Alchemy")
        self.tabs.addTab(self.usage_page, "📊 Usage & Projects")

        main_layout.addWidget(self.tabs)

        # 3. Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready | ImageStudio AI Workstation Active")

        # 4. Cross-Tab Signal Connections
        self.t2i_page.send_to_editor_requested.connect(self._on_send_to_alchemy)
        self.t2i_page.send_to_i2i_requested.connect(self._on_send_to_adapter)
        
        self.multi_char_page.send_to_editor_requested.connect(self._on_send_to_alchemy)
        self.multi_char_page.send_to_i2i_requested.connect(self._on_send_to_adapter)
        
        self.adapter_page.send_to_editor_requested.connect(self._on_send_to_alchemy)
        
        self.alchemy_page.send_to_t2i_requested.connect(self._on_send_to_t2i)
        self.alchemy_page.send_to_i2i_requested.connect(self._on_send_to_adapter)
        
        self.vault_page.send_to_t2i_requested.connect(self._on_reuse_params)
        self.vault_page.send_to_editor_requested.connect(self._on_send_to_alchemy)

        self.tabs.currentChanged.connect(self._on_tab_switched)

    def _on_tab_switched(self, index: int):
        if self.tabs.currentWidget() == self.vault_page:
            self.vault_page.scan_and_reload()
        elif self.tabs.currentWidget() == self.usage_page:
            self.usage_page.refresh_all()

    def _on_send_to_alchemy(self, img_path: str):
        self.alchemy_page.load_image(img_path)
        self.tabs.setCurrentWidget(self.alchemy_page)

    def _on_send_to_adapter(self, img_path: str):
        self.adapter_page.load_source_image(img_path)
        self.tabs.setCurrentWidget(self.adapter_page)

    def _on_send_to_t2i(self, img_path: str):
        self.tabs.setCurrentWidget(self.t2i_page)

    def _on_reuse_params(self, params: dict):
        self.t2i_page.prompt_text.setText(params.get("prompt", ""))
        self.t2i_page.neg_prompt_text.setText(params.get("negative_prompt", ""))
        self.t2i_page.width_card.setValue(params.get("width", 1024))
        self.t2i_page.height_card.setValue(params.get("height", 1024))
        self.t2i_page.steps_card.setValue(params.get("steps", 28))
        self.t2i_page.cfg_card.setValue(int(params.get("guidance_scale", 7.0)))
        self.tabs.setCurrentWidget(self.t2i_page)
