from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel, QPushButton
from core.prompt_enhancer import AIPromptEnhancer

class PromptEditorWidget(QGroupBox):
    """Prompt box with token estimation, clear button, and AI prompt enhancer."""
    def __init__(self, title: str = "PROMPT", default_text: str = "", parent=None):
        super().__init__(title, parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(6)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Describe your creation in detail...")
        self.text_edit.setText(default_text)
        self.text_edit.setMinimumHeight(70)
        self.text_edit.setMaximumHeight(110)
        layout.addWidget(self.text_edit)

        btn_layout = QHBoxLayout()
        self.token_label = QLabel("~0 tokens (0 chars)")
        self.token_label.setStyleSheet("font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #94A3B8;")
        btn_layout.addWidget(self.token_label)
        btn_layout.addStretch()

        self.enhance_btn = QPushButton("✨ AI Enhance")
        self.clear_btn = QPushButton("🧹 Clear")
        btn_layout.addWidget(self.enhance_btn)
        btn_layout.addWidget(self.clear_btn)
        layout.addLayout(btn_layout)

        self.text_edit.textChanged.connect(self._update_tokens)
        self.clear_btn.clicked.connect(self.text_edit.clear)
        self.enhance_btn.clicked.connect(self._enhance_prompt)
        self._update_tokens()

    def _update_tokens(self):
        text = self.text_edit.toPlainText()
        words = len(text.split()) if text else 0
        est_tokens = int(words * 1.3)
        self.token_label.setText(f"~{est_tokens} tokens ({len(text)} chars)")

    def _enhance_prompt(self):
        text = self.text_edit.toPlainText()
        if text.strip():
            enhanced = AIPromptEnhancer.enhance(text, architecture="sdxl")
            self.text_edit.setText(enhanced)

    def get_text(self) -> str:
        return self.text_edit.toPlainText().strip()

    def set_text(self, text: str):
        self.text_edit.setText(text)
