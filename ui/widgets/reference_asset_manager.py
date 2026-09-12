import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set
from PIL import Image

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QCheckBox, QFileDialog, QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QPixmap

class ReferenceAsset:
    """Represents a unified reference asset with consumer bindings."""
    def __init__(self, file_path: str, display_name: Optional[str] = None):
        self.id = str(uuid.uuid4())[:8]
        self.file_path = file_path
        self.path = Path(file_path)
        self.display_name = display_name or self.path.stem
        # Default consumers: Source, IP-Adapter, PuLID enabled for first upload
        self.consumers: Set[str] = {"source", "ip_adapter", "pulid"}
        self._pil_cache: Optional[Image.Image] = None

    def get_pil_image(self) -> Optional[Image.Image]:
        if not os.path.exists(self.file_path):
            return None
        try:
            if self._pil_cache is None:
                self._pil_cache = Image.open(self.file_path).convert("RGB")
            return self._pil_cache
        except Exception:
            return None


class ReferenceAssetCard(QFrame):
    """
    Card displaying a reference asset thumbnail, name, metadata,
    and consumer routing checkboxes (Source, IP-Adapter, PuLID, ControlNet).
    """
    removed = Signal(object)
    changed = Signal()

    def __init__(self, asset: ReferenceAsset, index: int = 1, parent=None):
        super().__init__(parent)
        self.asset = asset
        self.index = index
        self.setObjectName("cardPanel")
        self.setStyleSheet(
            "QFrame#cardPanel { background-color: #151A24; border: 1px solid #252C3A; border-radius: 10px; padding: 6px; } "
            "QFrame#cardPanel:hover { border-color: #353F54; }"
        )

        main_lay = QHBoxLayout(self)
        main_lay.setContentsMargins(6, 6, 6, 6)
        main_lay.setSpacing(10)

        # 1. Thumbnail Preview
        self.thumb_lbl = QLabel()
        self.thumb_lbl.setFixedSize(72, 72)
        self.thumb_lbl.setAlignment(Qt.AlignCenter)
        self.thumb_lbl.setStyleSheet(
            "background: #0F131B; border: 1px solid #252C3A; border-radius: 8px; overflow: hidden;"
        )
        pix = QPixmap(self.asset.file_path)
        if not pix.isNull():
            scaled = pix.scaled(72, 72, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.thumb_lbl.setPixmap(scaled)
        else:
            self.thumb_lbl.setText("No Preview")
            self.thumb_lbl.setStyleSheet("color: #8993A7; font-size: 9px;")
        main_lay.addWidget(self.thumb_lbl)

        # 2. Details & Consumer Checkboxes
        details_lay = QVBoxLayout()
        details_lay.setContentsMargins(0, 0, 0, 0)
        details_lay.setSpacing(4)

        # Title Row
        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        tag_lbl = QLabel(f"Ref #{self.index}")
        tag_lbl.setStyleSheet(
            "background: #7C6CFF; color: #FFFFFF; font-weight: 800; font-size: 10px; "
            "border-radius: 4px; padding: 1px 6px; font-family: 'JetBrains Mono', monospace;"
        )
        name_lbl = QLabel(self.asset.display_name)
        name_lbl.setStyleSheet("font-weight: 700; font-size: 12px; color: #E8ECF4;")
        title_row.addWidget(tag_lbl)
        title_row.addWidget(name_lbl, stretch=1)
        details_lay.addLayout(title_row)

        # Consumer Routing Checkboxes Row
        used_lbl = QLabel("Used By Subsystems:")
        used_lbl.setStyleSheet("color: #8993A7; font-size: 10px; font-weight: 600; text-transform: uppercase;")
        details_lay.addWidget(used_lbl)

        chk_row = QHBoxLayout()
        chk_row.setSpacing(12)

        self.chk_source = QCheckBox("Source (Img2Img)")
        self.chk_ip = QCheckBox("IP-Adapter")
        self.chk_pulid = QCheckBox("PuLID (FaceID)")
        self.chk_cnet = QCheckBox("ControlNet")

        chk_style = "QCheckBox { font-size: 11px; color: #E8ECF4; } QCheckBox::indicator { width: 14px; height: 14px; }"
        for chk in (self.chk_source, self.chk_ip, self.chk_pulid, self.chk_cnet):
            chk.setStyleSheet(chk_style)

        self.chk_source.setChecked("source" in self.asset.consumers)
        self.chk_ip.setChecked("ip_adapter" in self.asset.consumers)
        self.chk_pulid.setChecked("pulid" in self.asset.consumers)
        self.chk_cnet.setChecked("controlnet" in self.asset.consumers)

        chk_row.addWidget(self.chk_source)
        chk_row.addWidget(self.chk_ip)
        chk_row.addWidget(self.chk_pulid)
        chk_row.addWidget(self.chk_cnet)
        chk_row.addStretch()
        details_lay.addLayout(chk_row)

        main_lay.addLayout(details_lay, stretch=1)

        # 3. Remove Button
        self.del_btn = QPushButton("✕")
        self.del_btn.setFixedSize(22, 22)
        self.del_btn.setCursor(Qt.PointingHandCursor)
        self.del_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; color: #8993A7; font-size: 11px; font-weight: bold; } "
            "QPushButton:hover { color: #FF5C6C; }"
        )
        self.del_btn.clicked.connect(lambda: self.removed.emit(self))
        main_lay.addWidget(self.del_btn)

        # Connections
        self.chk_source.toggled.connect(self._sync_consumers)
        self.chk_ip.toggled.connect(self._sync_consumers)
        self.chk_pulid.toggled.connect(self._sync_consumers)
        self.chk_cnet.toggled.connect(self._sync_consumers)

    def _sync_consumers(self):
        consumers: Set[str] = set()
        if self.chk_source.isChecked():
            consumers.add("source")
        if self.chk_ip.isChecked():
            consumers.add("ip_adapter")
        if self.chk_pulid.isChecked():
            consumers.add("pulid")
        if self.chk_cnet.isChecked():
            consumers.add("controlnet")
        self.asset.consumers = consumers
        self.changed.emit()


class ReferenceAssetManagerWidget(QWidget):
    """
    Unified Reference Asset Shelf.
    Supports importing an image once and dynamically routing it to multiple subsystems.
    """
    references_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.assets: List[ReferenceAsset] = []
        self.cards: List[ReferenceAssetCard] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Header Row
        hdr_row = QHBoxLayout()
        hdr = QLabel("🖼️ REFERENCE ASSETS")
        hdr.setStyleSheet("color: #8993A7; font-weight: 800; font-size: 11px; letter-spacing: 0.5px;")
        hdr_row.addWidget(hdr)
        hdr_row.addStretch()

        self.add_btn = QPushButton("+ Add Reference")
        self.add_btn.setStyleSheet(
            "QPushButton { background: #151A24; border: 1px solid #252C3A; border-radius: 6px; color: #35D6C5; font-weight: 700; font-size: 11px; padding: 4px 10px; }"
            "QPushButton:hover { background: #1D2433; border-color: #35D6C5; color: #FFFFFF; }"
        )
        self.add_btn.clicked.connect(self._browse_and_add)
        hdr_row.addWidget(self.add_btn)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setStyleSheet(
            "QPushButton { background: transparent; border: 1px solid #252C3A; border-radius: 6px; color: #8993A7; font-size: 11px; padding: 4px 8px; }"
            "QPushButton:hover { color: #FF5C6C; border-color: #FF5C6C; }"
        )
        self.clear_btn.clicked.connect(self.clear_all)
        hdr_row.addWidget(self.clear_btn)

        layout.addLayout(hdr_row)

        # Cards Container
        self.stack_layout = QVBoxLayout()
        self.stack_layout.setSpacing(6)
        layout.addLayout(self.stack_layout)

        # Empty State Drop Zone
        self.empty_zone = QLabel("No reference images loaded.\nClick '+ Add Reference' to upload once and route to IP-Adapter, PuLID, or ControlNet.")
        self.empty_zone.setAlignment(Qt.AlignCenter)
        self.empty_zone.setFixedHeight(75)
        self.empty_zone.setStyleSheet(
            "background: #151A24; border: 1px dashed #252C3A; border-radius: 10px; "
            "color: #8993A7; font-size: 11px; padding: 8px;"
        )
        self.empty_zone.setCursor(Qt.PointingHandCursor)
        self.empty_zone.mousePressEvent = lambda e: self._browse_and_add()
        layout.addWidget(self.empty_zone)

        self._update_empty_state()

    def _browse_and_add(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Image", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if path:
            self.add_reference(path)

    def add_reference(self, file_path: str, display_name: Optional[str] = None, consumers: Optional[Set[str]] = None) -> ReferenceAsset:
        # Check if already in assets
        resolved = str(Path(file_path).resolve())
        for a in self.assets:
            if str(Path(a.file_path).resolve()) == resolved:
                if consumers:
                    a.consumers.update(consumers)
                self._rebuild_cards()
                self.references_changed.emit()
                return a

        asset = ReferenceAsset(resolved, display_name)
        if consumers is not None:
            asset.consumers = consumers
        self.assets.append(asset)
        self._rebuild_cards()
        self.references_changed.emit()
        return asset

    def _remove_card(self, card: ReferenceAssetCard):
        if card.asset in self.assets:
            self.assets.remove(card.asset)
        self._rebuild_cards()
        self.references_changed.emit()

    def clear_all(self):
        self.assets.clear()
        self._rebuild_cards()
        self.references_changed.emit()

    def _rebuild_cards(self):
        # Clear existing card widgets
        for c in self.cards:
            self.stack_layout.removeWidget(c)
            c.deleteLater()
        self.cards.clear()

        for idx, asset in enumerate(self.assets, start=1):
            card = ReferenceAssetCard(asset, index=idx)
            card.removed.connect(self._remove_card)
            card.changed.connect(self.references_changed.emit)
            self.cards.append(card)
            self.stack_layout.addWidget(card)

        self._update_empty_state()

    def _update_empty_state(self):
        self.empty_zone.setVisible(len(self.assets) == 0)
        self.clear_btn.setVisible(len(self.assets) > 0)

    # Convenience queries for downstream adapters
    def get_source_asset(self) -> Optional[ReferenceAsset]:
        for a in self.assets:
            if "source" in a.consumers:
                return a
        return self.assets[0] if self.assets else None

    def get_ip_adapter_asset(self) -> Optional[ReferenceAsset]:
        for a in self.assets:
            if "ip_adapter" in a.consumers:
                return a
        return None

    def get_pulid_asset(self) -> Optional[ReferenceAsset]:
        for a in self.assets:
            if "pulid" in a.consumers:
                return a
        return None

    def get_controlnet_asset(self) -> Optional[ReferenceAsset]:
        for a in self.assets:
            if "controlnet" in a.consumers:
                return a
        return None

    def get_source_pil_image(self) -> Optional[Image.Image]:
        a = self.get_source_asset()
        return a.get_pil_image() if a else None

    def get_ip_adapter_pil_image(self) -> Optional[Image.Image]:
        a = self.get_ip_adapter_asset()
        return a.get_pil_image() if a else None

    def get_pulid_pil_image(self) -> Optional[Image.Image]:
        a = self.get_pulid_asset()
        return a.get_pil_image() if a else None

    def get_controlnet_pil_image(self) -> Optional[Image.Image]:
        a = self.get_controlnet_asset()
        return a.get_pil_image() if a else None
