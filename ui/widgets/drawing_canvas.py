from typing import Optional
from PIL import Image, ImageDraw
import numpy as np

from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt, QPoint, QRect, Signal
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap, QImage, QMouseEvent

class DrawingCanvasWidget(QWidget):
    """
    Interactive drawing canvas allowing users to brush masks over loaded images
    for precision watermark/object removal via LaMa inpainting.
    """
    mask_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 400)

        self.base_pixmap: Optional[QPixmap] = None
        self.mask_pixmap: Optional[QPixmap] = None
        self.brush_size: int = 28
        self.drawing: bool = False
        self.last_point: Optional[QPoint] = None
        self.img_rect = QRect()
        self.original_image_path: Optional[str] = None

    def set_image(self, image_path: str):
        self.original_image_path = image_path
        self.base_pixmap = QPixmap(image_path)
        self.clear_mask()
        self.update()

    def clear_mask(self):
        if self.base_pixmap:
            self.mask_pixmap = QPixmap(self.base_pixmap.size())
            self.mask_pixmap.fill(Qt.transparent)
        self.update()
        self.mask_changed.emit()

    def set_brush_size(self, size: int):
        self.brush_size = max(4, size)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#090C12"))

        if not self.base_pixmap or self.base_pixmap.isNull():
            painter.setPen(QColor("#8993A7"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Drop or load an image to edit")
            return

        # Fit pixmap to widget area while maintaining aspect ratio
        scaled_base = self.base_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled_base.width()) // 2
        y = (self.height() - scaled_base.height()) // 2
        self.img_rect = QRect(x, y, scaled_base.width(), scaled_base.height())

        painter.drawPixmap(self.img_rect, scaled_base)

        if self.mask_pixmap and not self.mask_pixmap.isNull():
            scaled_mask = self.mask_pixmap.scaled(self.img_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(self.img_rect, scaled_mask)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.base_pixmap:
            self.drawing = True
            self.last_point = self._map_to_image_coords(event.position().toPoint())
            self._draw_stroke(self.last_point, self.last_point)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing and self.base_pixmap:
            curr_point = self._map_to_image_coords(event.position().toPoint())
            if self.last_point and curr_point:
                self._draw_stroke(self.last_point, curr_point)
                self.last_point = curr_point

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.last_point = None
            self.mask_changed.emit()

    def _map_to_image_coords(self, pos: QPoint) -> Optional[QPoint]:
        if not self.img_rect.contains(pos) or self.img_rect.width() == 0 or self.img_rect.height() == 0:
            return None
        rel_x = (pos.x() - self.img_rect.x()) / self.img_rect.width()
        rel_y = (pos.y() - self.img_rect.y()) / self.img_rect.height()
        orig_w = self.base_pixmap.width()
        orig_h = self.base_pixmap.height()
        return QPoint(int(rel_x * orig_w), int(rel_y * orig_h))

    def _draw_stroke(self, p1: Optional[QPoint], p2: Optional[QPoint]):
        if not p1 or not p2 or not self.mask_pixmap:
            return

        scale_factor = self.base_pixmap.width() / max(self.img_rect.width(), 1)
        actual_brush = int(self.brush_size * scale_factor)

        painter = QPainter(self.mask_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(239, 68, 68, 180), actual_brush, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(p1, p2)
        painter.end()
        self.update()

    def get_pil_image_and_mask(self):
        if not self.original_image_path or not self.base_pixmap:
            return None, None

        base_img = Image.open(self.original_image_path).convert("RGB")
        if not self.mask_pixmap:
            mask_img = Image.new("L", base_img.size, 0)
            return base_img, mask_img

        qimage = self.mask_pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.constBits()
        arr = np.array(ptr).reshape(height, width, 4)
        alpha_channel = arr[:, :, 3]
        mask_img = Image.fromarray((alpha_channel > 20).astype(np.uint8) * 255, mode="L")

        return base_img, mask_img
