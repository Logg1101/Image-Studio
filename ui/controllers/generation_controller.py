from PySide6.QtCore import QThread, Signal
from core.types import GenerationRequest, GenerationResult
from ui.state import coordinator

class GenerationWorker(QThread):
    """Non-blocking background generation thread."""
    progress_changed = Signal(int, int, str)
    finished_success = Signal(object)
    finished_error = Signal(str)
    cancelled = Signal()

    def __init__(self, request: GenerationRequest, parent=None):
        super().__init__(parent)
        self.request = request
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        def step_callback(step: int, total: int):
            if self._is_cancelled:
                raise RuntimeError("Generation cancelled by user.")
            self.progress_changed.emit(step, total, f"Rendering step {step}/{total}")

        self.request.step_callback = step_callback
        try:
            self.progress_changed.emit(0, self.request.steps, "Initializing pipeline & synchronizing adapters...")
            result = coordinator.generate(self.request)
            if self._is_cancelled:
                self.cancelled.emit()
            else:
                self.finished_success.emit(result)
        except Exception as e:
            if self._is_cancelled:
                self.cancelled.emit()
            else:
                self.finished_error.emit(str(e))
