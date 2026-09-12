from typing import Dict, Any

class ProgressTracker:
    def __init__(self):
        self.state: Dict[str, Any] = {
            'status': 'idle',
            'step': 0,
            'total': 30,
            'percent': 0,
            'message': 'Ready',
            'error': None,
            'result': None,
        }

    def update(self, step: int, total: int, message: str = ''):
        pct = int((step / max(total, 1)) * 100) if total > 0 else 0
        self.state['status'] = 'generating'
        self.state['step'] = step
        self.state['total'] = total
        self.state['percent'] = pct
        self.state['message'] = message or ('Sampling step ' + str(step) + '/' + str(total) + ' (' + str(pct) + '%)')

    def set_status(self, status: str, message: str = '', percent: int = 0):
        self.state['status'] = status
        self.state['message'] = message
        self.state['percent'] = percent

    def reset(self, total: int = 30):
        self.state['status'] = 'idle'
        self.state['step'] = 0
        self.state['total'] = total if total > 0 else 30
        self.state['percent'] = 0
        self.state['phase'] = 'idle'
        self.state['message'] = 'Ready'
        self.state['error'] = None
        self.state['result'] = None

    def get_progress(self) -> Dict[str, Any]:
        return self.state

progress_tracker = ProgressTracker()
