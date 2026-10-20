import cv2

from communicator import Communicator
from .renderer import Renderer
from .picture_generator import get_session

class LiveMonitor(Renderer):
    def __init__(self, bus: Communicator, window_name:str = "window_live", timeout: float = 0.1, logs_root = "logs"):
        super().__init__()
        self.bus = bus
        self.timeout = timeout
        self.window_name = window_name
        self._window_created = False
        self.session_path = get_session(logs_root)
        self._last_mtime = None # modification time for loading from files

    def _ensure_window(self):
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self._window_created = True

    def _safe_call(self, topic: str):
        try:
            try:
                return self.bus.call_service(topic, timeout=self.timeout)
            except TypeError:
                return self.bus.call_service(topic)
        except Exception:
            return None

    def setup(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError