from supervisor.renderer import Renderer
import os
from datetime import datetime

def get_session(logs_root, session_dir = None):
    if not os.path.exists(logs_root):
        return None
    if session_dir is not None and os.path.exists(os.path.join(logs_root, session_dir)):
        return os.path.join(logs_root, session_dir)

    def parse_session_timestamp(dir_name):
        try:
            return datetime.strptime(dir_name, "%Y-%m-%d_%H-%M")
        except ValueError:
            return None

    valid_sessions = []
    for entry in os.scandir(logs_root):
        if entry.is_dir() and parse_session_timestamp(entry.name):
            valid_sessions.append(entry.path)

    return max(valid_sessions, key=os.path.basename, default=None)

class PictureGenerator:
    def __init__(self, logs_root="logs", session_dir=None):
        self.logs_root = logs_root
        self.session_path = get_session(logs_root, session_dir)
        self.renderer = Renderer()

    def get_pictures_dir(self):
        out_dir = os.path.join(self.session_path, "pictures")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def run(self):
        raise NotImplementedError