import os
from supervisor.utils.core_utils import get_session
from supervisor.core.renderer import Renderer

class PictureGenerator:
    def __init__(self, logs_root='logs', session_dir=None):
        self.logs_root = logs_root
        self.session_path = get_session(logs_root, session_dir)
        self.renderer = Renderer()

    def get_pictures_dir(self):
        out_dir = os.path.join(self.session_path, 'pictures')
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def run(self):
        raise NotImplementedError
