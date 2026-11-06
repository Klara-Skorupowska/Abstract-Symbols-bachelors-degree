import os

class BaseLogger:
    def __init__(self, log_dir: str = None):
        self.log_dir = log_dir
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

    def get_path(self, filename: str) -> str:
        if not self.log_dir:
            return filename
        return os.path.join(self.log_dir, filename)
