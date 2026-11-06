import os, cv2, numpy as np, re
from pathlib import Path
from datetime import datetime
from scipy.stats import gaussian_kde
from supervisor.core.picture_generator import PictureGenerator
from supervisor.utils.dir_utils import collect_chain_folders

class HistogramPlotGenerator(PictureGenerator):
    @staticmethod
    def parse_timestamp_line(line: str) -> float | None:
        line = line.strip()
        if not line: return None
        try: return float(line)
        except ValueError: pass
        try: return datetime.fromisoformat(line).timestamp()
        except ValueError: return None

    def load_session_timestamps(self, session_dir: Path) -> np.ndarray:
        ts_file = session_dir / 'timestamps.txt'
        if not ts_file.exists(): return np.array([], dtype=float)
        timestamps = [val for line in ts_file.open('r', encoding='utf-8') if (val := self.parse_timestamp_line(line)) is not None]
        return np.sort(np.array(timestamps, dtype=float)) if timestamps else np.array([], dtype=float)

    def run(self):
        if not self.session_path or not os.path.exists(self.session_path): return
        file_path = Path(self.session_path) / 'timestamps.txt'
        if not file_path.exists(): return
        try:
            chain_dirs = collect_chain_folders(Path(self.session_path), Path(self.logs_root))
            print(f"[HistogramPlotGenerator] Found {len(chain_dirs)} epochs for KDE!")
        except Exception as e:
            print(f"[HistogramPlotGenerator] Error: {e}")

class UpdateFrequencyPictureGenerator(PictureGenerator):
    def __init__(self, logs_root='logs', session_dir=None, timestamp_file='timestamps.txt', bin_size_sec=60.0):
        super().__init__(logs_root, session_dir)
        self.timestamp_file = timestamp_file
        self.bin_size_sec = bin_size_sec

    def run(self):
        pass
