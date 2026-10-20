import os
import cv2

from .picture_generator import PictureGenerator

class UpdateFrequencyPictureGenerator(PictureGenerator):
    def __init__(self, logs_root="logs", session_dir=None, timestamp_file="timestamps.txt", bin_size_sec=60.0):
        super().__init__(logs_root, session_dir=session_dir)
        self.timestamp_file = timestamp_file
        self.bin_size_sec = bin_size_sec

    def _resolve_file_path(self):
        if self.session_path:
            sess_path = os.path.join(self.session_path, self.timestamp_file)
            if os.path.exists(sess_path):
                return sess_path
        if os.path.exists(self.timestamp_file):
            return self.timestamp_file
        return None

    def run(self):
        file_path = self._resolve_file_path()
        if not file_path:
            return
        try:
            with open(file_path, "r") as f:
                timestamps = [float(line.strip()) for line in f if line.strip()]
            img = self.renderer.create_base_canvas("UPDATE FREQUENCY OVER TIME")
            self.renderer.draw_frequency_plot(img, timestamps, bin_size_sec=self.bin_size_sec)
            self.renderer.draw_sample_badge(img, n_pos=len(timestamps), position=(35, 595))

            out_dir = self.get_pictures_dir() or "."
            out_path = os.path.join(out_dir, "update_frequency.png")
            cv2.imwrite(out_path, img)
            print(f"[PictureGenerator] Saved frequency plot to {out_path}")
        except Exception as e:
            print(f"[PictureGenerator] Failed to generate frequency plot: {e}")
