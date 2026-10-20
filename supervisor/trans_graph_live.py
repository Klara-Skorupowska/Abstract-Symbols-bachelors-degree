import cv2
import os
import json

from .live_monitor import LiveMonitor

class TransGraphLive(LiveMonitor):
    def setup(self):
        self.trans_filename = "trans_graph_PDDL.json"

    def update(self):
        self._ensure_window()
        cv2.waitKey(1)
        if not self.session_path:
            return
        trans_path = os.path.join(self.session_path, "graphs", self.trans_filename)

        try:
            if os.path.exists(trans_path) and os.path.getsize(trans_path) > 0:
                mtime = os.path.getmtime(trans_path)
                if self._last_mtime is None: self._last_mtime = mtime
                if mtime > self._last_mtime:
                    with open(trans_path, "r") as f:
                        graph_data = json.load(f)
                    img = self.create_base_canvas("LIVE TRANSITION GRAPH", "based on PDDL domain")
                    self.draw_transition_graph(img, graph_data)
                    cv2.imshow(self.window_name, img)
                    self._last_mtime = mtime
        except Exception as e:
            print(f"Error loading live graph: {e}")

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False
