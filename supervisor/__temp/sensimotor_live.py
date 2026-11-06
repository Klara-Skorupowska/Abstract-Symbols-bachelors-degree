from .live_monitor import LiveMonitor
import cv2
import numpy as np

class LiveSensimotorMonitor(LiveMonitor):
    def setup(self):
        pass

    def update(self):
        self._ensure_window()
        cv2.waitKey(1)

        lidar_readings = []
        for direction in self.LIDAR_ANGLES:
            val = self._safe_call(f"/lidar/{direction}/ask/value")
            if val is None:
                return
            lidar_readings.append(val)

        raw_coded = self._safe_call("/camera/ask/coded")
        if raw_coded is None:
            return
        cam_array = np.asarray(raw_coded, dtype=float).flatten()
        if len(cam_array) < (6 * 8 * 5):
            return

        speeds = self._safe_call("/wheels/ask/speed")
        max_vel = self._safe_call("/get/wheels/max_velocity")
        if speeds is None or max_vel is None:
            return

        img = self.create_base_canvas("LIVE SENSIMOTOR MONITOR")
        self.draw_lidar(img, lidar_readings, mode="live")
        self.draw_wheels(img, speeds[0], speeds[1], float(max_vel))
        self.draw_grids(img, cam_array, is_live=True)
        cv2.imshow(self.window_name, img)

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False
