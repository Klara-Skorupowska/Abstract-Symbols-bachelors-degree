# a parent class for virtual sensors

import cv2
import numpy as np

from communicator import Communicator


class VirtualSensor():
    def __init__(self, bus: Communicator):
        self.bus = bus

    def read(self): # from the bus
        raise NotImplementedError("The read() method must be implemented in the subclass.")

    def preprocess(self):
        raise NotImplementedError("The preprocess() method must be implemented in the subclass.")

class VirtualSensorArray(VirtualSensor):
    def __init__(self, bus: Communicator, sensors: list[VirtualSensor]):
        super().__init__(bus)
        self.sensors = sensors
        self.value = []
        self.coded = []

    def read(self):
        self.value = [sensor.read() for sensor in self.sensors]
        return self.value

    def preprocess(self):
        self.coded = [sensor.preprocess() for sensor in self.sensors]

class LidarSensor(VirtualSensor):
    def __init__(self, bus: Communicator, lidar_direction, max_range=0.1):
        '''
        lidar_direction: - in degrees, 0 is forward, 90 is left, 180 is backward, 270 is right
        '''
        super().__init__(bus)
        self.lidar_direction = lidar_direction
        self.max_range = max_range
        self.value = None
        self.coded = None

        self.bus.register_service(f"/lidar/{lidar_direction}/ask/value", self.send_value)

    def send_value(self, request=None):
        self.read()
        return self.value

    def read(self):
        self.value = self.bus.call_service(f"/sensor/lidar_{self.lidar_direction}/sense")
        return self.value

    def preprocess(self):
        """
        scale the value into 0-1
        """
        if  self.value is not None and self.max_range > 0:
            self.coded = self.value/self.max_range
        return self.coded

class CameraSensor(VirtualSensor):
    def __init__(self, bus: Communicator):
        '''
        Expected res (resolution) - [120, 160, 3]
        '''
        super().__init__(bus)
        self.frame = None 
        self.coded = None 

        # --- Dynamic Threshold Parameters (Weber's Law) ---
        self.base_threshold = 0.1   # Minimum contrast needed in absolute darkness to fire
        self.alpha = 0.01           # Sensitivity scaler: higher means it ignores more noise in bright light
        self.step = 0.01             # step, after (1/self.step) identical frames the threshold approaches the nominal value
        self.threshold = self.base_threshold  # Initial threshold for the first frame
        
        self.bus.register_service(f"/camera/ask/coded", self.send_coded)

    def send_coded(self, request=None):
        self.read()
        self.preprocess()
        return self.coded


    def read(self):
        self.frame = self.bus.call_service("/sensor/camera/sense")
        return self.frame

    def preprocess(self):
        '''
        Processes a uniform grid of receptive fields.
        Returns a flat array containing [ON-center, OFF-center, Mean R, Mean G, Mean B] per field.
        '''
        if self.frame is None:
            raise ValueError("No frame available. Please call read() before preprocess().")

        # Ensure frame is 120x160 for the grid mapping
        h, w = self.frame.shape[:2]
        if h != 120 or w != 160:
            frame = cv2.resize(self.frame, (160, 120))
        else:
            frame = self.frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Update self.threshold from the whole picture's mean luminance for the NEXT frame
        whole_frame_luminance = np.mean(gray) / 255.0
        new_threshold = (1-self.step) * self.threshold + self.step * self.alpha * whole_frame_luminance
        self.threshold = max(self.base_threshold, new_threshold)
        
        # Grid dimensions:
        patch_size = 20
        rows = 120 // patch_size  
        cols = 160 // patch_size  

        # Core size for receptive field
        core_s = patch_size // 3
        core_offset = (patch_size - core_s) // 2
        surround_pixels = (patch_size * patch_size) - (core_s * core_s)

        output = []
        for r in range(rows):
            for c in range(cols):
                y = r * patch_size
                x = c * patch_size

                bgr_roi = frame[y:y + patch_size, x:x + patch_size]
                gray_roi = gray[y:y + patch_size, x:x + patch_size]

                # Mean RGB normalized to [0.0, 1.0] (OpenCV uses BGR)
                mean_b = np.mean(bgr_roi[:, :, 0]) / 255.0
                mean_g = np.mean(bgr_roi[:, :, 1]) / 255.0
                mean_r = np.mean(bgr_roi[:, :, 2]) / 255.0

                # Center patch and surround calculation
                center_patch = gray_roi[core_offset:core_offset + core_s, core_offset:core_offset + core_s]
                mean_center = np.mean(center_patch) / 255.0

                total_sum = np.sum(gray_roi)
                center_sum = np.sum(center_patch)
                mean_surround = (total_sum - center_sum) / (surround_pixels * 255.0)

                # ON-center: Center (+) - Surround (-), using threshold from previous frame
                on_val = float(np.clip(mean_center - mean_surround, 0.0, 1.0))
                on_center = 1.0 if on_val > self.threshold else 0.0

                # OFF-center: Surround (+) - Center (-), using threshold from previous frame
                off_val = float(np.clip(mean_surround - mean_center, 0.0, 1.0))
                off_center = 1.0 if off_val > self.threshold else 0.0

                output.extend([on_center, off_center, mean_r, mean_g, mean_b])


        self.coded = np.array(output, dtype=float)
        return self.coded