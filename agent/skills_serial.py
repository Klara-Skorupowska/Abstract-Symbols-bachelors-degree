from turtle import goto

from .skills import *
from communicator import Communicator

import time
import cv2
import numpy as np

### all skills required to achive the goal, in order ###
# if sensors are not enough, the supervisor decides if the skill is succesfull
# 1) spot the door
# 2) go to the door
# 3) go through the door
# 4) spot the finish pad
# 5) go to the finish pad
# 6) Finish

class SerialSkill(Skill):
    '''
    mother class for all skills executed in order,
    contains all sensors, helper with reading them, and basic initialization
    '''
    def __init__(
        self,
        camera: CameraSensor,
        lidars: VirtualSensorArray,
        wheels: WheelsActuator,
        velocity: float,
        save_dist: float,
        timeout: float = 5.0,
    ):
        super().__init__()

        self.camera = camera
        self.lidars = lidars
        self.wheels = wheels
        self.velocity = velocity
        self.save_dist = save_dist
        self.timeout = timeout

    def read_sensors(self):
        frame = self.camera.read()
        distances = self.lidars.read()
        return frame, distances

class SpotTheColor(SerialSkill):
    """
    Turns around until it sees the <color> object, centers it,
    and returns True. Fails if an obstacle is too close or timeout is reached.
    """

    def __init__(
        self,
        hue: int,
        camera: CameraSensor,
        lidars: VirtualSensorArray,
        wheels: WheelsActuator,
        velocity: float,
        save_dist: float,
        timeout: float = 5.0,
    ):
        super().__init__(camera, lidars, wheels, velocity, save_dist, timeout)
        assert 0 <= hue <= 179, f"Hue {hue} out of bounds (0-179)"

        self.hue = hue

    def execute(self) -> bool:
        class_name = self.__class__.__name__
        print(f"\t[Skill] Executing {class_name} skill. Hue: {self.hue}.")

        hue_tol = 5
        lower_hue = (self.hue - hue_tol)%180
        higher_hue = (self.hue + hue_tol)%180
        if lower_hue > higher_hue:
            temp = lower_hue
            lower_hue = higher_hue
            higher_hue = temp
        lower_HSV = np.array([lower_hue, 50, 50])
        upper_HSV = np.array([higher_hue, 255, 255])
        _, distances = self.read_sensors()
        to_the_left = distances[0] < distances[7]

        start_time = time.time()
        while time.time() - start_time < self.timeout:

            # 1. Camera Snapshot & HSV Detection
            frame, distances = self.read_sensors()
            if frame is None:
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_HSV, upper_HSV)

            # 2. Locate Centroid
            moments = cv2.moments(mask)
            frame_center_x = frame.shape[1] // 2

            if moments["m00"] > 100:  # filtering noise
                color_center_x = moments["m10"]//moments["m00"]
                if (to_the_left and  color_center_x > frame_center_x) or (not to_the_left and color_center_x < frame_center_x):
                    self.wheels.set_parameters([0.0, 0.0])
                    print(f"\t[Skill] {self.__class__.__name__} finished successfully.")
                    return True
            # Not found nor centered: continue searching by rotating
            if to_the_left:
                self.wheels.set_parameters([ -self.velocity, self.velocity])
            else:
                self.wheels.set_parameters([ self.velocity, -self.velocity])

            time.sleep(0.01)

        # Timeout reached
        self.wheels.set_parameters([0.0, 0.0])
        print(f"\t[Skill] {class_name} failed. Timed out.")
        return False

class ClearThePath(SerialSkill):
    '''
    Clears the path in front by turning and running away.
    '''
    def __init__(
        self,
        bus: Communicator,
        camera, 
        lidars, 
        wheels, 
        velocity: float, 
        save_dist: float,
        run_time: float,
        timeout: float = 15.0
    ):
        super().__init__(camera, lidars, wheels, velocity, save_dist, timeout)
        self.bus = bus

    def execute(self) -> bool:
        print(f"\t[Skill] Executing {self.__class__.__name__} skill.")
        start_time = time.time()
        tolerance = 0.005
        # helpers
        def turn() -> bool:
            _, distances = self.read_sensors()
            distance_left = distances[0]
            distance_right = distances[7]
            if (( distances[0] > self.save_dist + tolerance ) and ( distances[7] > self.save_dist + tolerance )):
                return True
            if distance_left < distance_right:
                self.wheels.set_parameters([ - self.velocity, self.velocity])
            else:
                self.wheels.set_parameters([self.velocity, - self.velocity])
            while time.time() - start_time < self.timeout:
                _, distances = self.read_sensors()
                # turn until minimal value is on indexes 3 and 4
                min_idx = np.argmin(distances)
                clear_path = (( distances[0] > self.save_dist + tolerance ) and ( distances[7] > self.save_dist + tolerance ))
                if min_idx in [3, 4] and clear_path:
                    return True
            self.wheels.set_parameters([0, 0])
            print(f"\t[Skill] {self.__class__.__name__} failed. Timed out. TURN")
            return False

        def move() -> bool:
            start_run = time.time()
            self.wheels.set_parameters([self.velocity, self.velocity])
            while time.time() - start_run < self.run_time:
                # sensors
                _, distances = self.read_sensors()
                if time.time() - start_time > self.timeout:
                    self.wheels.set_parameters([0, 0])
                    print(f"\t[Skill] {self.__class__.__name__} failed. Timed out. MOVE")
                    return False
                clear_path = (( distances[0] > self.save_dist + tolerance ) and ( distances[7] > self.save_dist + tolerance ))
                if not clear_path:
                    return True if turn() else False
            return True


        # loops:::
        # sensors
        _, distances = self.read_sensors()
        # calculations
        ## all obstacles = bad
        if max(distances) < self.save_dist + tolerance:
            self.wheels.set_parameters([0, 0])
            print(f"\t[Skill] {self.__class__.__name__} failed. Nowhere to run.")
            return False
        ## find clear path:
        if turn() and move():
            self.wheels.set_parameters([0,0])
            print(f"\t[Skill] {self.__class__.__name__} finished successfully.")
            return True
        ## failed somewhere
        self.wheels.set_parameters([0, 0])
        print(f"\t[Skill] {self.__class__.__name__} failed.")
        return False

class GoToTheColor(SerialSkill):
    '''
    Approaches a specific colored object by Hue.
    '''
    def __init__(
        self,
        bus: Communicator,
        hue: int,
        camera, 
        lidars, 
        wheels, 
        velocity: float, 
        save_dist: float, 
        target_distance: float = 0.09,
        distance_tolerance: float = 0.005,
        timeout: float = 15.0
    ):
        super().__init__(camera, lidars, wheels, velocity, save_dist, timeout)
        self.bus = bus
        self.hue = hue
        self.target_distance = target_distance
        self.distance_tolerance = distance_tolerance
        self.center_tolerance = 10
        self.hue_tolerance = 5


    def _get_color_x_center(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        # 2. Handle Hue wrapping (0-179 range in OpenCV)
        lower1, upper1 = max(0, self.hue - self.hue_tolerance), min(179, self.hue + self.hue_tolerance)
        mask = cv2.inRange(hsv, np.array([lower1, 50, 50]), np.array([upper1, 255, 255]))
    
        # Wrap around 0 (e.g., target hue is 5, we need 175-179 as well)
        if self.hue - self.hue_tolerance < 0:
            lower2 = 180 + (self.hue - self.hue_tolerance)
            mask2 = cv2.inRange(hsv, np.array([lower2, 50, 50]), np.array([179, 255, 255]))
            mask = cv2.bitwise_or(mask, mask2)
        
        # Wrap around 179 (e.g., target hue is 175, we need 0-5 as well)
        elif self.hue + self.hue_tolerance > 179:
            upper2 = (self.hue + self.hue_tolerance) - 180
            mask2 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([upper2, 255, 255]))
            mask = cv2.bitwise_or(mask, mask2)
    
        # Find the center of the colored object
        M = cv2.moments(mask)
        if M["m00"] == 0:
            return None
        return int(M["m10"] / M["m00"])
    
    def conditions_check(self) -> bool:
        raise NotImplementedError(f"{self.__class__.__name__} needs checks implementations. No conditions_check")

    def effects_check(self) -> bool:
        raise NotImplementedError(f"{self.__class__.__name__} needs checks implementations. No effects_check")

    def execute(self) -> bool:
        print(f"\t[Skill] Executing {self.__class__.__name__} skill. Hue: {self.hue}.")
        start_time = time.time()
        # read sensors
        frame, distances = self.read_sensors()
        if frame is None: 
            return False
        # ask supervisor
        if not self.conditions_check():
            self.wheels.set_parameters([0, 0])
            print(f"\t[Skill] {self.__class__.__name__} failed. Starting conditions not met.")
            return False
        # make calculations
        image_center = frame.shape[1] / 2
        color_center = self._get_color_x_center(frame)
        if color_center is None:
            self.wheels.set_parameters([0, 0])
            print(f"\t[Skill] {self.__class__.__name__} failed. No target color in sight.")
            return False
        # loop
        while time.time() - start_time < self.timeout:
            # read sensors
            frame, distances = self.read_sensors()
            if frame is None: 
                continue
            # make calculations
            image_center = frame.shape[1] / 2
            color_center = self._get_color_x_center(frame)
            if color_center is None:
                self.wheels.set_parameters([0, 0])
                print(f"\t[Skill] {self.__class__.__name__} failed. No target color in sight.")
                return False
            front_dist = min(distances[0], distances[7])

            # check if we succeed
            at_target_distance = ( abs(front_dist - self.target_distance) < self.distance_tolerance )
            centered = ( abs(color_center - image_center) < self.center_tolerance )
            
            success = at_target_distance and centered
            
            if success:
                self.wheels.set_parameters([0, 0])
                # ask supervisor
                if not self.effects_check():
                    print(f"\t[Skill] {self.__class__.__name__} failed. Effects conditions not met.")
                    return False
                print(f"\t[Skill] {self.__class__.__name__} finished successfully.")
                return True

            # check if we can move freely
            clear_path = ( distances[0] >= self.save_dist and distances[7] >= self.save_dist )
            if not clear_path:
                self.wheels.set_parameters([0, 0])
                print(f"\t[Skill] {self.__class__.__name__} failed. Obstacle ahead.")
                return False

            # 2 bools so 4 cases but 2 actions
            def centering():
                if color_center < image_center:
                    self.wheels.set_parameters([self.velocity * 0.1, self.velocity])
                elif color_center > image_center:
                    self.wheels.set_parameters([self.velocity, self.velocity * 0.1])
                time.sleep(0.01)
                
            def aproaching():
                if front_dist < self.target_distance:
                    self.wheels.set_parameters([ - self.velocity, - self.velocity])
                elif front_dist > self.target_distance:
                    self.wheels.set_parameters([self.velocity, self.velocity])
                time.sleep(0.01)


            if not at_target_distance:
                if centered:
                    aproaching()
                    continue
                
            centering()

        self.wheels.set_parameters([0, 0])
        print(f"\t[Skill] {self.__class__.__name__} failed. Timed out.")
        return False

class GoToTheDoor(GoToTheColor):
    def conditions_check(self) -> bool:
        return True

    def effects_check(self) -> bool:
        door_zone = self.bus.call_service(f"/supervisor/ask/door_zone")
        return door_zone

class GoToTheGoal(GoToTheColor):
    def conditions_check(self) -> bool:
        if self.bus.call_service(f"/supervisor/ask/room_number") == 2:
            return True
        return False

    def effects_check(self) -> bool:
        return self.bus.call_service(f"/supervisor/ask/goal_zone")

class GoThroughTheDoor(SerialSkill):
    '''
    Opens door, any color.
    '''
    def __init__(
        self,
        bus: Communicator,
        camera: CameraSensor, 
        lidars: VirtualSensorArray, 
        wheels: VirtualActuator, 
        velocity: float, 
        save_dist: float,
        timeout: float = 15.0
    ):
        super().__init__(camera, lidars, wheels, velocity, save_dist, timeout)
        self.bus = bus

    def execute(self)-> bool:
        print(f"\t[Skill] Executing {self.__class__.__name__} skill")
        # Ask supervisor if we can start:
        if not self.bus.call_service(f"/supervisor/ask/door_zone"):
            self.wheels.set_parameters([0, 0])
            print(f"\t[Skill] {self.__class__.__name__} failed. Not in the door zone.")
            return False
        init_room = self.bus.call_service(f"/supervisor/ask/room_number")
        # Start moving forward
        self.wheels.set_parameters([self.velocity, self.velocity])
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            # sensors
            _, distances = self.read_sensors()
            room = self.bus.call_service(f"/supervisor/ask/room_number")
            zone = self.bus.call_service(f"/supervisor/ask/door_zone")
            # calculations
            open_space = min(distances) >= self.save_dist
            # logic
            if not room == init_room and not zone and open_space:
                self.wheels.set_parameters([0.0, 0.0])
                if room == 1: 
                    print(f"\t[Skill] {self.__class__.__name__} failed. Ended in the wrong room.")
                    return False
                print(f"\t[Skill] {self.__class__.__name__} finished successfully.")
                return True 
            time.sleep(0.05)
            
        # failed, back up or turn
        def backup()-> bool:
            _, distances = self.read_sensors()
            front = min(distances[0], distances[1], distances[6], distances[7])
            clear_path = front > self.save_dist
            self.wheels.set_parameters([-self.velocity, -self.velocity])
            while not clear_path:
                _, distances = self.read_sensors() # sensors (!!!)
                front = min(distances[0], distances[1], distances[6], distances[7])
                clear_path = front > self.save_dist
                if time.time() - start_time > 1.5 * self.timeout:
                    return False
                return True
        def turn()-> bool:
            _, distances = self.read_sensors()
            front = min(distances[0], distances[1], distances[6], distances[7])
            clear_path = front > self.save_dist
            if distances[0] < distances[7]:
                self.wheels.set_parameters([self.velocity, -self.velocity])
            else:
                self.wheels.set_parameters([-self.velocity, self.velocity])
            while not clear_path:
                _, distances = self.read_sensors()
                front = min(distances[0], distances[1], distances[6], distances[7])
                clear_path = front > self.save_dist
                if time.time() - start_time > 1.5 * self.timeout:
                    return False
            return True
        if not backup() or not turn():
                self.wheels.set_parameters([0.0, 0.0])
                print("\t[Skill] Cannot clear the path. Restart.")
                self.bus.call_service(f"/supervisor/do/restart")
                return False
        
        self.wheels.set_parameters([0.0, 0.0])
        print(f"\t[Skill] {self.__class__.__name__} failed. Timed out.")
        return False 

class Finish(SerialSkill):
    '''
    Checks if it is at goal zone and facing the goal then teleports agent back (restart)
    '''
    def __init__(
        self,
        bus: Communicator,
        hue: int,
        camera: CameraSensor, 
        lidars: VirtualSensorArray, 
        wheels: VirtualActuator, 
        velocity: float, 
        save_dist: float,
        timeout: float = 15.0
    ):
        super().__init__(camera, lidars, wheels, velocity, save_dist, timeout)
        self.bus = bus
        self.hue = hue

    def execute(self):
        print(f"\t[Skill] Executing {self.__class__.__name__} skill.")
        in_goal_zone = self.bus.call_service("/supervisor/ask/goal_zone")
        facing_goal = False
        hue_tol = 5

        frame, _ = self.read_sensors()
        if frame is None:
            return False

        # 1. DOWNSCALE FOR PERFORMANCE
        # Reduces the number of pixels processed by 75%, vastly improving FPS
        scale_percent = 0.5 
        width = int(frame.shape[1] * scale_percent)
        height = int(frame.shape[0] * scale_percent)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hue = self.hue - hue_tol
        higher_hue = self.hue + hue_tol

        # 2. HUE WRAP-AROUND LOGIC
        if lower_hue < 0:
            mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([higher_hue, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([180 + lower_hue, 50, 50]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(mask1, mask2)
        elif higher_hue > 180:
            mask1 = cv2.inRange(hsv, np.array([lower_hue, 50, 50]), np.array([180, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([higher_hue - 180, 255, 255]))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, np.array([lower_hue, 50, 50]), np.array([higher_hue, 255, 255]))

        # Center parameters mapped to the resized frame
        frame_center_x = frame.shape[1] // 2
        center_tolerance = frame.shape[1] // 4 # 25% each way
        MIN_PIXELS = 100

        # 3. CONTOUR TRACKING
        # Find all blobs in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Isolate the largest blob to avoid tracking background noise/distractions
            largest_contour = max(contours, key=cv2.contourArea)
        
            # contourArea returns the actual pixel area, so we don't multiply by 255 here
            if cv2.contourArea(largest_contour) > MIN_PIXELS:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    color_center_x = int(M["m10"] / M["m00"])
                    if abs(color_center_x - frame_center_x) < center_tolerance:
                        facing_goal = True

        # 4. FINAL STATE EVALUATION
        if in_goal_zone and facing_goal:
            print(f"\t[Skill] {self.__class__.__name__} succeed. Restart needed.")
            self.bus.publish("/supervisor/event/reset", {"reason": "finish"})
            return True
        
        mssg = ""
        if not in_goal_zone:
            mssg += " Not in the goal zone."
        if not facing_goal:
            mssg += " Not facing the goal."
        print(f"\t[Skill] {self.__class__.__name__} failed.{mssg}")
        return False

class Start(SerialSkill):
    '''
    Checks if it is in a room 1
    '''
    def __init__(
        self,
        bus: Communicator,
        camera: CameraSensor, 
        lidars: VirtualSensorArray, 
        wheels: VirtualActuator, 
        velocity: float, 
        save_dist: float,
        timeout: float = 15.0
    ):
        super().__init__(camera, lidars, wheels, velocity, save_dist, timeout)
        self.bus = bus

    def execute(self):
        print(f"\t[Skill] Executing {self.__class__.__name__} skill.")
        room = self.bus.call_service("/supervisor/ask/room_number")
        if room == 1:
            print(f"\t[Skill] {self.__class__.__name__} succeed.")
            return True
        print(f"\t[Skill] {self.__class__.__name__} failed. Wrong room.")
        return False