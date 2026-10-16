import random
from communicator import Communicator
import numpy as np
import math
import pybullet as p
import time

class Supervisor():
    '''
    supervisor is the omnipotent god: it tells the agent where it is and if the task were successful etc.
    it collects all the data, it can change the world
    It is yet another thread that uses communicator
    '''
    def __init__(self, bus: Communicator):
        self.bus = bus
        self.robot_id = self.bus.call_service(f"/realrobot/give_id")
        self.last_position = None
        self.arena_id = self.bus.call_service(f"/arena/give_id")
        
        self.bus.register_service(f"/supervisor/ask/position", self.robots_pos)
        self.bus.register_service(f"/supervisor/ask/room_number", self.which_room)
        self.bus.register_service(f"/supervisor/ask/door_zone", self.door_zone)
        self.bus.register_service(f"/supervisor/ask/goal_zone", self.goal_zone)

        self.bus.register_service(f"/supervisor/do/restart", self.restart)

        self.bus.register_service(f"/supervisor/do/switchSleep", self.switch_sleep)
        self.sleeping = False
        self.sleep_text_id = None

        self.bus.register_service(f"/supervisor/do/switchThink", self.switch_think)
        self.thinking = False
        self.think_text_id = None

    def setup(self):
        # call from other sources
        self.robot_id = self.bus.call_service(f"/realrobot/give_id")
        self.arena_id = self.bus.call_service(f"/arena/give_id")

        
    def smth_callback(self, message):
        '''
        for when data appear in the ether self.bus.subscribe("/supervisor/attend/smth", self.smth_callback)
        '''
        pass 

    def smth_handler(self, request=None):
        '''
        for when agents ask a question (self.bus.register_service(f"/supervisor/ask/smth", self.smth_handler) or gives a task (self.bus.register_service(f"/supervisor/do/smth", self.smth_handler)
        '''
        return 0

    def switch_sleep(self, request=None):
        '''
        Draw 'Zzz' above the robot, while is on
        '''
        if not self.sleeping:
            # Put to sleep: draw 'Zzz' above the robot
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            text_pos = [pos[0], pos[1], pos[2] + 0.2]  # 0.2m above robot base
            
            self.sleep_text_id = p.addUserDebugText(
                text="Zzz",
                textPosition=text_pos,
                textColorRGB=[0.2, 0.6, 1.0],  # Soft blue
                textSize=1.5,
                lifeTime=0  # 0 keeps it until manually removed
            )
            self.sleeping = True
        else:
            # Wake up: delete the writing
            if self.sleep_text_id is not None:
                p.removeUserDebugItem(self.sleep_text_id)
                self.sleep_text_id = None
            self.sleeping = False

        return self.sleeping

    def switch_think(self, request=None):
        '''
        Draw '???' above the robot, while is on
        '''
        if not self.thinking:
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            text_pos = [pos[0], pos[1], pos[2] + 0.4]  # 0.4 m above robot base
            
            self.think_text_id = p.addUserDebugText(
                text="???",
                textPosition=text_pos,
                textColorRGB=[0.2, 1.0, 0.6],  # Soft green
                textSize=1.5,
                lifeTime=0  # 0 keeps it until manually removed
            )
            self.thinking = True
        else:
            if self.think_text_id is not None:
                p.removeUserDebugItem(self.think_text_id)
                self.think_text_id = None
            self.thinking = False

        return self.thinking

    def robots_pos(self, request=None):
        position, _ = p.getBasePositionAndOrientation(self.robot_id)
        point = np.array(position[:2])
        if hasattr(point, "tolist"):
            point = point.tolist()
        elif isinstance(point, (list, tuple)):
            point = [float(x) if hasattr(x, "item") else x for x in point]
        return point

    def which_room(self, request=None):
        '''
        for when agent ask a question: in which room am I.
        1: first room without the goal
        2: second room with the goal
        '''
        position, _ = p.getBasePositionAndOrientation(self.robot_id)
        point = np.array(position[:2])
        if point[0] < 0:
            return 1
        else:
            return 2

    def stuck_check(self):
        position, _ = p.getBasePositionAndOrientation(self.robot_id)

        if self.last_position is not None:
            distance = math.dist(position, self.last_position)
            if distance < 0.01 and not self.sleeping and not self.thinking:
                print("[Supervisor] Robot stuck in place for too long.")
                # Notify the agent that an environmental reset is needed
                self.bus.publish("/supervisor/event/reset", {"reason": "stuck"})
        
        self.last_position = position
    
    def _joint_name_to_index(self, id):
        # Map all joint names to their indices
        joint_name_to_index = {}
        for i in range(p.getNumJoints(id)):
            joint_info = p.getJointInfo(id, i)
            joint_name = joint_info[1].decode('utf-8') 
            joint_name_to_index[joint_name] = i
        return joint_name_to_index

    def door_zone(self, request=None):
        # Robot position
        position, _ = p.getBasePositionAndOrientation(self.robot_id)
        x = position[0]
        y = position[1]

        # Doorway center position 
        # Based on your previous logic, the door spans from y = -0.1 to y = 0.1
        door_x = 0.0
        door_y_center = 0.0
    
        # Zone Configuration
        top_width = 0.2    # Width along the wall at the doorway (y = -0.1 to 0.1)
        bottom_width = 0.4 # Width along the wall at the far edge
        depth = 0.25       # How far the trapezoid extends into the rooms (along X)
    
        # 1. Calculate how far the robot is from the doorway (depth into the room)
        dx = abs(x - door_x)
    
        # 2. If the depth exceeds the configured max depth, it's outside the zone
        if dx > depth:
            return False
        
        # 3. Calculate the maximum allowed Y width at this specific depth
        # It expands linearly from top_width to bottom_width as dx goes from 0 to depth
        current_width = top_width + (bottom_width - top_width) * (dx / depth)
        max_allowed_dy = current_width / 2.0
    
        # 4. Check if the robot's Y is within this allowed width along the wall
        if abs(y - door_y_center) <= max_allowed_dy:
            return True
        
        return False

    def goal_zone(self, request=None):
        # robot position
        position, _ = p.getBasePositionAndOrientation(self.robot_id)
        pos_robot = np.array(position[:2])

        # goal position
        joint_name_to_index = self._joint_name_to_index(self.arena_id)
        goal_idx = joint_name_to_index.get("end_joint")
        if goal_idx is None:
            print("[Supervisor] Warning: 'end_joint' not found in URDF.")
            return None
        link_state = p.getLinkState(self.arena_id, goal_idx)
        pos_door = link_state[0] 

        # the distance
        max_dist = 0.15
        x_dist = abs(pos_door[0] - pos_robot[0])
        y_dist = abs(pos_door[1] - pos_robot[1])
        dist = (x_dist**2+y_dist**2)**(1/2)

        return dist < max_dist
       
    def restart(self, request=None):
        print("[Supervisor] Robot in environment reset.")
        # Clear any active sleep debug text on reset
        if self.sleep_text_id is not None:
            p.removeUserDebugItem(self.sleep_text_id)
            self.sleep_text_id = None
        self.sleeping = False
        # Reset robot position
        _, orientation = self.bus.call_service(f"/realrobot/give_initial_position")
        x_pos = random.uniform(-0.45, -0.25)
        y_pos = random.uniform(-0.45, 0.45)
        position = [x_pos, y_pos, 0.05]
        p.resetBasePositionAndOrientation(self.robot_id, position, orientation)
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])
        time.sleep(1.0)
        