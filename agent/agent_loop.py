from communicator import Communicator
from .core import Agent
import threading

def agent_loop(bus:Communicator, agent:Agent, stop_event: threading.Event):
    print("[Agent] Loading brain logic")
    robot = agent
    print("[Agent] Brain logic loaded.")
    if robot.max_runs > 0:
        robot.run() # the brain = (in)finite loop
    if robot.N_tests > 0:
        robot.test_run()
    stop_event.set()