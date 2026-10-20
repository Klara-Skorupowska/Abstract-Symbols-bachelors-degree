from communicator import Communicator
from .core import Supervisor
import threading
import time
from .pipeline_pic_gen import *
from .pipeline_live import *
from .graph_from_pddl import *

def supervisor_loop(bus: Communicator, setup_complete_event: threading.Event, stop_event: threading.Event):
    """Runs continuously at 5Hz, completely independent of the agent and simulation."""
    print("[Supervisor] Starting...")
    print("[Supervisor] Loading statistics...")
    # status boards
    # colectiong data
    time_step = 1.0 / 5.0
    stuck_check_time_step = 20 # sec
    god = Supervisor(bus)
    god.setup()
    monitors = PipelineLiveMonitor(bus)
    loggers = [GraphFromPDDL()]
    monitors.setup()
    print("[Supervisor] Loaded.")
    setup_complete_event.set() 
    
    try:
        i = 0
        while not stop_event.is_set():
            i+=1
            # updating statusboards and saving continuous data
            monitors.update()
            if i%(1/time_step*stuck_check_time_step) == 0:
                god.stuck_check()
            for log in loggers:
                if stop_event.is_set(): break
                log.run()
            time.sleep(time_step)

    except KeyboardInterrupt:
        print("[Supervisor] Thread stopped by user.")
    except Exception as e:
        import traceback
        print(f"[Agent] CRITICAL CRASH in run(): {e}")
        traceback.print_exc()
    finally:
        monitors.close()
        # pictures
        pics = PipelinePictureGenerator(logs_root="logs")
        pics.run()
        print("[Supervisor] Supervisor closed safely.")
