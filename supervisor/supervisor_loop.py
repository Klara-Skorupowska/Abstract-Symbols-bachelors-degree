from communicator import Communicator
from .supervisor import Supervisor
import threading
import time
from supervisor.generators.pipeline import PipelinePictureGenerator
from supervisor.monitors.pipeline import PipelineLiveMonitor

def supervisor_loop(bus: Communicator, setup_complete_event: threading.Event, stop_event: threading.Event):
    """Runs continuously at 5Hz, completely independent of the agent and simulation."""
    print("[Supervisor] Starting...")
    print("[Supervisor] Loading statistics...")
    time_step = 1.0 / 5.0
    stuck_check_time_step = 20 # sec
    god = Supervisor(bus)
    god.setup()
    
    # Initialize the new clean supervisor pipelines
    monitors = PipelineLiveMonitor(bus)
    monitors.setup()
    
    print("[Supervisor] Loaded.")
    setup_complete_event.set() 
    
    try:
        i = 0
        while not stop_event.is_set():
            i+=1
            # updating status boards and saving continuous data
            monitors.update()
            if i%(1/time_step*stuck_check_time_step) == 0:
                god.stuck_check()
            time.sleep(time_step)

    except KeyboardInterrupt:
        print("[Supervisor] Thread stopped by user.")
    except Exception as e:
        import traceback
        print(f"[Agent] CRITICAL CRASH in run(): {e}")
        traceback.print_exc()
    finally:
        monitors.close()
        
        # Call the unified picture generator pipeline
        pics = PipelinePictureGenerator(logs_root="logs")
        pics.run()
        print("[Supervisor] Supervisor closed safely.")