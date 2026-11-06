import threading
from simulation import RealRobot, Arena, simulation_loop
from agent import TheAgent, agent_loop
from supervisor import supervisor_loop
from communicator import Communicator
from supervisor.loggers import SessionInfoLogger

def main(config: dict):
    print("[Main] Initializing system...")
    bus = Communicator()

    agent = TheAgent(
        bus=bus,
        how_many_runs=config["runs_per_epoch"],
        how_many_tests=config["tests_between"],
        source_dir=config["source_dir"]
    )

    robot = RealRobot(bus)
    arena = Arena()

    session_dir = agent.log_dir
    config["session_dir"] = session_dir

    logger = SessionInfoLogger(log_dir=session_dir)
    logger.log_info(config, agent, robot, arena)

    simulation_ready_event = threading.Event()
    supervisor_ready_event = threading.Event()
    stop_event = threading.Event()

    physics_thread = threading.Thread(
        target=simulation_loop,
        args=(bus, simulation_ready_event, robot, arena, stop_event),
        daemon=True
    )
    supervisor_thread = threading.Thread(
        target=supervisor_loop,
        args=(bus, supervisor_ready_event, stop_event),
        daemon=True
    )
    agent_thread = threading.Thread(
        target=agent_loop,
        args=(bus, agent, config, stop_event),
        daemon=True
    )

    print("[Main] Starting threads...")
    physics_thread.start()
    simulation_ready_event.wait()

    supervisor_thread.start()
    supervisor_ready_event.wait()

    agent_thread.start()

    agent_thread.join()
    physics_thread.join()
    supervisor_thread.join()

    print("[Main] All threads closed cleanly. Session finished.")

if __name__ == "__main__":
    hyperparameters = {
        "source_dir": "new",          
        "runs_per_epoch": 1000,        
        "tests_between": 5,           
        "tests_at_end": 100,           
        "time_limit": None,            
        "min_success_rate": 0.75,      
        "max_epochs": 100              
    }
    main(hyperparameters)
