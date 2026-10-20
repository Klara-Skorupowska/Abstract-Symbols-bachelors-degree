## main file ###

# --- imports ---
import json
import os.path
import re
import threading
import time
from datetime import datetime

from simulation import *
from agent import *
from supervisor import *
from communicator import Communicator


def save_session_info(
    session_path: str,
    chosen_session_dir: str,
    runs: int,
    tests: int,
    time_limit: float,
    min_success_rate: float,
    agent: TheAgent,
    robot,
    arena
):
    """Generates an info.txt file with run parameters, brain hyperparameters, and component types."""
    if not os.path.exists(session_path):
        os.makedirs(session_path, exist_ok=True)

    info_file_path = os.path.join(session_path, "info.txt")

    brain = getattr(agent, "brain", None)
    
    # Extract brain_network hyperparameters with fallbacks
    hyperparams = {
        "min_overlap_ratio": getattr(brain, "min_overlap_ratio", '?'),
        "min_nodes_in_symbol": getattr(brain, "min_nodes", '?'),
        "allowed_variancy": getattr(brain, "allowed_variancy", '?'),
        "max_points": getattr(brain, "max_points", '?'),
        "base_radius": getattr(brain, "base_radius", '?'),
        "max_radius": getattr(brain, "max_radius", '?'),
        "learning_rate_b": getattr(brain, "learning_rate_b", '?'),
        "learning_rate_n": getattr(brain, "learning_rate_n", '?'),
        "max_edge_age": getattr(brain, "max_edge_age", '?'),
        "lambda_step": getattr(brain, "lambda_step", '?'),
        "alpha": getattr(brain, "alpha", '?'),
        "d": getattr(brain, "d", '?'),
    }

    content = [
        "========================================",
        "          EXPERIMENT RUN INFO           ",
        "========================================",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "--- SESSION CONFIGURATION ---",
        f"Chosen session_dir: {chosen_session_dir}",
        f"Runs per epoch: {runs}",
        f"Tests per epoch: {tests}",
        f"Time limit for learning: {time_limit} hours",
        f"Min success rate to stop learning: {min_success_rate}",
        "",
        "--- COMPONENT TYPES ---",
        f"Agent type: {type(agent).__name__}",
        f"Robot type: {type(robot).__name__}",
        f"Arena type: {type(arena).__name__}",
        "",
        "--- BRAIN NETWORK HYPERPARAMETERS ---",
    ]

    for param, val in hyperparams.items():
        content.append(f"{param}: {val}")

    content.append("========================================\n")

    with open(info_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content))
    print(f"[Info] Saved session metadata to {info_file_path}")


# --- let's go ---
def main(session_dir, runs, tests, time_limit=None, min_success_rate=None, is_first_run=False):
    print("Starting")
    print("Open communication")
    bus = Communicator()
    ### ---- CHOOSE PARAMETERS HERE ---- ###
    ## agent ##
    agent = TheAgent(bus, runs, tests, session_dir)
    ## simulation ##
    robot = RealRobot(bus)
    arena = Arena()
    ### ---- ------ ---------- ---- ---- ###

    # Write info.txt on the initial run
    if is_first_run:
        actual_session_dir = getattr(agent, "session_dir", None) or get_latest_session("logs")
        if actual_session_dir:
            save_session_info(
                session_path=actual_session_dir,
                chosen_session_dir=session_dir,
                runs=runs,
                tests=tests,
                time_limit=time_limit,
                min_success_rate=min_success_rate,
                agent=agent,
                robot=robot,
                arena=arena
            )

    # event for waiting for full simulation initialization
    simulation_ready_event = threading.Event()
    # event for waiting for full supervisor and data initialization
    supervisor_ready_event = threading.Event()
    # event for closing everything
    stop_event = threading.Event()
    # create threads
    physics_thread = threading.Thread(target=simulation_loop, args=(bus, simulation_ready_event, robot, arena, stop_event, ), daemon=True)
    agent_thread = threading.Thread(target=agent_loop, args=(bus, agent, stop_event, ), daemon=True)
    supervisor_thread = threading.Thread(target=supervisor_loop, args=(bus, supervisor_ready_event, stop_event,), daemon=True)
    # start 
    print("Start Nodes")
    physics_thread.start()
    simulation_ready_event.wait()
    supervisor_thread.start()
    supervisor_ready_event.wait()
    agent_thread.start()

    # end 
    agent_thread.join()
    physics_thread.join()
    supervisor_thread.join()

    print("All threads closed cleanly. Exiting program.")


def get_success_rate(file_path: str) -> float:
    """
    Reads the success rate from either the new test_statistics.json file,
    the new test_statistics.txt report, or falls back to execution_statistics.txt.
    Returns the success rate as a percentage (0.0 to 100.0).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Handling for JSON format (test_statistics.json)
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 'success_rate' in test_statistics.json is stored as a float between 0.0 and 1.0
        if "overall_statistics" in data:
            stats = data["overall_statistics"]
            if "success_rate" in stats:
                rate = stats["success_rate"]
                return float(rate)
        
        raise ValueError("Keys 'overall_statistics' or 'success_rate' not found in test statistics JSON file.")
    raise ValueError("No success rate in given file (JSON).")

def get_runs(file_path: str) -> int:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r"runs:\s*(\d+)", content)
    if not match:
        raise ValueError("Nie znaleziono informacji o liczbie uruchomień (runs).")

    return int(match.group(1))

def append_text_to_file(file_path: str, text: str, ensure_newline: bool = True) -> None:
    """
    Dopisuje podany tekst na koniec pliku.
    """
    if ensure_newline:
        prefix = ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content and not content.endswith("\n"):
                    prefix = "\n"
        except FileNotFoundError:
            pass

        suffix = "" if text.endswith("\n") else "\n"
        text = prefix + text + suffix

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(text)

def get_latest_session(logs_root):
    if not os.path.exists(logs_root):
        return None

    def parse_session_timestamp(dir_name):
        try:
            return datetime.strptime(dir_name, "%Y-%m-%d_%H-%M")
        except ValueError:
            return None

    valid_sessions = []
    for entry in os.scandir(logs_root):
        if entry.is_dir() and parse_session_timestamp(entry.name):
            valid_sessions.append(entry.path)

    return max(valid_sessions, key=os.path.basename, default=None)

def run_til_success(session_dir, runs, tests, time_limit = 2, max_sessions = 4, min_success_rate = 0.5):
    sum_runs = 0
    retries = 0
    start_time = time.time()
    if max_sessions is None:
        max_sessions = float('inf')
    
    # Store initial chosen session directory before loops switch to 'continue'
    initial_chosen_dir = session_dir

    while time.time() - start_time < time_limit * 3600:
        if retries > max_sessions:
            break
        retries += 1

        main(
            session_dir=session_dir,
            runs=runs,
            tests=tests,
            time_limit=time_limit,
            min_success_rate=min_success_rate,
            is_first_run=(retries == 1)
        )  # MAIN LOOP #

        session_dir = 'continue'
        stats_file_path = os.path.join(get_latest_session("logs"), "execution_statistics.txt")
        tests_file_path = os.path.join(get_latest_session("logs"), "test_statistics.json")
        sum_runs += get_runs(stats_file_path)
        if get_success_rate(tests_file_path) >= min_success_rate:
            break

    duration = time.time() - start_time
    hours, remainder = divmod(int(duration), 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    mssg = f"After {retries} attemps ({sum_runs} runs) program achieved success rate of {get_success_rate(tests_file_path):.2f}. Work time: {duration_str}. Initial dir: {initial_chosen_dir}\n"
    append_text_to_file(stats_file_path, mssg)

if __name__ == "__main__":
    ### Choose Parameters Here ###
    session_dir = 'continue'         # 'new' or 'continue' or <session dir> - direction where the brain is (under /logs/session_dir)
    runs = 1000                 # no. runs - either execution of plan or exploration - before test
    tests = 10                  # how many times tries to go from an active skill to finish from random spawn
    time_limit = 12             # in hours, passing the limit breaks the loop
    max_sessions = 1            # maximum number of session; break the loop early
    min_success_rate = 0.7      # % of successful tests to break the loop early

    run_til_success(session_dir, runs, tests, time_limit, max_sessions, min_success_rate)
    main("continue", 0, 100)   # extra tests
