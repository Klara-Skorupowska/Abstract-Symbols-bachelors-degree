import itertools
import re
import threading
import traceback
from pathlib import Path

from simulation import RealRobot, Arena, simulation_loop
from agent import TheAgent, agent_loop
from agent.brain_network import BrainNetwork
from supervisor import supervisor_loop
from supervisor.loggers import SessionInfoLogger
from communicator import Communicator


def get_next_experiment_index(experiments_root: Path) -> int:
    """Returns the next available experiment index in logs/experiments/."""
    if not experiments_root.exists():
        return 1

    max_idx = 0
    for entry in experiments_root.iterdir():
        if entry.is_dir():
            match = re.match(r"^exp_(\d+)$", entry.name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def build_experiment_configs(
    param_grid: dict[str, list],
    mode: str = "grid",
) -> list[tuple[dict[str, object], dict[str, object]]]:
    """
    Generates a list of (full_brain_params, varied_params) pairs.

    Args:
        param_grid: Dict mapping BrainNetwork hyperparameter names to lists of values.
        mode:
            - "grid": Cartesian product of all parameter lists.
            - "one_at_a_time": Vary one parameter at a time, holding all others at default.
    """
    if not param_grid:
        return [(dict(BrainNetwork.DEFAULT_HYPERPARAMETERS), {})]

    unknown = set(param_grid.keys()) - set(BrainNetwork.DEFAULT_HYPERPARAMETERS.keys())
    if unknown:
        raise ValueError(f"Unknown BrainNetwork hyperparameters in param_grid: {sorted(unknown)}")

    for key, values in param_grid.items():
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            raise ValueError(f"Parameter '{key}' must map to a non-empty list of values.")

    experiments: list[tuple[dict[str, object], dict[str, object]]] = []

    if mode == "grid":
        keys = list(param_grid.keys())
        for combo in itertools.product(*(param_grid[k] for k in keys)):
            varied = dict(zip(keys, combo))
            full_params = dict(BrainNetwork.DEFAULT_HYPERPARAMETERS)
            full_params.update(varied)
            experiments.append((full_params, varied))

    elif mode == "one_at_a_time":
        seen: set[tuple[tuple[str, object], ...]] = set()
        for key, values in param_grid.items():
            for val in values:
                full_params = dict(BrainNetwork.DEFAULT_HYPERPARAMETERS)
                full_params[key] = val
                signature = tuple(sorted(full_params.items()))
                if signature not in seen:
                    seen.add(signature)
                    experiments.append((full_params, {key: val}))
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'grid' or 'one_at_a_time'.")

    return experiments


def run_single_experiment(
    exp_dir: Path,
    base_config: dict,
    brain_params: dict[str, object],
    varied_params: dict[str, object],
) -> None:
    """Runs a single full multi-epoch training/testing session inside exp_dir."""
    config = dict(base_config)
    config["source_dir"] = "new"
    config["session_dir"] = str(exp_dir)
    config["brain_hyperparameters"] = brain_params
    config["varied_brain_params"] = varied_params

    print(f"\n{'#' * 60}")
    print(f"[Experiment] Starting {exp_dir.name}")
    print(f"[Experiment] Varied params: {varied_params}")
    print(f"{'#' * 60}")

    bus = Communicator()

    agent = TheAgent(
        bus=bus,
        how_many_runs=config["runs_per_epoch"],
        how_many_tests=config["tests_between"],
        source_dir="new",
        session_dir=exp_dir,
        brain_params=brain_params,
    )

    robot = RealRobot(bus)
    arena = Arena()

    # Log session & brain hyperparameter info strictly in exp_<number>/
    logger = SessionInfoLogger(log_dir=str(exp_dir))
    logger.log_info(config, agent, robot, arena)

    simulation_ready_event = threading.Event()
    supervisor_ready_event = threading.Event()
    stop_event = threading.Event()

    physics_thread = threading.Thread(
        target=simulation_loop,
        args=(bus, simulation_ready_event, robot, arena, stop_event),
        daemon=True,
    )
    supervisor_thread = threading.Thread(
        target=supervisor_loop,
        args=(bus, supervisor_ready_event, stop_event, str(exp_dir)),
        daemon=True,
    )
    agent_thread = threading.Thread(
        target=agent_loop,
        args=(bus, agent, config, stop_event),
        daemon=True,
    )

    try:
        physics_thread.start()
        simulation_ready_event.wait()

        supervisor_thread.start()
        supervisor_ready_event.wait()

        agent_thread.start()
        agent_thread.join()
    except KeyboardInterrupt:
        print(f"\n[Experiment] Interrupted during {exp_dir.name}. Shutting down threads...")
        stop_event.set()
        raise
    finally:
        stop_event.set()
        physics_thread.join(timeout=10.0)
        supervisor_thread.join()  # wait for PipelinePictureGenerator to finish saving all plots
        print(f"[Experiment] Finished {exp_dir.name}.")


def run_experiments(
    param_grid: dict[str, list],
    base_config: dict,
    mode: str = "grid",
    logs_root: str | Path = "logs",
) -> None:
    """Orchestrates a hyperparameter sweep under logs/experiments/exp_<number>."""
    experiments_root = Path(logs_root) / "EXPERIMENTS"
    experiments_root.mkdir(parents=True, exist_ok=True)

    configs = build_experiment_configs(param_grid, mode=mode)
    next_idx = get_next_experiment_index(experiments_root)

    print(f"[Experiments] Queued {len(configs)} experiment(s) in '{mode}' mode.")
    print(f"[Experiments] Starting at exp_{next_idx:03d} in {experiments_root}")

    for offset, (brain_params, varied_params) in enumerate(configs):
        exp_num = next_idx + offset
        exp_dir = experiments_root / f"exp_{exp_num:03d}"

        try:
            run_single_experiment(
                exp_dir=exp_dir,
                base_config=base_config,
                brain_params=brain_params,
                varied_params=varied_params,
            )
        except KeyboardInterrupt:
            print("[Experiments] Sweep aborted by user.")
            break
        except Exception as e:
            print(f"[Experiments] ERROR in {exp_dir.name}: {e}")
            traceback.print_exc()
            print("[Experiments] Continuing to next experiment...")


if __name__ == "__main__":
    # Brain hyperparameters to sweep (keys must match BrainNetwork.DEFAULT_HYPERPARAMETERS)
    param_grid = {
        "max_nodes": [5, 10, 30],
        "precision": [0.5, 1.0],
        "n_test_points": [100, 1000],
        "min_nodes": [0, None],
        "base_radius": [0.1, 0.5],
        "max_radius": [1.0, 16.0],
        "max_edge_age": [1, 50],
        "lambda_step": [1, 20],
        "radii_factor": [0.5, 1.0],
    }

    # Shared training settings across all experiments (always uses a fresh brain)
    shared_training_config = {
        "runs_per_epoch": 100,
        "tests_between": 1,
        "tests_at_end": 10, # not counted as an epoch
        "time_limit": None,
        "min_success_rate": 2.5, # so it will not be triggered
        "max_epochs": 10,
    }

    # mode: "grid" (all combinations) or "one_at_a_time" (vary one param at a time)
    run_experiments(
        param_grid=param_grid,
        base_config=shared_training_config,
        mode="one_at_a_time",
    )