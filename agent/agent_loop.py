import os
import time
import json
import threading
from communicator import Communicator
from .core import Agent


def _read_success_rate(test_stats_path: str) -> float:
    if not os.path.exists(test_stats_path):
        return 0.0
    try:
        with open(test_stats_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("overall_statistics", {}).get("success_rate", 0.0))
    except Exception:
        return 0.0


def agent_loop(bus: Communicator, agent: Agent, config: dict, stop_event: threading.Event):
    runs_per_epoch = config.get("runs_per_epoch", 100)
    tests_between = config.get("tests_between", 0)
    tests_at_end = config.get("tests_at_end", 0)
    time_limit_hours = config.get("time_limit", 2.0)
    min_success_rate = config.get("min_success_rate", 0.7)
    max_epochs = config.get("max_epochs", 100)
    session_dir = config.get("session_dir", agent.log_dir)

    time_limit_seconds = (time_limit_hours * 3600) if time_limit_hours else float("inf")
    start_time = time.time()
    epoch_idx = 0
    final_success_rate = 0.0

    print("[Agent] Starting multi-epoch agent loop.")

    try:
        while epoch_idx < max_epochs:
            elapsed = time.time() - start_time
            if elapsed >= time_limit_seconds:
                print(f"[Agent] Time limit reached ({elapsed / 3600:.2f} h). Stopping epochs.")
                break

            print(f"\n{'='*30} EPOCH {epoch_idx:03d} {'='*30}")

            epoch_dir = os.path.join(session_dir, f"epoch_{epoch_idx:03d}")
            agent.set_epoch_output_dir(epoch_dir)

            agent._reset_and_restart()

            if runs_per_epoch > 0:
                agent.max_runs = runs_per_epoch
                agent.run()

            if tests_between > 0:
                agent.N_tests = tests_between
                agent.test_run()
                final_success_rate = _read_success_rate(agent.test_stats_path)
                print(f"[Agent] Epoch {epoch_idx} Success Rate: {final_success_rate * 100:.2f}%")

                if final_success_rate >= min_success_rate:
                    print(f"[Agent] Target success rate {min_success_rate * 100:.1f}% reached!")
                    epoch_idx += 1
                    break

            epoch_idx += 1

        if tests_at_end > 0:
            print(f"\n{'='*30} FINAL TESTS ({tests_at_end} runs) {'='*30}")
            final_epoch_dir = os.path.join(session_dir, f"epoch_{epoch_idx:03d}_final_tests")
            agent.set_epoch_output_dir(final_epoch_dir)
            agent._reset_and_restart()

            agent.N_tests = tests_at_end
            agent.test_run()
            final_success_rate = _read_success_rate(agent.test_stats_path)
            print(f"[Agent] Final Evaluation Success Rate: {final_success_rate * 100:.2f}%")

    except Exception as e:
        import traceback
        print(f"[Agent Loop] Error during execution: {e}")
        traceback.print_exc()

    finally:
        print("[Agent] Multi-epoch execution completed. Setting stop_event.")
        stop_event.set()
