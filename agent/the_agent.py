from __future__ import annotations
import os
import sys
import shutil
import subprocess
import random
import time
from datetime import datetime
from pathlib import Path
import numpy as np

from .core import Agent
from communicator import Communicator
from .virtual_actuator import WheelsActuator
from .virtual_sensor import VirtualSensorArray, LidarSensor, CameraSensor
from .skills_serial import SpotTheColor, GoToTheDoor, GoThroughTheDoor, GoToTheGoal, Finish
from .brain_network import BrainNetwork
from .session_utils import resolve_source_directory
from supervisor.loggers import ExecutionStatsLogger, TestSummaryLogger


class TheAgent(Agent):
    """Agent capable of autonomous skill execution, PDDL planning, and continuous symbolic learning."""

    def __init__(self, bus: Communicator, how_many_runs: int, how_many_tests: int, source_dir: str = 'new'):
        super().__init__(bus)
        self.wheels = WheelsActuator(self.bus)
        self.lidars = VirtualSensorArray(
            self.bus, [LidarSensor(self.bus, ang) for ang in [17, 50, 90, 150, 210, 270, 310, 343]]
        )
        self.camera = CameraSensor(self.bus)

        self.max_runs = how_many_runs
        self.N_tests = how_many_tests

        self.data_buffer = []
        self.buffer_max_len = 10

        self.current_skill = None
        self.bus.register_service('agent/ask/action', self._give_current_action)
        self.reset_requested = False
        self.bus.subscribe("/supervisor/event/reset", self._handle_supervisor_reset)

        save_dist = 0.05
        velocity = 15

        self.skillset = {
            'find_door': SpotTheColor(60, self.camera, self.lidars, self.wheels, velocity, save_dist, timeout=4.0),
            'approach_door': GoToTheDoor(bus, 60, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=15.0),
            'open_door': GoThroughTheDoor(bus, self.camera, self.lidars, self.wheels, velocity, save_dist, timeout=10.0),
            'find_goal': SpotTheColor(26, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=4.0),
            'approach_goal': GoToTheGoal(bus, 26, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=15.0),
            'finish': Finish(bus, 26, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=1.0),
        }

        # 1. Create session directory (session root: logs/YYYY-MM-DD_HH-MM)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.session_dir = Path("logs") / timestamp
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = str(self.session_dir)

        # 2. Resolve continuation source (resolves to the last epoch dir of the target session)
        source_epoch_dir, parent_session = resolve_source_directory(
            source_dir,
            current_session=self.session_dir,
            logs_root="logs"
        )
        self.continued_from = str(parent_session) if parent_session else None

        # 3. Internal state holders; updated per epoch by set_epoch_output_dir()
        self.epoch_dir: str | None = None
        self.pddl_dir: str | None = None
        self.task_dir: str | None = None
        self.dataset_path: str | None = None
        self.test_stats_path: str | None = None
        self.exec_logger: ExecutionStatsLogger | None = None
        self.test_logger: TestSummaryLogger | None = None

        self.transition_records = []
        self.skill_stats = {
            name: {'true_positive': 0, 'true_negative': 0, 'false_positive': 0, 'false_negative': 0}
            for name in self.skillset.keys()
        }
        self.logs = []
        self.start_time = None
        self.end_time = None
        self.run_count = 0
        self.batch_update_count = 0
        self.completed_plans_count = 0
        self.timestamps = []

        # 4. Instantiate and load brain: Brain initialized with log_dir=None to prevent root session pollution
        self.brain = BrainNetwork(log_dir=None)
        if source_epoch_dir:
            self.brain.load_brain(source_epoch_dir)
        self.brain.prepare_nodes(self.skillset.keys())

    def set_epoch_output_dir(self, epoch_dir: str | Path):
        """Re-targets all agent and brain file outputs strictly inside the epoch directory."""
        self.epoch_dir = str(epoch_dir)
        os.makedirs(self.epoch_dir, exist_ok=True)

        self.pddl_dir = os.path.join(self.epoch_dir, "PDDL")
        self.task_dir = os.path.join(self.pddl_dir, "task")
        os.makedirs(self.task_dir, exist_ok=True)

        self.dataset_path = os.path.join(self.epoch_dir, "offline_transitions.npz")
        self.test_stats_path = os.path.join(self.epoch_dir, "test_statistics.json")

        # Re-point loggers
        self.exec_logger = ExecutionStatsLogger(self.epoch_dir)
        self.test_logger = TestSummaryLogger(self.epoch_dir)

        # Re-point brain output
        if hasattr(self, "brain") and self.brain is not None:
            self.brain.set_output_dir(self.epoch_dir)
            # Ensure domain.pddl is initialized inside task_dir if brain is already learned
            self.brain.generate_and_save_domain()

    def _give_current_action(self, request=None):
        return self.current_skill

    def _handle_supervisor_reset(self, message):
        self.reset_requested = True
        if message.get('reason') == 'finish':
            self.bus.call_service("/supervisor/do/FinishReached")

    def _reset_and_restart(self):
        self.current_skill = None
        self.reset_requested = False
        self.bus.call_service("/supervisor/do/restart")
        self.bus.call_service("/supervisor/do/FinishReset")

    def _logger(self, run: int):
        if not self.exec_logger:
            return
        self.brain.save_symbol_stats()
        self.exec_logger.log_statistics(
            start_time=self.start_time,
            end_time=self.end_time,
            run_count=run,
            batch_update_count=self.batch_update_count,
            completed_plans_count=self.completed_plans_count,
            skillset=self.skillset,
            brain_skills=self.brain.skills,
            skill_stats=self.skill_stats,
            logs=self.logs,
            continued_from=self.continued_from
        )
        self.exec_logger.log_timestamps(self.timestamps)

    def record_transition(self, prev_state: np.ndarray, skill_name: str, post_state: np.ndarray, succeed: bool):
        self.transition_records.append({
            'prev_state': np.asarray(prev_state, dtype=np.float32),
            'skill_name': str(skill_name),
            'post_state': np.asarray(post_state, dtype=np.float32),
            'succeed': bool(succeed),
            'timestamp': time.time()
        })

        if len(self.transition_records) % 50 == 0:
            self.save_offline_dataset()

    def save_offline_dataset(self):
        if not self.transition_records or not self.dataset_path:
            return

        prev_states = np.array([r['prev_state'] for r in self.transition_records], dtype=np.float32)
        skill_names = np.array([r['skill_name'] for r in self.transition_records], dtype=object)
        post_states = np.array([r['post_state'] for r in self.transition_records], dtype=np.float32)
        succeeds = np.array([r['succeed'] for r in self.transition_records], dtype=bool)
        timestamps = np.array([r['timestamp'] for r in self.transition_records], dtype=np.float64)

        np.savez_compressed(
            self.dataset_path,
            prev_states=prev_states,
            skill_names=skill_names,
            post_states=post_states,
            succeeds=succeeds,
            timestamps=timestamps
        )
        print(f"[Agent] Saved {len(self.transition_records)} transitions to {self.dataset_path}")

    def run(self):
        try:
            self.run_count = 0
            self.start_time = time.time()
            self._reset_and_restart()

            while self.run_count < self.max_runs:
                self._logger(self.run_count)
                self.run_count += 1
                print(f"[Agent] Run {self.run_count}/{self.max_runs} ---------------")
                if self.reset_requested:
                    self._reset_and_restart()

                self.bus.call_service("/supervisor/do/switchThink")
                plan = self.generate_plan(goal_skill='finish')
                self.bus.call_service("/supervisor/do/switchThink")
                if len(plan) < 1:
                    print("[Agent] Returned empty plan.")
                    self.explore()
                    continue

                print("[Agent] Start executing plan.")
                print(f"[Agent] Plan: {plan}")
                plan_completed = True
                for i, skill_name in enumerate(plan):
                    print(f"[Agent] Executing step {i+1}/{len(plan)}: {skill_name}")
                    state_before, _ = self.read_state()
                    status = self.brain.predicates_status(skill_name, state_before)

                    success = self.execute_skill(skill_name)
                    if not success:
                        print("[Agent] Failed action. Abandoning the plan.")
                        plan_completed = False
                        break

                    state_after, _ = self.read_state()
                    status = self.brain.predicates_status(skill_name, state_after)
                    if status != 'EFFECTS':
                        print("[Agent] Effects not there. Abandoning the plan.")
                        plan_completed = False
                        break

                    self.add_to_buffer((state_before, skill_name, state_after))

                if plan_completed:
                    self.completed_plans_count += 1
                else:
                    self.explore()

            if self.data_buffer:
                self.bus.call_service("/supervisor/do/switchSleep")
                self.brain.update(self.data_buffer)
                self.bus.call_service("/supervisor/do/switchSleep")
                self.batch_update_count += 1
                self.data_buffer = []

        except Exception as e:
            import traceback
            print(f"[Agent] CRITICAL CRASH in run(): {e}")
            traceback.print_exc()

        finally:
            if self.data_buffer:
                self.brain.update(self.data_buffer)
                self.data_buffer = []
            self.end_time = time.time()
            self.brain._logger()
            self.save_offline_dataset()
            self._logger(self.run_count)

    def execute_skill(self, skill_name: str) -> bool:
        print(f"[Agent] Execute Skill: {skill_name}")
        self.current_skill = skill_name
        prev_vector_state, _ = self.read_state()
        self.brain.create_node(skill_name)

        preconditions_are_met = self.brain.preconditions_met(skill_name, prev_vector_state)
        skill = self.skillset[skill_name]
        succeed = skill.execute()
        post_vector_state, _ = self.read_state()

        self.record_transition(prev_vector_state, skill_name, post_vector_state, succeed)

        if preconditions_are_met:
            if succeed:
                self.skill_stats[skill_name]['true_positive'] += 1
            else:
                self.skill_stats[skill_name]['false_positive'] += 1
        else:
            if succeed:
                self.skill_stats[skill_name]['false_negative'] += 1
            else:
                self.skill_stats[skill_name]['true_negative'] += 1

        return succeed

    def read_state(self) -> np.ndarray:
        self.lidars.read()
        self.camera.read()
        self.camera.preprocess()
        self.lidars.preprocess()

        vector_state = np.concatenate([self.lidars.coded, self.camera.coded]).flatten()

        active_symbols = self.brain.get_active_symbols(vector_state)

        return vector_state, active_symbols

    def explore(self):
        print("[Agent] Explore")
        skill_name = random.choice(list(self.skillset.keys()))
        self.execute_skill(skill_name)
        if self.reset_requested:
            self._reset_and_restart()

    def add_to_buffer(self, data: tuple):
        self.data_buffer.append(data)
        self.timestamps.append(time.time())
        print(f"[Agent] Add data to buffer. {len(self.data_buffer)}/{self.buffer_max_len}")
        if len(self.data_buffer) >= self.buffer_max_len:
            self.bus.call_service("/supervisor/do/switchSleep")
            self.brain.update(self.data_buffer)
            self.bus.call_service("/supervisor/do/switchSleep")
            self.batch_update_count += 1
            self.data_buffer = []

    def _archive_problem_files(self, plan_name: str):
        """Safely isolates and archives only problem.pddl and problem.pddl.soln, keeping domain.pddl intact in task/."""
        target_dir = os.path.join(self.pddl_dir, plan_name)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)

        for filename in ("problem.pddl", "problem.pddl.soln"):
            src_file = os.path.join(self.task_dir, filename)
            if os.path.exists(src_file):
                shutil.move(src_file, os.path.join(target_dir, filename))

    def generate_plan(self, goal_skill: str | None = None, plan_name: str | None = None) -> list[str]:
        print("[Agent] Generate Plan")
        vector_state, active_symbols = self.read_state()

        if not active_symbols:
            print("[Agent] No active symbols detected.")
            return []
        print(f"[Agent] Active symbols: {active_symbols}")

        nodes = self.brain.get_nodes()
        if not nodes:
            print("[Agent] No skill nodes available.")
            return []

        if goal_skill is None or goal_skill not in nodes:
            goal_skill = random.choice(nodes)

        print(f"[Agent] Generating plan to achieve {goal_skill}.")

        # In each epoch, both domain.pddl and problem.pddl reside in epoch_dir/PDDL/task/
        domain_file = os.path.join(self.task_dir, "domain.pddl")
        problem_file = os.path.join(self.task_dir, "problem.pddl")
        plan_file = os.path.join(self.task_dir, "problem.pddl.soln")

        if not os.path.exists(domain_file):
            print("[Agent] Missing domain.pddl in task/. Creating one.")
            self.brain.generate_and_save_domain()

        problem_data = self.brain.generate_problem_pddl(active_symbols, goal_skill)
        if not problem_data:
            print("[Agent] No problem file generated.")
            return []

        with open(problem_file, "w", encoding="utf-8") as f:
            f.write(problem_data)

        if os.path.exists(plan_file):
            os.remove(plan_file)

        try:
            subprocess.run(
                [sys.executable, "-m", "pyperplan", "-s", "astar", "-H", "hff", domain_file, problem_file],
                capture_output=True, text=True, check=True,
                timeout=10.0
            )
        except subprocess.TimeoutExpired:
            print("[Agent] Pyperplan timed out (no plan reachable).")
            self.logs.append(f"[Pyperplan] [RUN {self.run_count}] Pyperplan timed out.")
            return []
        except subprocess.CalledProcessError as e:
            print(f"[Agent] Pyperplan error: {e.stderr or e.stdout}")
            return []
        except Exception as e:
            print(f"[Agent] Unexpected error running Pyperplan: {e}")
            return []

        if not os.path.exists(plan_file):
            print("[Agent] Pyperplan failed (no plan file).")
            return []

        self.bus.call_service("/supervisor/do/switchThink")
        executable_plan = []
        with open(plan_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(";"):
                    continue
                parts = line.strip("()").split()
                if not parts:
                    continue
                action_name_lower = parts[0].lower()
                matched_skill = next(
                    (sk for sk in self.skillset.keys() if action_name_lower == sk.lower()),
                    None
                )
                if matched_skill:
                    executable_plan.append(matched_skill)

        if len(executable_plan) <= 0:
            print("[Agent] Error: no executable plan.")
        elif len(executable_plan) > 6:
            if plan_name is None:
                plan_name = f"plan_{self.run_count}"
            self._archive_problem_files(plan_name)

        return executable_plan

    def test_run(self):
        self.brain._logger()
        self.tests_data = {}

        for i in range(self.N_tests):
            test_key = str(i)
            print(f"[Agent] Test {i+1}/{self.N_tests}.")
            start = time.time()

            self.bus.call_service("/supervisor/do/restart")
            time.sleep(1.0)

            raw_spawn = self.bus.call_service("/supervisor/ask/position")
            if hasattr(raw_spawn, "tolist"):
                spawn_pos = raw_spawn.tolist()
            elif isinstance(raw_spawn, (list, tuple)):
                spawn_pos = [float(x) if hasattr(x, "item") else x for x in raw_spawn]
            else:
                spawn_pos = raw_spawn

            self.bus.call_service("/supervisor/do/switchThink")
            plan = self.generate_plan(goal_skill='finish', plan_name=f"test_{i}")
            self.bus.call_service("/supervisor/do/switchThink")

            test_record = {
                "test_idx": i,
                "plan": plan if plan else [],
                "plan_length": len(plan) if plan else 0,
                "steps_executed": 0,
                "last_step": "-",
                "result": False,
                "spawn_pos": spawn_pos,
                "duration": 0.0,
                "steps": []
            }

            if plan:
                result = True
                print(f"[Agent] Executing plan: {plan}")
                for step, skill_name in enumerate(plan):
                    print(f"[Agent] Executing step {step+1}/{len(plan)}: {skill_name}")
                    state_now, _ = self.read_state()
                    status = self.brain.predicates_status(skill_name, state_now)
                    if status != "PRECONDITIONS":
                        print(f"[Agent] Preconditions are not met: {skill_name}")
                        test_record["steps"].append({
                            "step": step,
                            "skill": skill_name,
                            "succeed": False,
                            "reason": "preconditions_failed"
                        })
                        test_record["steps_executed"] = step
                        result = False
                        break

                    skill = self.skillset[skill_name]
                    succeed = skill.execute()
                    test_record["last_step"] = skill_name
                    test_record["steps_executed"] = step + 1

                    state_now, _ = self.read_state()
                    status = self.brain.predicates_status(skill_name, state_now)
                    if status != "EFFECTS":
                        print(f"[Agent] Effects are not met: {skill_name}")
                        test_record["steps"].append({
                            "step": step,
                            "skill": skill_name,
                            "succeed": False,
                            "reason": "effects_failed"
                        })
                        result = False
                        break

                    test_record["steps"].append({
                        "step": step,
                        "skill": skill_name,
                        "succeed": bool(succeed),
                        "reason": "execution"
                    })

                    if not succeed:
                        print(f"[Agent] Skill {skill_name} failed during execution.")
                        result = False
                        break

                test_record["result"] = result

            test_record["duration"] = time.time() - start
            self.tests_data[test_key] = test_record

            if test_record["result"]:
                print("[Agent] Test succeed.")
            else:
                print("[Agent] Test failed.")

        if self.test_logger:
            self.test_logger.log_results(self.tests_data)
