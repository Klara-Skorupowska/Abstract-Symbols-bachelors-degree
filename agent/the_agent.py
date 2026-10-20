from .core import Agent
from communicator import Communicator
from .virtual_actuator import *
from .virtual_sensor import *
from .skills_serial import *
from .brain_network import *

import os
import sys
import shutil
import subprocess
import random
import numpy as np
from datetime import datetime
from statistics import mean, multimode
import json
import time


class TheAgent(Agent):
    """
    Explores the environment and abstracts symbols from used skills.
    Accomplishes the goal of moving from one pad to another.
    """
    def __init__(self, bus: Communicator, how_many_runs: int, how_many_tests: int, source_dir='new'):
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

        save_dist = 0.08
        velocity = 15
        self.skillset = {
            'Start': Start(bus, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=1.0),
            'Find_The_Door': SpotTheColor(60, self.camera, self.lidars, self.wheels, velocity, save_dist, timeout=4.0),
            'Go_To_The_Door': GoToTheDoor(bus, 60, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=15.0),
            'Open_The_Door': GoThroughTheDoor(bus, self.camera, self.lidars, self.wheels, velocity, save_dist, timeout=10.0),
            'Find_The_Goal': SpotTheColor(26, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=4.0),
            'Go_To_The_Goal': GoToTheGoal(bus, 26, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=15.0),
            'Finish': Finish(bus, 26, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=1.0)
        }

        ### --- logging --- ###
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.log_dir = os.path.join("logs", timestamp)
        self.continued_from = None
        resolved_source_dir = None

        if source_dir == 'new':
            pass
        elif source_dir == 'continue':
            resolved_source_dir = self._get_latest_session("logs")
            if resolved_source_dir:
                self.continued_from = resolved_source_dir
            else:
                print("[Agent] No previous session found. Creating new one.")
        else:
            resolved_source_dir = os.path.join("logs", source_dir)
            if os.path.isdir(resolved_source_dir):
                self.continued_from = resolved_source_dir
            else:
                print(f"[Agent] Directory {resolved_source_dir} does not exist. Creating new one.")
                resolved_source_dir = None

        self.pddl_dir = os.path.join(self.log_dir, "PDDL")
        self.stats_path = os.path.join(self.log_dir, "execution_statistics.txt")
        self.timestamps_path = os.path.join(self.log_dir, "timestamps.txt")
        self.dataset_path = os.path.join(self.log_dir, "offline_transitions.npz")
        self.test_stats_path = os.path.join(self.log_dir, "test_statistics.json")

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
        self.timestamps = []

        ### --- brain --- ###
        self.brain = BrainNetwork(log_dir=self.log_dir)
        self.brain.load_brain(resolved_source_dir)
        self.brain.prepare_nodes(self.skillset.keys())

    def _give_current_action(self, request=None):
        return self.current_skill

    def _handle_supervisor_reset(self, message):
        self.reset_requested = True
        if message['reason'] == 'finish':
            self.bus.call_service("/supervisor/do/restart")

    def _reset_and_restart(self):
        self.current_skill = None
        self.reset_requested = False
        self.bus.call_service("/supervisor/do/restart")

    def _get_latest_session(self, logs_root):
        if not os.path.exists(logs_root):
            return None

        def parse_session_timestamp(dir_name):
            try:
                return datetime.strptime(dir_name, "%Y-%m-%d_%H-%M")
            except ValueError:
                return None

        valid_sessions = [
            entry.path for entry in os.scandir(logs_root)
            if entry.is_dir() and parse_session_timestamp(entry.name)
        ]
        return max(valid_sessions, key=os.path.basename, default=None)

    def _logger(self, run):
        if self.stats_path:
            end_time = time.time() if self.end_time is None else self.end_time
            with open(self.stats_path, "w") as f:
                if self.continued_from:
                    f.write(f"CONTINUATION FROM SESSION: {self.continued_from}\n")
                    f.write("-" * 100 + "\n")

                start_str = datetime.fromtimestamp(self.start_time).strftime("%d-%m-%Y %H:%M:%S")
                end_str = datetime.fromtimestamp(end_time).strftime("%d-%m-%Y %H:%M:%S")

                duration = end_time - self.start_time
                hours, remainder = divmod(int(duration), 3600)
                minutes, seconds = divmod(remainder, 60)
                duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                f.write(f"start time: {start_str} | end time: {end_str} | duration: {duration_str}\n")
                f.write(f"runs: {run}; brain updates: {self.batch_update_count}\n")
                f.write(f"{'Skill Name':<20} | {'True Positive':<15} | {'True Negative':<15} | {'False Positive':<15} | {'False Negative':<15} | {'Total':<15}\n")
                f.write("-" * 100 + "\n")
                for sk_name, counts in self.skill_stats.items():
                    sumall = (
                        counts['true_positive'] + counts['true_negative'] +
                        counts['false_positive'] + counts['false_negative']
                    )
                    f.write(
                        f"{sk_name:<20} | {counts['true_positive']:<15} | {counts['true_negative']:<15} | "
                        f"{counts['false_positive']:<15} | {counts['false_negative']:<15} | {sumall:<10}\n"
                    )
                f.write("-" * 100 + "\n")
                for log in self.logs:
                    f.write(f"{log}\n")

        if self.timestamps_path:
            with open(self.timestamps_path, "w") as f:
                for stmp in self.timestamps:
                    f.write(f"{stmp}\n")

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
        if not self.transition_records:
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
                plan = self.generate_plan(goal_skill='Finish')
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
                    state_now = self.read_state()
                    status = self.brain.preconditions_status(skill_name, state_now)
                    if status == "FALSE":
                        print(f"[Agent] Plan safety abort: Precondition for {skill_name} is {status}.")
                        plan_completed = False
                        break
                    success = self.execute_skill(skill_name)
                    if not success:
                        print("[Agent] Failed action. Abandoning the plan.")
                        plan_completed = False
                        break

        except Exception as e:
            import traceback
            print(f"[Agent] CRITICAL CRASH in run(): {e}")
            traceback.print_exc()

        finally:
            self.brain.update(self.data_buffer)
            self.end_time = time.time()
            self.brain._logger()
            self.save_offline_dataset()
            self._logger(self.run_count)

    def execute_skill(self, skill_name):
        print(f"[Agent] Execute Skill: {skill_name}")
        self.current_skill = skill_name
        prev_vector_state = self.read_state()
        self.brain.create_node(skill_name)

        preconditions_are_met = self.brain.preconditions_met(skill_name, prev_vector_state) # symbols or predicate if no symbols

        skill = self.skillset[skill_name]
        succeed = skill.execute()
        post_vector_state = self.read_state()

        self.record_transition(prev_vector_state, skill_name, post_vector_state, succeed)

        if preconditions_are_met:
            if succeed:
                self.skill_stats[skill_name]['true_positive'] += 1
            else:
                self.skill_stats[skill_name]['false_positive'] += 1
        else:
            if succeed:
                self.skill_stats[skill_name]['false_negative'] += 1
                self.add_to_buffer((prev_vector_state, skill_name, post_vector_state)) # we add only unpredictable succeeses
            else:
                self.skill_stats[skill_name]['true_negative'] += 1

        return succeed

    def read_state(self):
        self.lidars.read()
        self.camera.read()
        self.camera.preprocess()
        self.lidars.preprocess()
        return np.concatenate([self.lidars.coded, self.camera.coded]).flatten()

    def explore(self):
        print("[Agent] Explore")
        skill_name = random.choice(list(self.skillset.keys()))

        self.execute_skill(skill_name)

        if self.reset_requested:
            self._reset_and_restart()

    def add_to_buffer(self, data):
        self.data_buffer.append(data)
        self.timestamps.append(time.time())
        print(f"[Agent] Add data to buffer. {len(self.data_buffer)}/{self.buffer_max_len}")
        if len(self.data_buffer) >= self.buffer_max_len:
            self.bus.call_service("/supervisor/do/switchSleep")
            self.brain.update(self.data_buffer)
            self.bus.call_service("/supervisor/do/switchSleep")
            self.batch_update_count += 1
            self.data_buffer = []

    def generate_plan(self, goal_skill=None, plan_name=None):
        print("[Agent] Generate Plan")
        vector_state = self.read_state()

        # 1. Check currently active symbols instead of skill preconditions
        active_symbols = self.brain.get_active_symbols(vector_state)
        if not active_symbols:
            print("[Agent] No active symbols detected.")
            return []
        print(f"[Agent] Active symbols: {active_symbols}")

        # 2. Pick a goal skill if none provided
        nodes = self.brain.get_nodes()
        if not nodes:
            print("[Agent] No skill nodes available.")
            return []

        if goal_skill is None:
            goal_skill = random.choice(nodes)

        print(f"[Agent] Generating plan to achieve {goal_skill}.")

        # 3. Setup PDDL directory and files
        plan_dir = os.path.join(self.pddl_dir, "task")
        os.makedirs(plan_dir, exist_ok=True)
        domain_file = os.path.join(plan_dir, "domain.pddl")
        problem_file = os.path.join(plan_dir, "problem.pddl")
        plan_file = os.path.join(plan_dir, "problem.pddl.soln")

        # 4. Generate domain and problem PDDL based on the active symbols
        with open(domain_file, "w") as f:
            f.write(self.brain.generate_domain_pddl())
        with open(problem_file, "w") as f:
            f.write(self.brain.generate_problem_pddl(active_symbols, goal_skill))

        if os.path.exists(plan_file):
            os.remove(plan_file)

        # 5. Run planner
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
        with open(plan_file, "r") as f:
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
            exec_plan_dir = os.path.join(self.pddl_dir, plan_name)
            if os.path.exists(exec_plan_dir):
                shutil.rmtree(exec_plan_dir)
            os.rename(plan_dir, exec_plan_dir)

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
            plan = self.generate_plan(goal_skill='Finish', plan_name=f"test_{i}")
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
                    prev_vector_state = self.read_state()
                    preconditions_are_met = self.brain.preconditions_met(skill_name, prev_vector_state)

                    if not preconditions_are_met:
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

        self._test_logger()

    def _test_logger(self):
        if not self.tests_data:
            print("[Agent] No test records to log.")
            return

        tests_list = list(self.tests_data.values())
        n_tests = len(tests_list)

        plan_lengths = [t["plan_length"] for t in tests_list]
        steps_executed = [t["steps_executed"] for t in tests_list]
        durations = [t["duration"] for t in tests_list]
        results = [1 if t["result"] else 0 for t in tests_list]

        successful_spawns = [
            t["spawn_pos"] for t in tests_list
            if t["result"] and t["spawn_pos"] is not None
        ]

        acceptable_start_plane = None
        if successful_spawns:
            try:
                dims = len(successful_spawns[0]) if isinstance(successful_spawns[0], (list, tuple)) else 1
                if dims > 1:
                    min_bounds = [float(min(p[d] for p in successful_spawns)) for d in range(dims)]
                    max_bounds = [float(max(p[d] for p in successful_spawns)) for d in range(dims)]
                else:
                    min_bounds = [float(min(successful_spawns))]
                    max_bounds = [float(max(successful_spawns))]
                acceptable_start_plane = {
                    "min_bounds": min_bounds,
                    "max_bounds": max_bounds,
                    "successful_positions": successful_spawns
                }
            except Exception:
                acceptable_start_plane = {"successful_positions": successful_spawns}

        plan_frequencies = {}
        transition_outcomes = {}

        for t in tests_list:
            plan_tuple = tuple(t.get("plan", []))
            if plan_tuple:
                plan_str = " -> ".join(plan_tuple)
                if plan_str not in plan_frequencies:
                    plan_frequencies[plan_str] = {"total": 0, "succeeded": 0, "failed": 0}
                plan_frequencies[plan_str]["total"] += 1
                if t["result"]:
                    plan_frequencies[plan_str]["succeeded"] += 1
                else:
                    plan_frequencies[plan_str]["failed"] += 1

            steps = t.get("steps", [])
            for k in range(len(steps)):
                current_sk = steps[k]["skill"]
                succeed = steps[k]["succeed"]

                trans_key = f"Start -> {current_sk}" if k == 0 else f"{steps[k-1]['skill']} -> {current_sk}"
                if trans_key not in transition_outcomes:
                    transition_outcomes[trans_key] = {"attempts": 0, "failed": 0}

                transition_outcomes[trans_key]["attempts"] += 1
                if not succeed:
                    transition_outcomes[trans_key]["failed"] += 1

        success_rate = mean(results) if results else 0.0
        mean_plan_len = mean(plan_lengths) if plan_lengths else 0.0
        mode_plan_len = multimode(plan_lengths) if plan_lengths else []
        mean_steps = mean(steps_executed) if steps_executed else 0.0
        mode_steps = multimode(steps_executed) if steps_executed else []
        mean_dur = mean(durations) if durations else 0.0

        summary = {
            "overall_statistics": {
                "total_tests": n_tests,
                "success_rate": success_rate,
                "mean_plan_length": mean_plan_len,
                "mode_plan_length": mode_plan_len,
                "mean_steps_executed": mean_steps,
                "mode_steps_executed": mode_steps,
                "mean_execution_time_s": mean_dur,
                "acceptable_start_position_plane": acceptable_start_plane,
            },
            "plan_frequencies": plan_frequencies,
            "transition_outcomes": transition_outcomes,
            "tests": self.tests_data
        }

        total_passed = sum(results)
        total_failed = n_tests - total_passed
        lines = [
            "=" * 60,
            "                    AGENT TEST SUMMARY",
            "=" * 60,
            "",
            "OVERALL PERFORMANCE",
            "-" * 40,
            f"Total Tests Run       : {n_tests}",
            f"Passed / Failed       : {total_passed} / {total_failed}",
            f"Success Rate          : {success_rate * 100:.2f}%",
            f"Mean Duration         : {mean_dur:.2f} s",
            f"Mean Plan Length      : {mean_plan_len:.2f} (modes: {mode_plan_len})",
            f"Mean Steps Executed   : {mean_steps:.2f} (modes: {mode_steps})",
            "",
        ]

        if acceptable_start_plane and "min_bounds" in acceptable_start_plane:
            min_b = [f"{b:.2f}" for b in acceptable_start_plane["min_bounds"]]
            max_b = [f"{b:.2f}" for b in acceptable_start_plane["max_bounds"]]
            lines.extend([
                "SUCCESSFUL SPAWN ENVELOPE",
                "-" * 40,
                f"Min Bounds : [{', '.join(min_b)}]",
                f"Max Bounds : [{', '.join(max_b)}]",
                f"Total Valid Spawns : {len(acceptable_start_plane.get('successful_positions', []))}",
                "",
            ])

        lines.extend([
            "PLAN SEQUENCES (Sorted by Frequency)",
            "-" * 40,
        ])
        sorted_plans = sorted(plan_frequencies.items(), key=lambda kv: kv[1]["total"], reverse=True)
        if sorted_plans:
            for plan_str, counts in sorted_plans:
                pct = (counts["succeeded"] / counts["total"]) * 100 if counts["total"] else 0.0
                lines.append(f"• [{counts['total']}x] {plan_str}")
                lines.append(f"    Passed: {counts['succeeded']} | Failed: {counts['failed']} | Success: {pct:.1f}%")
        else:
            lines.append("  (No plans recorded)")
        lines.append("")

        lines.extend([
            "TRANSITION RELIABILITY (Sorted by Fail Rate)",
            "-" * 40,
        ])
        sorted_trans = sorted(
            transition_outcomes.items(),
            key=lambda kv: (kv[1]["failed"] / kv[1]["attempts"] if kv[1]["attempts"] else 0.0),
            reverse=True
        )
        if sorted_trans:
            for trans_str, counts in sorted_trans:
                att = counts["attempts"]
                fail = counts["failed"]
                fail_rate = (fail / att) * 100 if att else 0.0
                lines.append(f"• {trans_str:<32} -> {att - fail}/{att} ok ({fail_rate:.1f}% fail rate)")
        else:
            lines.append("  (No transitions recorded)")
        lines.append("")
        lines.append("=" * 60)

        txt_summary = "\n".join(lines)

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                if isinstance(obj, np.bool_):
                    return bool(obj)
                return super().default(obj)

        if self.test_stats_path:
            with open(self.test_stats_path, "w") as f:
                json.dump(summary, f, indent=4, cls=NumpyEncoder)
            print(f"[Agent] Saved test data to {self.test_stats_path}")

            base_root, _ = os.path.splitext(self.test_stats_path)
            txt_path = f"{base_root}_summary.txt"
            with open(txt_path, "w") as f:
                f.write(txt_summary)
            print(f"[Agent] Saved readable test summary to {txt_path}")