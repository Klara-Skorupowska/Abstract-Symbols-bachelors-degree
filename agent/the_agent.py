from .core import Agent
from communicator import Communicator
from .virtual_actuator import *
from .virtual_sensor import *
from .skills_serial import *
from .brain_network import *

import os
import sys
import subprocess
import random
import numpy as np
from datetime import datetime
import time

class TheAgent(Agent):
    '''
    It explores the environment and abstracts symbols from used skills. 
    It also accomplishes the goal of moving from one pad to another.
    '''
    class ExecutedSkill:
        def __init__(self, skill_name: str, succeed: bool):
            self.name = skill_name
            self.succeed = succeed

    def __init__(self, bus: Communicator, how_many_runs: int, how_many_tests: int, source_dir='new'):
        super().__init__(bus)
        self.wheels = WheelsActuator(self.bus)
        self.lidars = VirtualSensorArray(self.bus, [LidarSensor(self.bus, ang) for ang in [17, 50, 90, 150, 210, 270, 310, 343]])
        self.camera = CameraSensor(self.bus)
        
        self.max_runs = how_many_runs
        self.N_tests = how_many_tests

        self.fail_buffor = []
        self.buffor_max_len = 10 

        self.previous_skill = self.ExecutedSkill(None, None)
        self.current_skill = None
        self.bus.register_service('agent/ask/action', self._give_current_action)
        self.reset_requested = False
        self.bus.subscribe("/supervisor/event/reset", self._handle_supervisor_reset)

        save_dist = 0.08
        velocity = 15
        self.skillset = {
            'Start': Start(bus, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=1.0),
            'Find_the_Door': SpotTheColor(60, self.camera, self.lidars, self.wheels, velocity, save_dist, timeout=4.0),
            'Go_to_the_Door': GoToTheDoor(bus, 60, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=15.0),
            'Open_the_Door': GoThroughTheDoor(bus, self.camera, self.lidars, self.wheels, velocity, save_dist, timeout=10.0),
            'Find_the_Goal': SpotTheColor(26, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=4.0),
            'Go_to_the_Goal': GoToTheGoal(bus, 26, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=15.0),
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
                print(f"[Agent] No previous session found. Creating new one.")
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
        self.transition_records = []

        self.skill_stats = {name: {'true_positive': 0, 'true_negative': 0, 'false_positive': 0, 'false_negative': 0} for name in self.skillset.keys()}
        self.logs = []
        self.tests = []
        self.start_time = None
        self.end_time = None
        self.run_count = 0
        self.full_clean_run_count = 0
        self.batch_update_count = 0
        self.timestamps = []
        self.test_stats_path = os.path.join(self.log_dir, "test_statistics.json")
        self.test_stats_txt_path = os.path.join(self.log_dir, "test_statistics.txt")
        self.test_step_executions = []

        ### --- brain --- ###
        
        self.brain = BrainNetwork(log_dir=self.log_dir)
        self.brain.load_brain(resolved_source_dir)
        
    def _give_current_action(self, request=None):
        return self.current_skill

    def _handle_supervisor_reset(self, message):
        self.reset_requested = True
        if message['reason'] == 'finish':
            self.bus.call_service("/supervisor/do/restart")

    def _reset_and_restart(self):
        self.previous_skill = self.ExecutedSkill(None, None)
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

        valid_sessions = []
        for entry in os.scandir(logs_root):
            if entry.is_dir() and parse_session_timestamp(entry.name):
                valid_sessions.append(entry.path)

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
                f.write(f"runs: {run}; brain updates: {self.batch_update_count}; clean runs: {self.full_clean_run_count}\n")
                f.write(f"{'Skill Name':<20} | {'True Positive':<15} | {'True Negative':<15} | {'false Positive':<15} | {'false Negative':<15} | {'Total':<15}\n")
                f.write("-" * 100 + "\n")
                for sk_name, counts in self.skill_stats.items():
                    sumall = counts['true_positive'] + counts['true_negative'] + counts['false_positive'] + counts['false_negative']
                    f.write(f"{sk_name:<20} | {counts['true_positive']:<15} | {counts['true_negative']:<15} | {counts['false_positive']:<15} | {counts['false_negative']:<15} | {sumall:<10}\n")
                f.write("-" * 100 + "\n")
                for log in self.logs:
                    f.write(f"{log}\n")

        if self.timestamps_path:
            with open(self.timestamps_path, "w") as f:
                for stmp in self.timestamps:
                    f.write(f"{stmp}\n")

    def record_transition(self, prev_state: np.ndarray, skill_name: str, post_state: np.ndarray, succeed: bool):
        """
        Stores individual transition events in memory and periodically flushes 
        them to compressed numpy archives on disk.
        """
        self.transition_records.append({
            'prev_state': np.asarray(prev_state, dtype=np.float32),
            'skill_name': str(skill_name),
            'post_state': np.asarray(post_state, dtype=np.float32),
            'succeed': bool(succeed),
            'timestamp': time.time()
        })
        
        # Periodically dump every 50 transitions to avoid data loss on crash
        if len(self.transition_records) % 50 == 0:
            self.save_offline_dataset()

    def save_offline_dataset(self):
        """Flushes recorded transitions to disk in compressed NPZ format."""
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

                plan, start_skill, finish_skill = self.generate_plan()
                if len(plan) < 1:
                    print("[Agent] Returned empty plan.")
                    self.explore() 
                    continue

                print("[Agent] Start executing plan.")
                print(f"[Agent] Plan: {plan}")
                plan_completed = True
                for i, skill_name in enumerate(plan):
                    print(f"[Agent] Executing step {i+1}/{len(plan)}: {skill_name}")
                    success = self.execute_skill(skill_name, self.previous_skill)
                    if not success:
                        print("[Agent] Failed action. Abandoning the plan.")
                        self.explore()
                        plan_completed = False
                        break
                    self.previous_skill.name = skill_name
                    self.previous_skill.succeed = True
                if start_skill == 'Start' and finish_skill == 'Finish' and plan_completed:
                    print("[Agent] Clear Run from Start to Finish.")
                    self.full_clean_run_count += 1

        except Exception as e:
            import traceback
            print(f"[Agent] CRITICAL CRASH in run(): {e}")
            traceback.print_exc()

        finally:
            self.brain.update(self.fail_buffor)
            self.end_time = time.time()
            self.brain._logger()
            self.save_offline_dataset()
            self._logger(self.run_count)

    def execute_skill(self, skill_name, prev_skill: ExecutedSkill):
        print(f"[Agent] Execute Skill")
        self.current_skill = skill_name
        prev_vector_state = self.read_state()
        self.brain.create_node(skill_name)
        preconditions_are_met = self.brain.preconditions_met(skill_name, prev_vector_state)
        
        skill = self.skillset[skill_name]
        succeed = skill.execute()
        post_vector_state = self.read_state()

        self.record_transition(prev_vector_state, skill_name, post_vector_state, succeed)
        
        data = (prev_vector_state, skill_name, post_vector_state, succeed)
        expectable = (preconditions_are_met and succeed) or (not preconditions_are_met and not succeed)
        if not expectable:
            self.add_to_buffor(data)

        # Confusion Matrix Correction:
        # Predicted True (preconditions met), Actual False -> False Positive
        # Predicted False (preconditions not met), Actual True -> False Negative
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

    def read_state(self):
        self.lidars.read()
        self.camera.read()
        self.camera.preprocess()
        self.lidars.preprocess()
        return np.concatenate([self.lidars.coded, self.camera.coded]).flatten()

    def explore(self):
        print("[Agent] Explore")
        available_skills = list(self.skillset.keys())
        if not available_skills:
            return None
        skill_name = random.choice(available_skills)
        succeed = self.execute_skill(skill_name, self.previous_skill)
        self.previous_skill = self.ExecutedSkill(skill_name, succeed)
        if self.reset_requested: 
            self._reset_and_restart()
     
    def add_to_buffor(self, data):
        print(f"[Agent] Add data to buffor. {len(self.fail_buffor)}/{self.buffor_max_len}")
        if len(self.fail_buffor) >= self.buffor_max_len:
            self.bus.call_service("/supervisor/do/switchSleep")
            self.brain.update(self.fail_buffor)
            self.bus.call_service("/supervisor/do/switchSleep")
            self.batch_update_count += 1
            self.fail_buffor = []
        self.fail_buffor.append(data)
        self.timestamps.append(time.time())

    def generate_plan(self, start_skill=None, goal_skill=None, plan_name = None):
        print("[Agent] Generate Plan")
        vector_state = self.read_state()
        if start_skill is None:
            active_skills = self.brain.get_active_skills(vector_state)
            if not active_skills:
                print(f"[Agent] No active skills.")
                return [], None, None 
            print(f"[Agent] Active predicates: {active_skills}")
            start_skill = random.choice(active_skills)
        if goal_skill is None:
            nodes = self.brain.get_nodes()
            nodes = [n for n in nodes if n != start_skill]
            if not nodes:
                print(f"[Agent] No nodes in transitional graph.")
                return [], None, None
            goal_skill = random.choice(nodes)

        print(f"[Agent] Generating plan from {start_skill} to {goal_skill}.")

        plan_dir = os.path.join(self.pddl_dir, "newest")
        os.makedirs(plan_dir, exist_ok=True)
        domain_file = os.path.join(plan_dir, "domain.pddl")
        problem_file = os.path.join(plan_dir, "problem.pddl")
        plan_file = os.path.join(plan_dir, "problem.pddl.soln")

        self.brain.abstract_symbols()
        with open(domain_file, "w") as f:
            f.write(self.brain.generate_domain_pddl())
        with open(problem_file, "w") as f:
            f.write(self.brain.generate_problem_pddl(start_skill, goal_skill))

        if not os.path.exists(domain_file) or not os.path.exists(problem_file):
            print(f"[Agent] Error: no .pddl files")
            return [], None, None

        if os.path.exists(plan_file):
            os.remove(plan_file)

        try:
            subprocess.run(
                [sys.executable, "-m", "pyperplan", "-s", "astar", "-H", "hff", domain_file, problem_file],
                capture_output=True, text=True, check=True,
                timeout=10.0
            )
        except subprocess.TimeoutExpired:
            print(f"[Agent] Pyperplan timed out (no plan reachable).")
            self.logs.append(f"[Pyperplan] [RUN {self.run_count}]  Pyperplan timed out.")
            return [], None, None
        except subprocess.CalledProcessError as e:
            print(f"[Agent] Pyperplan error: {e.stderr or e.stdout}")
            return [], None, None
        except Exception as e:
            print(f"[Agent] Unexpected error running Pyperplan: {e}")
            return [], None, None

        if not os.path.exists(plan_file):
            return [], None, None

        executable_plan = []
        with open(plan_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(";"):
                    continue

                parts = line.strip("()").split()
                if len(parts) < 1:
                    continue

                action_name_lower = parts[0].lower()
        
                # Skip internal transition connectors
                if action_name_lower.startswith("transition_"):
                    continue

                matched_skill = next(
                    (sk for sk in self.skillset.keys() if sk.lower() == action_name_lower),
                    None
                )
                if matched_skill:
                    executable_plan.append(matched_skill)

        if len(executable_plan) <= 0:
            print(f"[Agent] Error: no executable plan.")
        elif len(executable_plan) > 6:
            if plan_name is None: plan_name = f"plan_{self.run_count}"
            exec_plan_dir = os.path.join(self.pddl_dir, plan_name)
            os.rename(plan_dir, exec_plan_dir)

        return executable_plan, start_skill, goal_skill

    def test_run(self):
        self.brain.abstract_symbols()
        self.brain._logger()
        self.test_step_executions = []
        for i in range(self.N_tests):
            print(f"[Agent] Test {i+1}/{self.N_tests}.")
            start = time.time()
            self.bus.call_service("/supervisor/do/restart")
            time.sleep(1.0)
            plan, _, _ = self.generate_plan(goal_skill='Finish', plan_name=f"test_{i}")
            
            if len(plan) < 1:
                steps_executed = 0
                last_step = '-'
                result = False
            else:
                last_step = '-'
                result = True
                print(f"[Agent] Executing plan: {plan}")
                for step, skill_name in enumerate(plan):
                    print(f"[Agent] Executing step {step+1}/{len(plan)}: {skill_name}")
                    prev_vector_state = self.read_state()
                    preconditions_are_met = self.brain.preconditions_met(skill_name, prev_vector_state)
                    
                    if not preconditions_are_met:
                        print(f"[Agent] Preconditions are not met: {skill_name}")
                        # Record failure due to unmet preconditions
                        self.test_step_executions.append({
                            'test_idx': i,
                            'step': step,
                            'skill': skill_name,
                            'succeed': False,
                            'reason': 'preconditions_failed'
                        })
                        steps_executed = step
                        result = False
                        break
                    
                    skill = self.skillset[skill_name]
                    succeed = skill.execute()
                    last_step = skill_name
                    steps_executed = step + 1

                    # Record execution result for this step
                    self.test_step_executions.append({
                        'test_idx': i,
                        'step': step,
                        'skill': skill_name,
                        'succeed': bool(succeed),
                        'reason': 'execution'
                    })
                    
                    if not succeed:
                        print(f"[Agent] Skill {skill_name} failed during execution.")
                        result = False
                        break

            duration = time.time() - start 
            self.tests.append((i, len(plan), steps_executed, last_step, duration, result))
            if result:
                print(f"[Agent] Test succeed.")
            else:
                print(f"[Agent] Test failed.")

        # Run the logger at the end of all tests
        self._test_logger()

    def _test_logger(self):
        if not self.tests:
            print("[Agent] No test records to log.")
            return

        from statistics import mean, multimode

        plan_lengths = [t[1] for t in self.tests]
        steps_executed = [t[2] for t in self.tests]
        durations = [t[4] for t in self.tests]
        results = [1 if t[5] else 0 for t in self.tests]

        # Calculate metrics
        mean_plan_length = mean(plan_lengths) if plan_lengths else 0.0
        mode_plan_length = multimode(plan_lengths) if plan_lengths else []

        mean_steps_executed = mean(steps_executed) if steps_executed else 0.0
        mode_steps_executed = multimode(steps_executed) if steps_executed else []

        mean_execution_duration = mean(durations) if durations else 0.0
        success_rate = mean(results) if results else 0.0

        # Build graph-ready matrix: skill -> step -> {success: count, fail: count}
        # Supports visualizing stacked/grouped bar charts of skill execution per step index
        step_graph_data = {}
        for entry in self.test_step_executions:
            skill = entry['skill']
            step = entry['step']
            succeed = entry['succeed']

            if skill not in step_graph_data:
                step_graph_data[skill] = {}
            if step not in step_graph_data[skill]:
                step_graph_data[skill][step] = {'succeed': 0, 'fail': 0, 'total': 0}

            if succeed:
                step_graph_data[skill][step]['succeed'] += 1
            else:
                step_graph_data[skill][step]['fail'] += 1
            step_graph_data[skill][step]['total'] += 1

        summary = {
            "total_tests": len(self.tests),
            "success_rate": success_rate,
            "mean_plan_length": mean_plan_length,
            "mode_plan_length": mode_plan_length,
            "mean_steps_executed": mean_steps_executed,
            "mode_steps_executed": mode_steps_executed,
            "mean_execution_time_s": mean_execution_duration,
            "step_graph_data": step_graph_data,
            "raw_step_executions": self.test_step_executions
        }

        # 1. Save machine-readable JSON for graph generation
        if hasattr(self, 'test_stats_path') and self.test_stats_path:
            with open(self.test_stats_path, "w") as f:
                json.dump(summary, f, indent=4)
            print(f"[Agent] Saved test plot data to {self.test_stats_path}")

        # 2. Save human-readable text report
        if hasattr(self, 'test_stats_txt_path') and self.test_stats_txt_path:
            with open(self.test_stats_txt_path, "w") as f:
                f.write("=" * 80 + "\n")
                f.write(f"TEST SESSION STATISTICS (N = {len(self.tests)})\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Success Rate:                 {success_rate * 100:.2f}%\n")
                f.write(f"Mean Plan Length:             {mean_plan_length:.2f}\n")
                f.write(f"Mode Plan Length:             {mode_plan_length}\n")
                f.write(f"Mean Steps Executed:          {mean_steps_executed:.2f}\n")
                f.write(f"Mode Steps Executed:          {mode_steps_executed}\n")
                f.write(f"Mean Plan Execution Time (s): {mean_execution_duration:.3f}\n\n")
                
                f.write("-" * 80 + "\n")
                f.write("STEP-BY-STEP EXECUTION FREQUENCY & OUTCOMES\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Skill Name':<20} | {'Step':<6} | {'Succeed':<8} | {'Failed':<8} | {'Total':<8}\n")
                f.write("-" * 80 + "\n")
                for sk_name, steps in sorted(step_graph_data.items()):
                    for step_idx, counts in sorted(steps.items()):
                        f.write(f"{sk_name:<20} | {step_idx:<6} | {counts['succeed']:<8} | {counts['fail']:<8} | {counts['total']:<8}\n")
                f.write("=" * 80 + "\n")