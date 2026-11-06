from communicator import Communicator
from .core import Agent
from collections import Counter
from .predicate import Predicate
from .symbol_generator import SymbolGenerator
from .virtual_actuator import *
from .virtual_sensor import *
from .skills_serial import *
from .skills import *
from .brain_network import *
from datetime import datetime
from statistics import mean, multimode
import os, time, json, threading, re, numpy as np, sys, shutil, subprocess, random
def _read_success_rate(test_stats_path: str) -> float:
    if (not os.path.exists(test_stats_path)):
        return 0.0
    try:
        with open(test_stats_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return float(data.get('overall_statistics', {}).get('success_rate', 0.0))
    except Exception:
        return 0.0
class BrainNetwork():

    def __init__(self, log_dir=None):
        self.state_dim = None
        self.skills = {}
        self.precision = 0.9
        self.n_test_points = 100
        self.min_nodes = None
        self.span_tolerance = 0.1
        self.max_nodes = 40
        self.base_radius = 0.1
        self.max_radius = 1.0
        self.learning_rate_b = 0.2
        self.learning_rate_n = 0.01
        self.max_edge_age = 15
        self.lambda_step = 5
        self.alpha = 0.7
        self.d = 0.99
        self.symbol_gen = SymbolGenerator(precision=self.precision, min_nodes_in_symbol=self.min_nodes, span_tolerance=self.span_tolerance, n_test_points=self.n_test_points)
        self.log_dir = log_dir
        if self.log_dir:
            self.pddl_dir = os.path.join(self.log_dir, 'PDDL')
            self.gas_dir = os.path.join(self.log_dir, 'PDDL', 'GNG')
            os.makedirs(self.pddl_dir, exist_ok=True)
            os.makedirs(self.gas_dir, exist_ok=True)
            self.symbol_stats_path = os.path.join(self.log_dir, 'symbol_statistics.txt')

    def add_node(self, skill_name: str, precondition: Predicate, effect: Predicate):
        skill_name = skill_name.lower().strip()
        self.skills[skill_name] = {'precondition': precondition, 'effect': effect}

    def create_node(self, skill_name: str):
        skill_name = skill_name.lower().strip()
        if (skill_name not in self.skills):
            precondition = Predicate(self.max_nodes, self.state_dim, self.base_radius, self.max_radius, self.learning_rate_b, self.learning_rate_n, self.max_edge_age, self.lambda_step, self.alpha, self.d)
            effect = Predicate(self.max_nodes, self.state_dim, self.base_radius, self.max_radius, self.learning_rate_b, self.learning_rate_n, self.max_edge_age, self.lambda_step, self.alpha, self.d)
            self.add_node(skill_name, precondition, effect)

    def prepare_nodes(self, skillset):
        for skill_name in skillset:
            self.create_node(skill_name)

    def effects_met(self, skill_name: str, state_vector) -> bool:
        skill_name = skill_name.lower().strip()
        if (skill_name not in self.skills):
            return False
        symbols_map = self.symbol_gen.get_all_symbols()
        if (not symbols_map):
            return False
        skill_map = self.symbol_gen.get_skill_map()
        required_symbols = skill_map.get(skill_name, {}).get('effect', set())
        if (not required_symbols):
            return False
        return all((((sym in symbols_map) and symbols_map[sym].is_active(state_vector)) for sym in required_symbols))

    def preconditions_met(self, skill_name: str, state_vector) -> bool:
        skill_name = skill_name.lower().strip()
        if (skill_name not in self.skills):
            return False
        symbols_map = self.symbol_gen.get_all_symbols()
        if (not symbols_map):
            return False
        skill_map = self.symbol_gen.get_skill_map()
        required_symbols = skill_map.get(skill_name, {}).get('precondition', set())
        if (not required_symbols):
            return False
        return all((((sym in symbols_map) and symbols_map[sym].is_active(state_vector)) for sym in required_symbols))

    def predicates_status(self, skill_name: str, state_vector) -> str:
        pre = self.preconditions_met(skill_name, state_vector)
        eff = self.effects_met(skill_name, state_vector)
        if (pre and eff):
            return 'BOTH'
        if pre:
            return 'PRECONDITIONS'
        if eff:
            return 'EFFECTS'
        return 'FALSE'

    def get_nodes(self) -> list:
        return list(self.skills.keys())

    def update(self, buffer):
        if (not buffer):
            return
        print('\t[Brain] Updating Predicates (GNG)')
        pre_states = [item[0] for item in buffer]
        post_states = [item[2] for item in buffer]
        all_states = np.vstack([*pre_states, *post_states])
        if (self.state_dim is None):
            self.state_dim = all_states.shape[1]
        self.symbol_gen.update_bounds(all_states)
        updated_skills = set()
        for (pre_vector_state, skill_name, post_vector_state) in buffer:
            skill_name = skill_name.lower().strip()
            if (skill_name not in self.skills):
                self.create_node(skill_name)
            self.skills[skill_name]['precondition'].update(pre_vector_state)
            self.skills[skill_name]['effect'].update(post_vector_state)
            updated_skills.add(skill_name)
        self.abstract_symbols(updated_skills=updated_skills)
        self.generate_and_save_domain()
        self._logger()

    def abstract_symbols(self, updated_skills=None) -> None:
        if (updated_skills is None):
            updated_skills = set(self.skills.keys())
        for skill_name in updated_skills:
            if (skill_name in self.skills):
                skill_name = skill_name.lower().strip()
                self.symbol_gen.generate_symbols(skill_name, self.skills[skill_name])
                self.symbol_gen.agregate_symbols(skill_name)

    def get_active_symbols(self, vector) -> list:
        symbols_map = self.symbol_gen.get_all_symbols()
        if (not symbols_map):
            self.abstract_symbols()
            symbols_map = self.symbol_gen.get_all_symbols()
        if symbols_map:
            return [sym_name for (sym_name, pred) in symbols_map.items() if pred.is_active(vector)]
        return []

    def _get_skill_symbol_interactions(self, symbols_map: dict=None):
        skill_map = self.symbol_gen.get_skill_map()
        action_pre = {}
        action_eff = {}
        for (skill_name, roles) in skill_map.items():
            sk_lower = skill_name.lower()
            pre = set(roles.get('precondition', set()))
            eff = set(roles.get('effect', set()))
            if (symbols_map is not None):
                pre = {s for s in pre if (s in symbols_map)}
                eff = {s for s in eff if (s in symbols_map)}
            action_pre[sk_lower] = pre
            action_eff[sk_lower] = eff
        return (action_pre, action_eff)

    def generate_and_save_domain(self, domain_name='robot_domain'):
        if (not self.log_dir):
            return
        task_dir = os.path.join(self.pddl_dir, 'task')
        os.makedirs(task_dir, exist_ok=True)
        domain_file = os.path.join(task_dir, 'domain.pddl')
        domain_content = self.generate_domain_pddl(domain_name=domain_name)
        with open(domain_file, 'w') as f:
            f.write(domain_content)

    def generate_domain_pddl(self, domain_name='robot_domain') -> str:
        self.symbol_gen.merge_symbols()
        symbols_map = self.symbol_gen.get_all_symbols()
        skill_map = self.symbol_gen.get_skill_map()
        all_predicates = set(symbols_map.keys())
        for roles in skill_map.values():
            all_predicates.update(roles['precondition'])
            all_predicates.update(roles['effect'])
        pddl = [f'(define (domain {domain_name})', '  (:requirements :strips)', '  (:predicates']
        for pred in sorted(all_predicates):
            pddl.append(f'    ({pred})')
        pddl.append('  )')
        for skill_name in sorted(self.skills.keys()):
            sk_lower = skill_name.lower()
            roles = skill_map.get(skill_name, {'precondition': set(), 'effect': set()})
            pre_conditions = set(roles['precondition'])
            eff_conditions = set(roles['effect'])
            adds = set(eff_conditions)
            dels = set((pre_conditions - eff_conditions))
            pre_body = (f"(and {' '.join((f'({p})' for p in sorted(pre_conditions)))})" if pre_conditions else '(and)')
            eff_tokens = ([f'({a})' for a in sorted(adds)] + [f'(not ({d}))' for d in sorted(dels)])
            eff_body = (f"(and {' '.join(eff_tokens)})" if eff_tokens else '(and)')
            pddl.extend(['', f'  (:action {sk_lower}', '    :parameters ()', f'    :precondition {pre_body}', f'    :effect {eff_body}', '  )'])
        pddl.append(')')
        return '\n'.join(pddl)

    def generate_problem_pddl(self, active_symbols, goal_skill: str, problem_name='plan_problem', domain_name='robot_domain') -> str:
        goal_skill = goal_skill.lower().strip()
        skill_map = self.symbol_gen.get_skill_map()
        goal_predicates = set(skill_map.get(goal_skill, {}).get('effect', set()))
        if (not goal_predicates):
            return ''
        pddl = [f'(define (problem {problem_name})', f'  (:domain {domain_name})', '  (:init']
        for pred in sorted(active_symbols):
            pddl.append(f'    ({pred})')
        pddl.append('  )')
        pddl.append('  (:goal')
        goals_str = ' '.join((f'({g})' for g in sorted(goal_predicates)))
        pddl.append(f'    (and {goals_str})')
        pddl.append('  )')
        pddl.append(')')
        return '\n'.join(pddl)

    def export_symbol_statistics(self, file_path=None):
        if ((not self.log_dir) and (file_path is None)):
            return
        if (file_path is None):
            file_path = os.path.join(self.log_dir, 'symbol_statistics.txt')
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        symbols_map = self.symbol_gen.get_all_symbols()
        skill_map = self.symbol_gen.get_skill_map()
        lines = [('=' * 70), 'SYMBOL & SKILL ABSTRACTION STATISTICS', ('=' * 70), f'State Dimension: {self.state_dim}', f'Total Aggregated Symbols: {len(symbols_map)}', ('-' * 70), 'SYMBOLS:']
        for sym_name in sorted(symbols_map.keys()):
            pred = symbols_map[sym_name]
            lines.append(f'  {sym_name:<10} | dims: {len(pred.active_dimensions)} | nodes: {len(pred.nodes)}')
        lines.append(('-' * 70))
        lines.append('SKILL INTERACTIONS:')
        for skill_name in sorted(self.skills.keys()):
            roles = skill_map.get(skill_name, {'precondition': set(), 'effect': set()})
            pre_syms = sorted(roles['precondition'])
            eff_syms = sorted(roles['effect'])
            lines.append(f'Skill: {skill_name}')
            lines.append(f'  - Precondition Symbols : {pre_syms}')
            lines.append(f'  - Effect Symbols       : {eff_syms}')
            lines.append(('-' * 70))
        temp_file = f'{file_path}.tmp'
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(('\n'.join(lines) + '\n'))
        for _ in range(5):
            try:
                os.replace(temp_file, file_path)
                break
            except (PermissionError, OSError):
                time.sleep(0.05)

    def save_abstract_symbols(self):
        if (not self.log_dir):
            return
        os.makedirs(self.gas_dir, exist_ok=True)
        symbols_map = self.symbol_gen.get_all_symbols()
        active_filenames = {f'{sym_name}.json' for sym_name in symbols_map.keys()}
        for fname in os.listdir(self.gas_dir):
            if (fname.endswith('.json') and re.match('^sym_\\d+\\.json$', fname)):
                if (fname not in active_filenames):
                    try:
                        os.remove(os.path.join(self.gas_dir, fname))
                    except OSError:
                        pass
        for (sym_name, predicate) in symbols_map.items():
            predicate.save_predicate(self.gas_dir, sym_name)
        self.export_symbol_statistics(self.symbol_stats_path)

    def save_skill_map(self):
        if (not self.log_dir):
            return
        map_path = os.path.join(self.gas_dir, 'skill_map.json')
        serializable_map = {'skills': {sk: {role: list(syms) for (role, syms) in roles.items()} for (sk, roles) in self.symbol_gen.skill_map.items()}, 'bounds': {'dim_min': (self.symbol_gen.dim_min.tolist() if (self.symbol_gen.dim_min is not None) else None), 'dim_max': (self.symbol_gen.dim_max.tolist() if (self.symbol_gen.dim_max is not None) else None), 'state_dim': self.state_dim}}
        with open(map_path, 'w') as f:
            json.dump(serializable_map, f, indent=2)

    def _logger(self):
        if (not self.log_dir):
            return
        for (skill_name, predicates) in self.skills.items():
            predicates['precondition'].save_predicate(self.gas_dir, f'{skill_name}_init')
            predicates['effect'].save_predicate(self.gas_dir, f'{skill_name}_eff')
        self.save_abstract_symbols()
        self.save_skill_map()

    def save_symbol_stats(self, filename='symbol_stats_log.txt') -> str:
        if (not self.log_dir):
            target_path = filename
        else:
            target_path = os.path.join(self.log_dir, filename)
        symbols_map = (self.symbol_gen.get_all_symbols() or {})
        skill_map = (self.symbol_gen.get_skill_map() or {})
        total_symbols = len(symbols_map)
        total_skills = len(self.skills)
        if (total_symbols > 0):
            dim_counts = [len(getattr(pred, 'active_dimensions', [])) for pred in symbols_map.values()]
            counter = Counter(dim_counts)
            min_val = min(counter.keys())
            min_dim_str = f'{min_val} ({counter[min_val]})'
            max_val = max(counter.keys())
            max_dim_str = f'{max_val} ({counter[max_val]})'
            (dominant_val, dominant_count) = counter.most_common(1)[0]
            dominant_dim_str = f'{dominant_val} ({dominant_count})'
        else:
            min_dim_str = '0 (0)'
            dominant_dim_str = '0 (0)'
            max_dim_str = '0 (0)'
        if (total_skills > 0):
            pre_counts = [len(roles.get('precondition', set())) for roles in skill_map.values()]
            eff_counts = [len(roles.get('effect', set())) for roles in skill_map.values()]
            avg_pre_symbols = np.mean(pre_counts)
            avg_eff_symbols = np.mean(eff_counts)
        else:
            avg_pre_symbols = 0.0
            avg_eff_symbols = 0.0
        if (total_symbols > 0):
            symbol_usage = Counter()
            for roles in skill_map.values():
                for sym in (set(roles.get('precondition', set())) | set(roles.get('effect', set()))):
                    symbol_usage[sym] += 1
            reusability_ratio = (sum(symbol_usage.values()) / total_symbols)
        else:
            reusability_ratio = 0.0
        header = 'timestamp,total_symbols,total_skills,min_dims,dominant_dims,max_dims,avg_pre_symbols,avg_eff_symbols,reusability_ratio\n'
        row = f'''{time.time():.2f},{total_symbols},{total_skills},"{min_dim_str}","{dominant_dim_str}","{max_dim_str}",{avg_pre_symbols:.2f},{avg_eff_symbols:.2f},{reusability_ratio:.2f}
'''
        file_exists = (os.path.exists(target_path) and (os.path.getsize(target_path) > 0))
        with open(target_path, 'a', encoding='utf-8') as f:
            if (not file_exists):
                f.write(header)
            f.write(row)
        return row.strip()

    def load_brain(self, source_dir, regenerate_symbols: bool=False):
        print('\t[Brain] Loading brain from files.')
        if ((source_dir is None) or (not os.path.isdir(source_dir))):
            print('\t[Brain] Invalid source directory.')
            return
        src_gas_dir = os.path.join(source_dir, 'PDDL', 'GNG')
        if (not os.path.exists(src_gas_dir)):
            print(f'	[Brain] Could not find GNG directory: {src_gas_dir}')
            return
        skill_names = set()
        for filename in os.listdir(src_gas_dir):
            match = re.match('^(.*)_init\\.json$', filename)
            if match:
                skill_names.add(match.group(1))
        for name in skill_names:
            pre_file = os.path.join(src_gas_dir, f'{name}_init.json')
            eff_file = os.path.join(src_gas_dir, f'{name}_eff.json')
            if (os.path.exists(pre_file) and os.path.exists(eff_file)):
                precondition = Predicate(self.max_nodes).load_predicate(pre_file)
                effect = Predicate(self.max_nodes).load_predicate(eff_file)
                self.add_node(name, precondition, effect)
                if ((self.state_dim is None) and (len(precondition.nodes) > 0)):
                    self.state_dim = precondition.nodes.shape[1]
            else:
                print(f"	[Brain] Incomplete predicates for skill '{name}', skipping.")
        loaded_symbols = {}
        for filename in os.listdir(src_gas_dir):
            match = re.match('^(sym_\\d+)\\.json$', filename)
            if match:
                sym_name = match.group(1)
                sym_file = os.path.join(src_gas_dir, filename)
                loaded_symbols[sym_name] = Predicate(self.max_nodes).load_predicate(sym_file)
        map_path = os.path.join(src_gas_dir, 'skill_map.json')
        if os.path.exists(map_path):
            with open(map_path, 'r') as f:
                data = json.load(f)
            loaded_map = data.get('skills', data)
            for (sk, roles) in loaded_map.items():
                self.symbol_gen.skill_map[sk]['precondition'] = set(roles.get('precondition', []))
                self.symbol_gen.skill_map[sk]['effect'] = set(roles.get('effect', []))
            if ('bounds' in data):
                b_min = data['bounds'].get('dim_min')
                b_max = data['bounds'].get('dim_max')
                if ((b_min is not None) and (b_max is not None)):
                    self.symbol_gen.dim_min = np.asarray(b_min, dtype=np.float64)
                    self.symbol_gen.dim_max = np.asarray(b_max, dtype=np.float64)
                if data['bounds'].get('state_dim'):
                    self.state_dim = data['bounds']['state_dim']
        if (self.symbol_gen.dim_min is None):
            for skill in self.skills.values():
                if (len(skill['precondition'].nodes) > 0):
                    self.symbol_gen.update_bounds(skill['precondition'].nodes)
                if (len(skill['effect'].nodes) > 0):
                    self.symbol_gen.update_bounds(skill['effect'].nodes)
        if (loaded_symbols and (not regenerate_symbols)):
            self.symbol_gen.symbols = loaded_symbols
            print(f'	[Brain] Successfully restored {len(loaded_symbols)} symbols from disk.')
        else:
            print('\t[Brain] Re-abstracting symbols from scratch.')
            self.symbol_gen.symbols.clear()
            self.abstract_symbols()
        self.generate_and_save_domain()
        self.save_abstract_symbols()
        self._logger()
class TheAgent(Agent):

    def __init__(self, bus: Communicator, how_many_runs: int, how_many_tests: int, source_dir='new'):
        super().__init__(bus)
        self.wheels = WheelsActuator(self.bus)
        self.lidars = VirtualSensorArray(self.bus, [LidarSensor(self.bus, ang) for ang in [17, 50, 90, 150, 210, 270, 310, 343]])
        self.camera = CameraSensor(self.bus)
        self.max_runs = how_many_runs
        self.N_tests = how_many_tests
        self.data_buffer = []
        self.buffer_max_len = 10
        self.current_skill = None
        self.bus.register_service('agent/ask/action', self._give_current_action)
        self.reset_requested = False
        self.bus.subscribe('/supervisor/event/reset', self._handle_supervisor_reset)
        save_dist = 0.05
        velocity = 15
        self.skillset = {'find_door': SpotTheColor(60, self.camera, self.lidars, self.wheels, velocity, save_dist, timeout=4.0), 'approach_door': GoToTheDoor(bus, 60, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=15.0), 'open_door': GoThroughTheDoor(bus, self.camera, self.lidars, self.wheels, velocity, save_dist, timeout=10.0), 'find_goal': SpotTheColor(26, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=4.0), 'approach_goal': GoToTheGoal(bus, 26, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=15.0), 'finish': Finish(bus, 26, self.camera, self.lidars, self.wheels, velocity, save_dist=save_dist, timeout=1.0)}
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        self.log_dir = os.path.join('logs', timestamp)
        self.continued_from = None
        resolved_source_dir = None
        if (source_dir == 'new'):
            pass
        elif (source_dir == 'continue'):
            resolved_source_dir = self._get_latest_session('logs')
            if resolved_source_dir:
                self.continued_from = resolved_source_dir
            else:
                print('[Agent] No previous session found. Creating new one.')
        else:
            resolved_source_dir = os.path.join('logs', source_dir)
            if os.path.isdir(resolved_source_dir):
                self.continued_from = resolved_source_dir
            else:
                print(f'[Agent] Directory {resolved_source_dir} does not exist. Creating new one.')
                resolved_source_dir = None
        self.pddl_dir = os.path.join(self.log_dir, 'PDDL')
        self.stats_path = os.path.join(self.log_dir, 'execution_statistics.txt')
        self.timestamps_path = os.path.join(self.log_dir, 'timestamps.txt')
        self.dataset_path = os.path.join(self.log_dir, 'offline_transitions.npz')
        self.test_stats_path = os.path.join(self.log_dir, 'test_statistics.json')
        self.transition_records = []
        self.skill_stats = {name: {'true_positive': 0, 'true_negative': 0, 'false_positive': 0, 'false_negative': 0} for name in self.skillset.keys()}
        self.logs = []
        self.start_time = None
        self.end_time = None
        self.run_count = 0
        self.batch_update_count = 0
        self.completed_plans_count = 0
        self.timestamps = []
        self.brain = BrainNetwork(log_dir=self.log_dir)
        self.brain.load_brain(resolved_source_dir)
        self.brain.prepare_nodes(self.skillset.keys())

    def set_epoch_output_dir(self, epoch_dir: str):
        self.epoch_dir = epoch_dir
        os.makedirs(self.epoch_dir, exist_ok=True)
        self.pddl_dir = os.path.join(self.epoch_dir, 'PDDL')
        os.makedirs(self.pddl_dir, exist_ok=True)
        self.stats_path = os.path.join(self.epoch_dir, 'execution_statistics.txt')
        self.timestamps_path = os.path.join(self.epoch_dir, 'timestamps.txt')
        self.dataset_path = os.path.join(self.epoch_dir, 'offline_transitions.npz')
        self.test_stats_path = os.path.join(self.epoch_dir, 'test_statistics.json')
        if (hasattr(self, 'brain') and (self.brain is not None)):
            self.brain.log_dir = self.epoch_dir

    def _give_current_action(self, request=None):
        return self.current_skill

    def _handle_supervisor_reset(self, message):
        self.reset_requested = True
        if (message['reason'] == 'finish'):
            self.bus.call_service('/supervisor/do/FinishReached')

    def _reset_and_restart(self):
        self.current_skill = None
        self.reset_requested = False
        self.bus.call_service('/supervisor/do/restart')
        self.bus.call_service('/supervisor/do/FinishReset')

    def _get_latest_session(self, logs_root):
        if (not os.path.exists(logs_root)):
            return None

        def parse_session_timestamp(dir_name):
            try:
                return datetime.strptime(dir_name, '%Y-%m-%d_%H-%M')
            except ValueError:
                return None
        valid_sessions = [entry.path for entry in os.scandir(logs_root) if (entry.is_dir() and parse_session_timestamp(entry.name))]
        return max(valid_sessions, key=os.path.basename, default=None)

    def _logger(self, run):
        self.brain.save_symbol_stats()
        if self.stats_path:
            end_time = (time.time() if (self.end_time is None) else self.end_time)
            with open(self.stats_path, 'w') as f:
                if self.continued_from:
                    f.write(f'''CONTINUATION FROM SESSION: {self.continued_from}
''')
                    f.write((('-' * 100) + '\n'))
                start_str = datetime.fromtimestamp(self.start_time).strftime('%d-%m-%Y %H:%M:%S')
                end_str = datetime.fromtimestamp(end_time).strftime('%d-%m-%Y %H:%M:%S')
                duration = (end_time - self.start_time)
                (hours, remainder) = divmod(int(duration), 3600)
                (minutes, seconds) = divmod(remainder, 60)
                duration_str = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
                f.write(f'''start time: {start_str} | end time: {end_str} | duration: {duration_str}
''')
                f.write(f'''runs: {run}; brain updates: {self.batch_update_count}; completed plans: {self.completed_plans_count}
''')
                f.write((('-' * 100) + '\n'))
                f.write(f'''Collected Data:
''')
                f.write(f'''{'Skill Name':<20} | {'precondition (nodes)':<20} | {'effect (nodes)':<20} 
''')
                for skill in self.skillset.keys():
                    predicates = self.brain.skills[skill]
                    f.write(f'''{skill:<20} | {len(predicates['precondition'].nodes):<20} | {len(predicates['precondition'].nodes):<20} 
''')
                f.write((('-' * 100) + '\n'))
                f.write(f'''Executions:
''')
                f.write(f'''{'Skill Name':<20} | {'True Positive':<15} | {'True Negative':<15} | {'False Positive':<15} | {'False Negative':<15} | {'Total':<15}
''')
                f.write((('-' * 100) + '\n'))
                for (sk_name, counts) in self.skill_stats.items():
                    sumall = (((counts['true_positive'] + counts['true_negative']) + counts['false_positive']) + counts['false_negative'])
                    f.write(f'''{sk_name:<20} | {counts['true_positive']:<15} | {counts['true_negative']:<15} | {counts['false_positive']:<15} | {counts['false_negative']:<15} | {sumall:<10}
''')
                f.write((('-' * 100) + '\n'))
                for log in self.logs:
                    f.write(f'''{log}
''')
        if self.timestamps_path:
            with open(self.timestamps_path, 'w') as f:
                for stmp in self.timestamps:
                    f.write(f'''{stmp}
''')

    def record_transition(self, prev_state: np.ndarray, skill_name: str, post_state: np.ndarray, succeed: bool):
        self.transition_records.append({'prev_state': np.asarray(prev_state, dtype=np.float32), 'skill_name': str(skill_name), 'post_state': np.asarray(post_state, dtype=np.float32), 'succeed': bool(succeed), 'timestamp': time.time()})
        if ((len(self.transition_records) % 50) == 0):
            self.save_offline_dataset()

    def save_offline_dataset(self):
        if (not self.transition_records):
            return
        prev_states = np.array([r['prev_state'] for r in self.transition_records], dtype=np.float32)
        skill_names = np.array([r['skill_name'] for r in self.transition_records], dtype=object)
        post_states = np.array([r['post_state'] for r in self.transition_records], dtype=np.float32)
        succeeds = np.array([r['succeed'] for r in self.transition_records], dtype=bool)
        timestamps = np.array([r['timestamp'] for r in self.transition_records], dtype=np.float64)
        np.savez_compressed(self.dataset_path, prev_states=prev_states, skill_names=skill_names, post_states=post_states, succeeds=succeeds, timestamps=timestamps)
        print(f'[Agent] Saved {len(self.transition_records)} transitions to {self.dataset_path}')

    def run(self):
        try:
            self.run_count = 0
            self.start_time = time.time()
            self._reset_and_restart()
            while (self.run_count < self.max_runs):
                self._logger(self.run_count)
                self.run_count += 1
                print(f'[Agent] Run {self.run_count}/{self.max_runs} ---------------')
                if self.reset_requested:
                    self._reset_and_restart()
                self.bus.call_service('/supervisor/do/switchThink')
                plan = self.generate_plan(goal_skill='finish')
                self.bus.call_service('/supervisor/do/switchThink')
                if (len(plan) < 1):
                    print('[Agent] Returned empty plan.')
                    self.explore()
                    continue
                print('[Agent] Start executing plan.')
                print(f'[Agent] Plan: {plan}')
                plan_completed = True
                for (i, skill_name) in enumerate(plan):
                    print(f'[Agent] Executing step {(i + 1)}/{len(plan)}: {skill_name}')
                    state_now = self.read_state()
                    status = self.brain.predicates_status(skill_name, state_now)
                    success = self.execute_skill(skill_name)
                    if (not success):
                        print('[Agent] Failed action. Abandoning the plan.')
                        plan_completed = False
                        break
                    state_now = self.read_state()
                    status = self.brain.predicates_status(skill_name, state_now)
                    if (not (status == 'EFFECTS')):
                        print('[Agent] Effects not there. Abandoning the plan.')
                        plan_completed = False
                        break
                if plan_completed:
                    self.completed_plans_count += 1
                else:
                    self.explore()
            if self.data_buffer:
                self.bus.call_service('/supervisor/do/switchSleep')
                self.brain.update(self.data_buffer)
                self.bus.call_service('/supervisor/do/switchSleep')
                self.batch_update_count += 1
                self.data_buffer = []
        except Exception as e:
            import traceback
            print(f'[Agent] CRITICAL CRASH in run(): {e}')
            traceback.print_exc()
        finally:
            if self.data_buffer:
                self.brain.update(self.data_buffer)
                self.data_buffer = []
            self.end_time = time.time()
            self.brain._logger()
            self.save_offline_dataset()
            self._logger(self.run_count)

    def execute_skill(self, skill_name):
        print(f'[Agent] Execute Skill: {skill_name}')
        self.current_skill = skill_name
        prev_vector_state = self.read_state()
        self.brain.create_node(skill_name)
        preconditions_are_met = self.brain.preconditions_met(skill_name, prev_vector_state)
        skill = self.skillset[skill_name]
        succeed = skill.execute()
        post_vector_state = self.read_state()
        self.record_transition(prev_vector_state, skill_name, post_vector_state, succeed)
        if preconditions_are_met:
            if succeed:
                self.skill_stats[skill_name]['true_positive'] += 1
            else:
                self.skill_stats[skill_name]['false_positive'] += 1
        elif succeed:
            self.skill_stats[skill_name]['false_negative'] += 1
            self.add_to_buffer((prev_vector_state, skill_name, post_vector_state))
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
        print('[Agent] Explore')
        skill_name = random.choice(list(self.skillset.keys()))
        self.execute_skill(skill_name)
        if self.reset_requested:
            self._reset_and_restart()

    def add_to_buffer(self, data):
        self.data_buffer.append(data)
        self.timestamps.append(time.time())
        print(f'[Agent] Add data to buffer. {len(self.data_buffer)}/{self.buffer_max_len}')
        if (len(self.data_buffer) >= self.buffer_max_len):
            self.bus.call_service('/supervisor/do/switchSleep')
            self.brain.update(self.data_buffer)
            self.bus.call_service('/supervisor/do/switchSleep')
            self.batch_update_count += 1
            self.data_buffer = []

    def generate_plan(self, goal_skill=None, plan_name=None):
        print('[Agent] Generate Plan')
        vector_state = self.read_state()
        active_symbols = self.brain.get_active_symbols(vector_state)
        if (not active_symbols):
            print('[Agent] No active symbols detected.')
            return []
        print(f'[Agent] Active symbols: {active_symbols}')
        nodes = self.brain.get_nodes()
        if (not nodes):
            print('[Agent] No skill nodes available.')
            return []
        if ((goal_skill is None) or (goal_skill not in nodes)):
            goal_skill = random.choice(nodes)
        print(f'[Agent] Generating plan to achieve {goal_skill}.')
        plan_dir = os.path.join(self.pddl_dir, 'task')
        os.makedirs(plan_dir, exist_ok=True)
        domain_file = os.path.join(plan_dir, 'domain.pddl')
        problem_file = os.path.join(plan_dir, 'problem.pddl')
        plan_file = os.path.join(plan_dir, 'problem.pddl.soln')
        if (not os.path.exists(domain_file)):
            print('[Agent] Missing pre-generated domain.pddl. Creating one.')
            self.brain.generate_and_save_domain()
        problem_data = self.brain.generate_problem_pddl(active_symbols, goal_skill)
        if (not problem_data):
            print('[Agent] No problem file generated.')
            return []
        with open(problem_file, 'w') as f:
            f.write(problem_data)
        if os.path.exists(plan_file):
            os.remove(plan_file)
        try:
            subprocess.run([sys.executable, '-m', 'pyperplan', '-s', 'astar', '-H', 'hff', domain_file, problem_file], capture_output=True, text=True, check=True, timeout=10.0)
        except subprocess.TimeoutExpired:
            print('[Agent] Pyperplan timed out (no plan reachable).')
            self.logs.append(f'[Pyperplan] [RUN {self.run_count}] Pyperplan timed out.')
            return []
        except subprocess.CalledProcessError as e:
            print(f'[Agent] Pyperplan error: {(e.stderr or e.stdout)}')
            return []
        except Exception as e:
            print(f'[Agent] Unexpected error running Pyperplan: {e}')
            return []
        if (not os.path.exists(plan_file)):
            print('[Agent] Pyperplan failed (no plan file).')
            return []
        self.bus.call_service('/supervisor/do/switchThink')
        executable_plan = []
        with open(plan_file, 'r') as f:
            for line in f:
                line = line.strip()
                if ((not line) or line.startswith(';')):
                    continue
                parts = line.strip('()').split()
                if (not parts):
                    continue
                action_name_lower = parts[0].lower()
                matched_skill = next((sk for sk in self.skillset.keys() if (action_name_lower == sk.lower())), None)
                if matched_skill:
                    executable_plan.append(matched_skill)
        if (len(executable_plan) <= 0):
            print('[Agent] Error: no executable plan.')
        elif (len(executable_plan) > 6):
            if (plan_name is None):
                plan_name = f'plan_{self.run_count}'
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
            print(f'[Agent] Test {(i + 1)}/{self.N_tests}.')
            start = time.time()
            self.bus.call_service('/supervisor/do/restart')
            time.sleep(1.0)
            raw_spawn = self.bus.call_service('/supervisor/ask/position')
            if hasattr(raw_spawn, 'tolist'):
                spawn_pos = raw_spawn.tolist()
            elif isinstance(raw_spawn, (list, tuple)):
                spawn_pos = [(float(x) if hasattr(x, 'item') else x) for x in raw_spawn]
            else:
                spawn_pos = raw_spawn
            self.bus.call_service('/supervisor/do/switchThink')
            plan = self.generate_plan(goal_skill='finish', plan_name=f'test_{i}')
            self.bus.call_service('/supervisor/do/switchThink')
            test_record = {'test_idx': i, 'plan': (plan if plan else []), 'plan_length': (len(plan) if plan else 0), 'steps_executed': 0, 'last_step': '-', 'result': False, 'spawn_pos': spawn_pos, 'duration': 0.0, 'steps': []}
            if plan:
                result = True
                print(f'[Agent] Executing plan: {plan}')
                for (step, skill_name) in enumerate(plan):
                    print(f'[Agent] Executing step {(step + 1)}/{len(plan)}: {skill_name}')
                    state_now = self.read_state()
                    status = self.brain.predicates_status(skill_name, state_now)
                    if (not (status == 'PRECONDITIONS')):
                        print(f'[Agent] Preconditions are not met: {skill_name}')
                        test_record['steps'].append({'step': step, 'skill': skill_name, 'succeed': False, 'reason': 'preconditions_failed'})
                        test_record['steps_executed'] = step
                        result = False
                        break
                    skill = self.skillset[skill_name]
                    succeed = skill.execute()
                    test_record['last_step'] = skill_name
                    test_record['steps_executed'] = (step + 1)
                    state_now = self.read_state()
                    status = self.brain.predicates_status(skill_name, state_now)
                    if (not (status == 'EFFECTS')):
                        print(f'[Agent] Preconditions are not met: {skill_name}')
                        test_record['steps'].append({'step': step, 'skill': skill_name, 'succeed': False, 'reason': 'effects_failed'})
                        result = False
                        break
                    test_record['steps'].append({'step': step, 'skill': skill_name, 'succeed': bool(succeed), 'reason': 'execution'})
                    if (not succeed):
                        print(f'[Agent] Skill {skill_name} failed during execution.')
                        result = False
                        break
                test_record['result'] = result
            test_record['duration'] = (time.time() - start)
            self.tests_data[test_key] = test_record
            if test_record['result']:
                print('[Agent] Test succeed.')
            else:
                print('[Agent] Test failed.')
        self._test_logger()

    def _test_logger(self):
        if (not self.tests_data):
            print('[Agent] No test records to log.')
            return
        tests_list = list(self.tests_data.values())
        n_tests = len(tests_list)
        plan_lengths = [t['plan_length'] for t in tests_list]
        steps_executed = [t['steps_executed'] for t in tests_list]
        durations = [t['duration'] for t in tests_list]
        results = [(1 if t['result'] else 0) for t in tests_list]
        successful_spawns = [t['spawn_pos'] for t in tests_list if (t['result'] and (t['spawn_pos'] is not None))]
        acceptable_start_plane = None
        if successful_spawns:
            try:
                dims = (len(successful_spawns[0]) if isinstance(successful_spawns[0], (list, tuple)) else 1)
                if (dims > 1):
                    min_bounds = [float(min((p[d] for p in successful_spawns))) for d in range(dims)]
                    max_bounds = [float(max((p[d] for p in successful_spawns))) for d in range(dims)]
                else:
                    min_bounds = [float(min(successful_spawns))]
                    max_bounds = [float(max(successful_spawns))]
                acceptable_start_plane = {'min_bounds': min_bounds, 'max_bounds': max_bounds, 'successful_positions': successful_spawns}
            except Exception:
                acceptable_start_plane = {'successful_positions': successful_spawns}
        plan_frequencies = {}
        transition_outcomes = {}
        for t in tests_list:
            plan_tuple = tuple(t.get('plan', []))
            if plan_tuple:
                plan_str = ' -> '.join(plan_tuple)
                if (plan_str not in plan_frequencies):
                    plan_frequencies[plan_str] = {'total': 0, 'succeeded': 0, 'failed': 0}
                plan_frequencies[plan_str]['total'] += 1
                if t['result']:
                    plan_frequencies[plan_str]['succeeded'] += 1
                else:
                    plan_frequencies[plan_str]['failed'] += 1
            steps = t.get('steps', [])
            for k in range(len(steps)):
                current_sk = steps[k]['skill']
                succeed = steps[k]['succeed']
                trans_key = (f'Start -> {current_sk}' if (k == 0) else f"{steps[(k - 1)]['skill']} -> {current_sk}")
                if (trans_key not in transition_outcomes):
                    transition_outcomes[trans_key] = {'attempts': 0, 'failed': 0}
                transition_outcomes[trans_key]['attempts'] += 1
                if (not succeed):
                    transition_outcomes[trans_key]['failed'] += 1
        success_rate = (mean(results) if results else 0.0)
        mean_plan_len = (mean(plan_lengths) if plan_lengths else 0.0)
        mode_plan_len = (multimode(plan_lengths) if plan_lengths else [])
        mean_steps = (mean(steps_executed) if steps_executed else 0.0)
        mode_steps = (multimode(steps_executed) if steps_executed else [])
        mean_dur = (mean(durations) if durations else 0.0)
        summary = {'overall_statistics': {'total_tests': n_tests, 'success_rate': success_rate, 'mean_plan_length': mean_plan_len, 'mode_plan_length': mode_plan_len, 'mean_steps_executed': mean_steps, 'mode_steps_executed': mode_steps, 'mean_execution_time_s': mean_dur, 'acceptable_start_position_plane': acceptable_start_plane}, 'plan_frequencies': plan_frequencies, 'transition_outcomes': transition_outcomes, 'tests': self.tests_data}
        total_passed = sum(results)
        total_failed = (n_tests - total_passed)
        lines = [('=' * 60), '                    AGENT TEST SUMMARY', ('=' * 60), '', 'OVERALL PERFORMANCE', ('-' * 40), f'Total Tests Run       : {n_tests}', f'Passed / Failed       : {total_passed} / {total_failed}', f'Success Rate          : {(success_rate * 100):.2f}%', f'Mean Duration         : {mean_dur:.2f} s', f'Mean Plan Length      : {mean_plan_len:.2f} (modes: {mode_plan_len})', f'Mean Steps Executed   : {mean_steps:.2f} (modes: {mode_steps})', '']
        if (acceptable_start_plane and ('min_bounds' in acceptable_start_plane)):
            min_b = [f'{b:.2f}' for b in acceptable_start_plane['min_bounds']]
            max_b = [f'{b:.2f}' for b in acceptable_start_plane['max_bounds']]
            lines.extend(['SUCCESSFUL SPAWN ENVELOPE', ('-' * 40), f"Min Bounds : [{', '.join(min_b)}]", f"Max Bounds : [{', '.join(max_b)}]", f"Total Valid Spawns : {len(acceptable_start_plane.get('successful_positions', []))}", ''])
        lines.extend(['PLAN SEQUENCES (Sorted by Frequency)', ('-' * 40)])
        sorted_plans = sorted(plan_frequencies.items(), key=(lambda kv: kv[1]['total']), reverse=True)
        if sorted_plans:
            for (plan_str, counts) in sorted_plans:
                pct = (((counts['succeeded'] / counts['total']) * 100) if counts['total'] else 0.0)
                lines.append(f"• [{counts['total']}x] {plan_str}")
                lines.append(f"    Passed: {counts['succeeded']} | Failed: {counts['failed']} | Success: {pct:.1f}%")
        else:
            lines.append('  (No plans recorded)')
        lines.append('')
        lines.extend(['TRANSITION RELIABILITY (Sorted by Fail Rate)', ('-' * 40)])
        sorted_trans = sorted(transition_outcomes.items(), key=(lambda kv: ((kv[1]['failed'] / kv[1]['attempts']) if kv[1]['attempts'] else 0.0)), reverse=True)
        if sorted_trans:
            for (trans_str, counts) in sorted_trans:
                att = counts['attempts']
                fail = counts['failed']
                fail_rate = (((fail / att) * 100) if att else 0.0)
                lines.append(f'• {trans_str:<32} -> {(att - fail)}/{att} ok ({fail_rate:.1f}% fail rate)')
        else:
            lines.append('  (No transitions recorded)')
        lines.append('')
        lines.append(('=' * 60))
        txt_summary = '\n'.join(lines)

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
            with open(self.test_stats_path, 'w') as f:
                json.dump(summary, f, indent=4, cls=NumpyEncoder)
            print(f'[Agent] Saved test data to {self.test_stats_path}')
            (base_root, _) = os.path.splitext(self.test_stats_path)
            txt_path = f'{base_root}_summary.txt'
            with open(txt_path, 'w') as f:
                f.write(txt_summary)
            print(f'[Agent] Saved readable test summary to {txt_path}')
def agent_loop(bus: Communicator, agent: Agent, config: dict, stop_event: threading.Event):
    runs_per_epoch = config.get('runs_per_epoch', 100)
    tests_between = config.get('tests_between', 0)
    tests_at_end = config.get('tests_at_end', 0)
    time_limit_hours = config.get('time_limit', 2.0)
    min_success_rate = config.get('min_success_rate', 0.7)
    max_epochs = config.get('max_epochs', 100)
    session_dir = config.get('session_dir', agent.log_dir)
    time_limit_seconds = ((time_limit_hours * 3600) if time_limit_hours else float('inf'))
    start_time = time.time()
    epoch_idx = 0
    final_success_rate = 0.0
    print('[Agent] Starting multi-epoch agent loop.')
    try:
        while (epoch_idx < max_epochs):
            elapsed = (time.time() - start_time)
            if (elapsed >= time_limit_seconds):
                print(f'[Agent] Time limit reached ({(elapsed / 3600):.2f} h). Stopping epochs.')
                break
            print(f'''
{('=' * 30)} EPOCH {epoch_idx:03d} {('=' * 30)}''')
            epoch_dir = os.path.join(session_dir, f'epoch_{epoch_idx:03d}')
            agent.set_epoch_output_dir(epoch_dir)
            agent._reset_and_restart()
            if (runs_per_epoch > 0):
                agent.max_runs = runs_per_epoch
                agent.run()
            if (tests_between > 0):
                agent.N_tests = tests_between
                agent.test_run()
                final_success_rate = _read_success_rate(agent.test_stats_path)
                print(f'[Agent] Epoch {epoch_idx} Success Rate: {(final_success_rate * 100):.2f}%')
                if (final_success_rate >= min_success_rate):
                    print(f'[Agent] Target success rate {(min_success_rate * 100):.1f}% reached!')
                    epoch_idx += 1
                    break
            epoch_idx += 1
        if (tests_at_end > 0):
            print(f'''
{('=' * 30)} FINAL TESTS ({tests_at_end} runs) {('=' * 30)}''')
            final_epoch_dir = os.path.join(session_dir, f'epoch_{epoch_idx:03d}_final_tests')
            agent.set_epoch_output_dir(final_epoch_dir)
            agent._reset_and_restart()
            agent.N_tests = tests_at_end
            agent.test_run()
            final_success_rate = _read_success_rate(agent.test_stats_path)
            print(f'[Agent] Final Evaluation Success Rate: {(final_success_rate * 100):.2f}%')
    except Exception as e:
        import traceback
        print(f'[Agent Loop] Error during execution: {e}')
        traceback.print_exc()
    finally:
        print('[Agent] Multi-epoch execution completed. Setting stop_event.')
        stop_event.set()
