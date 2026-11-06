import json
import os
import numpy as np
from statistics import mean, multimode
from .base import BaseLogger

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

class TestSummaryLogger(BaseLogger):
    def log_results(self, tests_data: dict, json_filename: str = "test_statistics.json", txt_filename: str = None):
        if not tests_data:
            print("[Logger] No test records to log.")
            return

        tests_list = list(tests_data.values())
        n_tests = len(tests_list)
        plan_lengths = [t['plan_length'] for t in tests_list]
        steps_executed = [t['steps_executed'] for t in tests_list]
        durations = [t['duration'] for t in tests_list]
        results = [1 if t['result'] else 0 for t in tests_list]
        successful_spawns = [t['spawn_pos'] for t in tests_list if t['result'] and t['spawn_pos'] is not None]

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
                acceptable_start_plane = {'min_bounds': min_bounds, 'max_bounds': max_bounds, 'successful_positions': successful_spawns}
            except Exception:
                acceptable_start_plane = {'successful_positions': successful_spawns}

        plan_frequencies = {}
        transition_outcomes = {}
        for t in tests_list:
            plan_tuple = tuple(t.get('plan', []))
            if plan_tuple:
                plan_str = ' -> '.join(plan_tuple)
                if plan_str not in plan_frequencies:
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
                trans_key = f"Start -> {current_sk}" if k == 0 else f"{steps[k - 1]['skill']} -> {current_sk}"
                if trans_key not in transition_outcomes:
                    transition_outcomes[trans_key] = {'attempts': 0, 'failed': 0}
                transition_outcomes[trans_key]['attempts'] += 1
                if not succeed:
                    transition_outcomes[trans_key]['failed'] += 1

        success_rate = mean(results) if results else 0.0
        mean_plan_len = mean(plan_lengths) if plan_lengths else 0.0
        mode_plan_len = multimode(plan_lengths) if plan_lengths else []
        mean_steps = mean(steps_executed) if steps_executed else 0.0
        mode_steps = multimode(steps_executed) if steps_executed else []
        mean_dur = mean(durations) if durations else 0.0

        summary = {
            'overall_statistics': {
                'total_tests': n_tests,
                'success_rate': success_rate,
                'mean_plan_length': mean_plan_len,
                'mode_plan_length': mode_plan_len,
                'mean_steps_executed': mean_steps,
                'mode_steps_executed': mode_steps,
                'mean_execution_time_s': mean_dur,
                'acceptable_start_position_plane': acceptable_start_plane
            },
            'plan_frequencies': plan_frequencies,
            'transition_outcomes': transition_outcomes,
            'tests': tests_data
        }

        total_passed = sum(results)
        total_failed = n_tests - total_passed
        lines = [
            '=' * 60,
            '                    AGENT TEST SUMMARY',
            '=' * 60, '',
            'OVERALL PERFORMANCE', '-' * 40,
            f'Total Tests Run       : {n_tests}',
            f'Passed / Failed       : {total_passed} / {total_failed}',
            f'Success Rate          : {success_rate * 100:.2f}%',
            f'Mean Duration         : {mean_dur:.2f} s',
            f'Mean Plan Length      : {mean_plan_len:.2f} (modes: {mode_plan_len})',
            f'Mean Steps Executed   : {mean_steps:.2f} (modes: {mode_steps})', ''
        ]

        if acceptable_start_plane and 'min_bounds' in acceptable_start_plane:
            min_b = [f"{b:.2f}" for b in acceptable_start_plane['min_bounds']]
            max_b = [f"{b:.2f}" for b in acceptable_start_plane['max_bounds']]
            lines.extend([
                'SUCCESSFUL SPAWN ENVELOPE', '-' * 40,
                f"Min Bounds : [{', '.join(min_b)}]",
                f"Max Bounds : [{', '.join(max_b)}]",
                f"Total Valid Spawns : {len(acceptable_start_plane.get('successful_positions', []))}", ''
            ])

        lines.extend(['PLAN SEQUENCES (Sorted by Frequency)', '-' * 40])
        sorted_plans = sorted(plan_frequencies.items(), key=lambda kv: kv[1]['total'], reverse=True)
        if sorted_plans:
            for plan_str, counts in sorted_plans:
                pct = (counts['succeeded'] / counts['total'] * 100) if counts['total'] else 0.0
                lines.append(f"• [{counts['total']}x] {plan_str}")
                lines.append(f"    Passed: {counts['succeeded']} | Failed: {counts['failed']} | Success: {pct:.1f}%")
        else:
            lines.append('  (No plans recorded)')
        lines.append('')

        lines.extend(['TRANSITION RELIABILITY (Sorted by Fail Rate)', '-' * 40])
        sorted_trans = sorted(transition_outcomes.items(), key=lambda kv: (kv[1]['failed'] / kv[1]['attempts']) if kv[1]['attempts'] else 0.0, reverse=True)
        if sorted_trans:
            for trans_str, counts in sorted_trans:
                att, fail = counts['attempts'], counts['failed']
                fail_rate = (fail / att * 100) if att else 0.0
                lines.append(f"• {trans_str:<32} -> {att - fail}/{att} ok ({fail_rate:.1f}% fail rate)")
        else:
            lines.append('  (No transitions recorded)')
        lines.append('')
        lines.append('=' * 60)
        
        txt_summary = '\n'.join(lines)

        json_path = self.get_path(json_filename)
        if txt_filename is None:
            base_root, _ = os.path.splitext(json_path)
            txt_path = f"{base_root}_summary.txt"
        else:
            txt_path = self.get_path(txt_filename)

        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=4, cls=NumpyEncoder)
        print(f"[Logger] Saved test data to {json_path}")
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(txt_summary)
        print(f"[Logger] Saved readable test summary to {txt_path}")
