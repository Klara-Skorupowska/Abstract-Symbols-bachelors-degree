import os
import time
from collections import Counter
import numpy as np
from .base import BaseLogger

class SymbolStatsLogger(BaseLogger):
    def export_statistics(self, state_dim, symbols_map, skill_map, brain_skills, filename="symbol_statistics.txt"):
        file_path = self.get_path(filename)
        if not file_path:
            return
            
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        lines = [
            '=' * 70,
            'SYMBOL & SKILL ABSTRACTION STATISTICS',
            '=' * 70,
            f'State Dimension: {state_dim}',
            f'Total Aggregated Symbols: {len(symbols_map)}',
            '-' * 70,
            'SYMBOLS:'
        ]
        
        for sym_name in sorted(symbols_map.keys()):
            pred = symbols_map[sym_name]
            lines.append(f"  {sym_name:<10} | dims: {len(getattr(pred, 'active_dimensions', []))} | nodes: {len(pred.nodes)}")
        
        lines.append('-' * 70)
        lines.append('SKILL INTERACTIONS:')
        for skill_name in sorted(brain_skills.keys()):
            roles = skill_map.get(skill_name, {'precondition': set(), 'effect': set()})
            pre_syms = sorted(roles['precondition'])
            eff_syms = sorted(roles['effect'])
            lines.append(f"Skill: {skill_name}")
            lines.append(f"  - Precondition Symbols : {pre_syms}")
            lines.append(f"  - Effect Symbols       : {eff_syms}")
            lines.append('-' * 70)
            
        temp_file = f"{file_path}.tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
            
        for _ in range(5):
            try:
                os.replace(temp_file, file_path)
                break
            except (PermissionError, OSError):
                time.sleep(0.05)

    def log_continuous_stats(self, symbols_map, skill_map, total_skills: int, filename="symbol_stats_log.txt") -> str:
        target_path = self.get_path(filename)
        
        total_symbols = len(symbols_map)
        
        if total_symbols > 0:
            dim_counts = [len(getattr(pred, 'active_dimensions', [])) for pred in symbols_map.values()]
            counter = Counter(dim_counts)
            min_val = min(counter.keys())
            min_dim_str = f"{min_val} ({counter[min_val]})"
            max_val = max(counter.keys())
            max_dim_str = f"{max_val} ({counter[max_val]})"
            dominant_val, dominant_count = counter.most_common(1)[0]
            dominant_dim_str = f"{dominant_val} ({dominant_count})"
        else:
            min_dim_str, dominant_dim_str, max_dim_str = "0 (0)", "0 (0)", "0 (0)"
            
        if total_skills > 0:
            pre_counts = [len(roles.get('precondition', set())) for roles in skill_map.values()]
            eff_counts = [len(roles.get('effect', set())) for roles in skill_map.values()]
            avg_pre_symbols = np.mean(pre_counts)
            avg_eff_symbols = np.mean(eff_counts)
        else:
            avg_pre_symbols = 0.0
            avg_eff_symbols = 0.0
            
        if total_symbols > 0:
            symbol_usage = Counter()
            for roles in skill_map.values():
                for sym in set(roles.get('precondition', set())) | set(roles.get('effect', set())):
                    symbol_usage[sym] += 1
            reusability_ratio = sum(symbol_usage.values()) / total_symbols
        else:
            reusability_ratio = 0.0
            
        header = "timestamp,total_symbols,total_skills,min_dims,dominant_dims,max_dims,avg_pre_symbols,avg_eff_symbols,reusability_ratio\n"
        row = f"{time.time():.2f},{total_symbols},{total_skills},\"{min_dim_str}\",\"{dominant_dim_str}\",\"{max_dim_str}\",{avg_pre_symbols:.2f},{avg_eff_symbols:.2f},{reusability_ratio:.2f}\n"
        
        file_exists = os.path.exists(target_path) and os.path.getsize(target_path) > 0
        with open(target_path, 'a', encoding='utf-8') as f:
            if not file_exists:
                f.write(header)
            f.write(row)
        return row.strip()
