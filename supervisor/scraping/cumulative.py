import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from supervisor.utils.dir_utils import collect_chain_folders

@dataclass
class SkillFinalStats:
    nodes: int
    ratio: float | None
    pos_ratio: float | None
    neg_ratio: float | None

@dataclass
class CumulativeExecutionStats:
    num_sessions: int = 0
    total_duration_sec: int = 0
    total_runs: int = 0
    total_brain_updates: int = 0
    total_completed_plans: int = 0
    final_skills_data: dict[str, SkillFinalStats] = field(default_factory=dict)

class CumulativeStatsScraper:
    def __init__(self, logs_root: str='logs', session_dir: str | None=None):
        self.logs_root = logs_root
        self.session_path = session_dir

    @staticmethod
    def parse_duration_to_seconds(duration_str: str) -> int:
        parts = list(map(int, duration_str.strip().split(':')))
        if len(parts) == 3: return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2: return parts[0] * 60 + parts[1]
        return parts[0]

    @staticmethod
    def format_total_seconds(total_seconds: int) -> str:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

    def parse_session_aggregates(self, stats_file: Path) -> tuple[int, int, int, int]:
        duration_sec, runs, brain_updates, completed_plans = 0, 0, 0, 0
        duration_pat = re.compile(r'duration:\s*([0-9:]+)', re.IGNORECASE)
        stats_pat = re.compile(r'runs:\s*(\d+);\s*brain updates:\s*(\d+);\s*completed plans:\s*(\d+)', re.IGNORECASE)
        with stats_file.open('r', encoding='utf-8') as f:
            for line in f:
                d_match = duration_pat.search(line)
                if d_match: duration_sec = self.parse_duration_to_seconds(d_match.group(1))
                s_match = stats_pat.search(line)
                if s_match:
                    runs, brain_updates, completed_plans = int(s_match.group(1)), int(s_match.group(2)), int(s_match.group(3))
        return duration_sec, runs, brain_updates, completed_plans

    def parse_final_session_skills(self, stats_file: Path) -> dict[str, SkillFinalStats]:
        skills_nodes, skills_ratios, skills_positive_ratios, skills_negative_ratios = {}, {}, {}, {}
        mode = None
        with stats_file.open('r', encoding='utf-8') as f:
            for line in f:
                raw_line = line.strip()
                if not raw_line or raw_line.startswith('---'): continue
                if 'Collected Data:' in raw_line:
                    mode = 'collected_data'; continue
                elif 'Executions:' in raw_line:
                    mode = 'executions'; continue
                parts = [p.strip() for p in raw_line.split('|')]
                if len(parts) < 2 or parts[0].lower() == 'skill name': continue
                skill_name = parts[0]
                if mode == 'collected_data' and len(parts) >= 2:
                    try: skills_nodes[skill_name] = int(parts[1])
                    except ValueError: continue
                elif mode == 'executions' and len(parts) >= 5:
                    try:
                        tp, tn, fp, fn = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        okay = (fn + tp + fp + tn) > 0
                        skills_ratios[skill_name] = (tp + tn) / (fn + tp + fp + tn) if okay else None
                        skills_positive_ratios[skill_name] = tp / (tp + fn) if (tp + fn) > 0 else None
                        skills_negative_ratios[skill_name] = tn / (tn + fp) if (tn + fp) > 0 else None
                    except ValueError: continue
        all_skills = set(skills_nodes.keys()).union(skills_ratios.keys())
        return {skill: SkillFinalStats(
            nodes=skills_nodes.get(skill, 0),
            ratio=skills_ratios.get(skill, None),
            pos_ratio=skills_positive_ratios.get(skill, None),
            neg_ratio=skills_negative_ratios.get(skill, None)
        ) for skill in all_skills}

    def scrape(self) -> CumulativeExecutionStats | None:
        if not self.session_path or not os.path.exists(self.session_path):
            return None
        chain_dirs = collect_chain_folders(Path(self.session_path), Path(self.logs_root))
        if not chain_dirs: return None
        cumulative = CumulativeExecutionStats(num_sessions=len(chain_dirs))
        for s_dir in chain_dirs:
            s_file = s_dir / 'execution_statistics.txt'
            if not s_file.exists(): continue
            d_sec, runs, updates, plans = self.parse_session_aggregates(s_file)
            cumulative.total_duration_sec += d_sec
            cumulative.total_runs += runs
            cumulative.total_brain_updates += updates
            cumulative.total_completed_plans += plans
        final_stats_file = chain_dirs[-1] / 'execution_statistics.txt'
        if final_stats_file.exists():
            cumulative.final_skills_data = self.parse_final_session_skills(final_stats_file)
        return cumulative

    def save_to_txt(self, stats: CumulativeExecutionStats, output_path: str | Path) -> None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            '-'*60, 'CUMULATIVE EXECUTION STATISTICS', '-'*60,
            f'number of sessions: {stats.num_sessions}',
            f'total duration: {self.format_total_seconds(stats.total_duration_sec)}',
            f'total number of runs: {stats.total_runs}',
            f'total number of brain updates: {stats.total_brain_updates}',
            f'total number of completed plans: {stats.total_completed_plans}',
            '-'*60, 'FINAL SESSION SKILLS', '-'*100,
            f"{'Skill Name':<20} | {'Nodes':<10} | {'true/all ratio':<20} | {'true success ratio':<20} | {'true fail ratio':<20} ",
            '-'*100
        ]
        for skill, data in stats.final_skills_data.items():
            r = f'{data.ratio:.3f}' if data.ratio is not None else 'N/A'
            p = f'{data.pos_ratio:.3f}' if data.pos_ratio is not None else 'N/A'
            n = f'{data.neg_ratio:.3f}' if data.neg_ratio is not None else 'N/A'
            lines.append(f'{skill:<20} | {data.nodes:<10} | {r:<20} | {p:<20} | {n:<20}')
        lines.append('-'*100 + '\n')
        with output_file.open('w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def export_stats(self, output_filename: str='cumulative_statistics.txt') -> Path | None:
        stats = self.scrape()
        if not stats or not self.session_path: return None
        out_path = Path(self.session_path) / output_filename
        self.save_to_txt(stats, out_path)
        return out_path
