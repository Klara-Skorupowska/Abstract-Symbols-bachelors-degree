import os, cv2, numpy as np
from pathlib import Path
from supervisor.core.picture_generator import PictureGenerator
from supervisor.utils.dir_utils import collect_chain_folders

class ExecStatsGenerator(PictureGenerator):
    SKILL_PALETTE = [(195, 105, 30), (71, 90, 224), (74, 168, 43), (6, 119, 217), (246, 92, 139), (212, 182, 6), (153, 72, 236), (139, 116, 100)]

    @staticmethod
    def parse_execution_statistics(file_path: Path) -> dict:
        ret = {'collected_data': {}, 'executions': {}, 'tp': {}, 'tn': {}, 'correct_counts': {}, 'incorrect_counts': {}}
        if not file_path.exists(): return ret
        mode = None
        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith('---') or line_str.startswith('start time:') or line_str.startswith('CONTINUATION'): continue
                if 'Collected Data:' in line_str: mode = 'collected_data'; continue
                elif 'Executions:' in line_str: mode = 'executions'; continue
                parts = [p.strip() for p in line_str.split('|')]
                if len(parts) >= 2 and parts[0].lower().startswith('skill name'): continue
                skill = parts[0]
                if mode == 'collected_data' and len(parts) >= 2:
                    try: ret['collected_data'][skill] = int(parts[1])
                    except ValueError: pass
                elif mode == 'executions' and len(parts) >= 6:
                    try:
                        tp_val, tn_val, fp_val, fn_val, total_val = map(int, parts[1:6])
                        both = max(tp_val + tn_val, 1)
                        ret['tp'][skill] = tp_val / both
                        ret['tn'][skill] = tn_val / both
                        ret['executions'][skill] = total_val
                        ret['correct_counts'][skill] = tp_val + tn_val
                        ret['incorrect_counts'][skill] = fp_val + fn_val
                    except ValueError: pass
        return ret

    def run(self):
        if not self.session_path or not os.path.exists(self.session_path): return
        file_path = Path(self.session_path) / 'execution_statistics.txt'
        if not file_path.exists(): return
        try:
            chain_dirs = collect_chain_folders(Path(self.session_path), Path(self.logs_root))
            all_skills, session_stats = [], []
            for s_dir in chain_dirs:
                stats = self.parse_execution_statistics(s_dir / 'execution_statistics.txt')
                session_stats.append(stats)
                for skill in list(stats['collected_data'].keys()) + list(stats['executions'].keys()):
                    if skill not in all_skills: all_skills.append(skill)
            data_nodes_by_skill = {skill: [] for skill in all_skills}
            for stats in session_stats:
                for skill in all_skills: data_nodes_by_skill[skill].append(stats['collected_data'].get(skill, 0))
            out_dir = self.get_pictures_dir() or '.'
            print(f'[ExecStatsGenerator] Parsed {len(session_stats)} epochs/sessions successfully!')
        except Exception as e:
            print(f'[ExecStatsGenerator] Failed: {e}')
