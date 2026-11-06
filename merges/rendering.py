from __future__ import annotations
from pathlib import Path
from supervisor.picture_generator import PictureGenerator
from communicator import Communicator
from .renderer import Renderer
from .picture_generator import get_session
from supervisor.renderer import Renderer
from datetime import datetime
from .live_monitor import LiveMonitor
from .trans_graph_live import TransGraphLive
from .sensimotor_live import LiveSensimotorMonitor
from .symbols_live import SymbolsLive
from supervisor.symbol_pic_gen import SymbolIconPictureGenerator, SymbolCategoryPictureGenerator
from supervisor.trans_graph_pic_gen import TransGraphPictureGenerator
from supervisor.update_freq_pic_gen import UpdateFrequencyPictureGenerator
from supervisor.update_freq_cumulative_pic_gen import HistogramPlotGenerator
from supervisor.exec_stats_cumulative_pic_gen import ExecStatsGenerator
from supervisor.symbol_stats_cumulative_pic_gen import SymbolStatsGenerator
from supervisor.renderer import lower_median
from .picture_generator import PictureGenerator
from dataclasses import dataclass, field
from scipy.stats import gaussian_kde
import os, re, cv2, numpy as np, textwrap, math, networkx as nx, glob, json, csv
class ExecStatsGenerator(PictureGenerator):
    SKILL_PALETTE = [(195, 105, 30), (71, 90, 224), (74, 168, 43), (6, 119, 217), (246, 92, 139), (212, 182, 6), (153, 72, 236), (139, 116, 100)]

    def __init__(self, logs_root: str='logs', session_dir: (str | None)=None):
        super().__init__(logs_root=logs_root, session_dir=session_dir)

    @staticmethod
    def parse_continuation(stats_file: Path) -> (Path | None):
        if (not stats_file.exists()):
            return None
        pattern = re.compile('CONTINUATION FROM SESSION:\\s*(.+)', re.IGNORECASE)
        with stats_file.open('r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line.strip())
                if match:
                    return Path(match.group(1).strip())
        return None

    def collect_chain_folders(self, current_dir: Path, logs_root: Path) -> list[Path]:
        visited_dirs = {current_dir.resolve()}
        chain_dirs: list[Path] = [current_dir]
        curr = current_dir
        while True:
            stats_file = (curr / 'execution_statistics.txt')
            continued_rel = self.parse_continuation(stats_file)
            if (not continued_rel):
                break
            if (logs_root.parent / continued_rel).exists():
                next_dir = (logs_root.parent / continued_rel)
            elif (logs_root / continued_rel.name).exists():
                next_dir = (logs_root / continued_rel.name)
            else:
                break
            next_resolved = next_dir.resolve()
            if (next_resolved in visited_dirs):
                break
            visited_dirs.add(next_resolved)
            chain_dirs.append(next_dir)
            curr = next_dir
        chain_dirs.reverse()
        return chain_dirs

    @staticmethod
    def parse_execution_statistics(file_path: Path) -> dict:
        if (not file_path.exists()):
            return {'collected_data': {}, 'executions': {}, 'tp': {}, 'tn': {}, 'correct_counts': {}, 'incorrect_counts': {}}
        collected_data = {}
        executions = {}
        tp = {}
        tn = {}
        correct_counts = {}
        incorrect_counts = {}
        mode = None
        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                line_str = line.strip()
                if (not line_str):
                    continue
                if ('Collected Data:' in line_str):
                    mode = 'collected_data'
                    continue
                elif ('Executions:' in line_str):
                    mode = 'executions'
                    continue
                elif (line_str.startswith('---') or line_str.startswith('start time:') or line_str.startswith('CONTINUATION')):
                    continue
                parts = [p.strip() for p in line_str.split('|')]
                if ((len(parts) >= 2) and parts[0].lower().startswith('skill name')):
                    continue
                if ((mode == 'collected_data') and (len(parts) >= 2)):
                    skill = parts[0]
                    try:
                        collected_data[skill] = int(parts[1])
                    except ValueError:
                        pass
                elif ((mode == 'executions') and (len(parts) >= 6)):
                    skill = parts[0]
                    try:
                        tp_val = int(parts[1])
                        tn_val = int(parts[2])
                        fp_val = int(parts[3])
                        fn_val = int(parts[4])
                        total_val = int(parts[5])
                        both = max((tp_val + tn_val), 1)
                        tp[skill] = (tp_val / both)
                        tn[skill] = (tn_val / both)
                        executions[skill] = total_val
                        correct_counts[skill] = (tp_val + tn_val)
                        incorrect_counts[skill] = (fp_val + fn_val)
                    except ValueError:
                        pass
        return {'collected_data': collected_data, 'executions': executions, 'tp': tp, 'tn': tn, 'correct_counts': correct_counts, 'incorrect_counts': incorrect_counts}

    def render_trend_plot(self, n_sessions: int, skills_data: dict[(str, list[int])], title: str, y_label: str, width: int=1000, height: int=620) -> np.ndarray:
        img = self.renderer.create_base_canvas(title.upper(), width=width, height=height)
        (plot_x, plot_y) = (90, 110)
        (plot_w, plot_h) = (710, 430)
        cv2.rectangle(img, (plot_x, plot_y), ((plot_x + plot_w), (plot_y + plot_h)), self.renderer.PANEL_BORDER, 1)
        all_vals = [v for vals in skills_data.values() for v in vals]
        max_val = (max((max(all_vals) * 1.05), 1.0) if all_vals else 1.0)
        grid_steps = 4
        for i in range((grid_steps + 1)):
            y_val = ((plot_y + plot_h) - int((i * (plot_h / grid_steps))))
            cv2.line(img, (plot_x, y_val), ((plot_x + plot_w), y_val), self.renderer.tertiary_color, 1, cv2.LINE_AA)
            cv2.putText(img, f'{int((i * (max_val / grid_steps)))}', ((plot_x - 45), (y_val + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.renderer.neutral_gray, 1, cv2.LINE_AA)

        def get_x_px(s_idx: int) -> int:
            if (n_sessions <= 1):
                return (plot_x + (plot_w // 2))
            return (plot_x + int(((s_idx / (n_sessions - 1)) * plot_w)))
        tick_sessions = ([1] + [s for s in range(10, (n_sessions + 1), 10)])
        for s_num in tick_sessions:
            tx = get_x_px((s_num - 1))
            cv2.line(img, (tx, (plot_y + plot_h)), (tx, ((plot_y + plot_h) + 5)), self.renderer.BORDER_COLOR, 1, cv2.LINE_AA)
            cv2.putText(img, str(s_num), ((tx - 8), ((plot_y + plot_h) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        legend_x = ((plot_x + plot_w) + 20)
        legend_y = (plot_y + 20)
        for (idx, (skill_name, vals)) in enumerate(skills_data.items()):
            color = self.SKILL_PALETTE[(idx % len(self.SKILL_PALETTE))]
            pts = []
            for (s_idx, val) in enumerate(vals):
                xi = get_x_px(s_idx)
                yi = ((plot_y + plot_h) - int(((val / max_val) * plot_h)))
                pts.append((xi, yi))
            for i in range((len(pts) - 1)):
                cv2.line(img, pts[i], pts[(i + 1)], color, 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(img, pt, 3, color, (- 1), cv2.LINE_AA)
            ly = (legend_y + (idx * 24))
            if (ly < (height - 60)):
                cv2.rectangle(img, (legend_x, (ly - 8)), ((legend_x + 12), (ly + 4)), color, (- 1))
                cv2.putText(img, skill_name[:16], ((legend_x + 18), (ly + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.putText(img, 'Session', (((plot_x + (plot_w // 2)) - 25), ((plot_y + plot_h) + 45)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.putText(img, y_label, ((plot_x - 30), (plot_y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        return img

    def render_stacked_outcomes_count(self, sessions: (int | list), all_skills: list[str], correct_data: dict[(str, list[int])], incorrect_data: dict[(str, list[int])], title: str='Skill Executions Count', width: int=1000) -> np.ndarray:
        n_sessions = (len(sessions) if isinstance(sessions, list) else int(sessions))
        n_skills = len(all_skills)
        row_height = 80
        total_h = max(620, (130 + (n_skills * row_height)))
        img = self.renderer.create_base_canvas(title.upper(), width=width, height=total_h)
        if (n_skills == 0):
            return img
        (plot_x, start_y) = (150, 100)
        plot_w = 780
        bar_w = max(int(((plot_w / n_sessions) * 0.7)), 2)
        color_correct = self.renderer.status_green
        color_incorrect = self.renderer.status_red
        leg_y = 45
        cv2.rectangle(img, (630, (leg_y - 10)), (645, (leg_y + 2)), color_correct, (- 1))
        cv2.putText(img, 'Expected Outcome', (652, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.rectangle(img, (800, (leg_y - 10)), (815, (leg_y + 2)), color_incorrect, (- 1))
        cv2.putText(img, 'Unexpected Outcome', (822, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        global_max = 1.0
        for skill in all_skills:
            c_vals = correct_data.get(skill, [])
            inc_vals = incorrect_data.get(skill, [])
            for (c, inc) in zip(c_vals, inc_vals):
                total = (c + inc)
                if (total > global_max):
                    global_max = float(total)
        common_y_max = max((global_max * 1.15), 1.0)
        for (s_idx, skill) in enumerate(all_skills):
            cur_y = (start_y + (s_idx * row_height))
            sub_h = (row_height - 20)
            cv2.rectangle(img, (plot_x, cur_y), ((plot_x + plot_w), (cur_y + sub_h)), self.renderer.PANEL_BORDER, 1)
            cv2.putText(img, skill[:16], (35, ((cur_y + (sub_h // 2)) + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
            cv2.putText(img, f'{int(common_y_max)}', ((plot_x - 38), (cur_y + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, self.renderer.neutral_gray, 1, cv2.LINE_AA)
            cv2.putText(img, '0', ((plot_x - 18), ((cur_y + sub_h) - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, self.renderer.neutral_gray, 1, cv2.LINE_AA)
            c_vals = correct_data.get(skill, ([0] * n_sessions))
            inc_vals = incorrect_data.get(skill, ([0] * n_sessions))
            for i in range(n_sessions):
                bx = ((plot_x + int(((i + 0.5) * (plot_w / n_sessions)))) - (bar_w // 2))
                base_y = (cur_y + sub_h)
                c_h = int(((c_vals[i] / common_y_max) * (sub_h - 6)))
                inc_h = int(((inc_vals[i] / common_y_max) * (sub_h - 6)))
                if (c_h > 0):
                    cv2.rectangle(img, (bx, (base_y - c_h)), ((bx + bar_w), base_y), color_correct, (- 1))
                if (inc_h > 0):
                    cv2.rectangle(img, (bx, ((base_y - c_h) - inc_h)), ((bx + bar_w), (base_y - c_h)), color_incorrect, (- 1))
            if (s_idx == (n_skills - 1)):
                tick_sessions = ([1] + [s for s in range(10, (n_sessions + 1), 10)])
                for s_num in tick_sessions:
                    bx = (plot_x + int(((s_num - 0.5) * (plot_w / n_sessions))))
                    cv2.line(img, (bx, (cur_y + sub_h)), (bx, ((cur_y + sub_h) + 5)), self.renderer.BORDER_COLOR, 1, cv2.LINE_AA)
                    cv2.putText(img, str(s_num), ((bx - 8), ((cur_y + sub_h) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        return img

    def render_stacked_expected_outcome(self, n_sessions: int, all_skills: list[str], tp_data: dict[(str, list[float])], tn_data: dict[(str, list[float])], title: str, width: int=1000) -> np.ndarray:
        n_skills = len(all_skills)
        row_height = 80
        total_h = max(620, (130 + (n_skills * row_height)))
        img = self.renderer.create_base_canvas(title.upper(), width=width, height=total_h)
        if (n_skills == 0):
            return img
        (plot_x, start_y) = (150, 100)
        plot_w = 780
        bar_w = max(int(((plot_w / n_sessions) * 0.7)), 2)
        color_tp = self.renderer.status_green
        color_tn = self.renderer.status_red
        leg_y = 45
        cv2.rectangle(img, (700, (leg_y - 10)), (715, (leg_y + 2)), color_tp, (- 1))
        cv2.putText(img, 'True Positive', (722, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.rectangle(img, (820, (leg_y - 10)), (835, (leg_y + 2)), color_tn, (- 1))
        cv2.putText(img, 'True Negative', (842, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        for (s_idx, skill) in enumerate(all_skills):
            cur_y = (start_y + (s_idx * row_height))
            sub_h = (row_height - 20)
            cv2.rectangle(img, (plot_x, cur_y), ((plot_x + plot_w), (cur_y + sub_h)), self.renderer.PANEL_BORDER, 1)
            cv2.putText(img, skill[:16], (35, ((cur_y + (sub_h // 2)) + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
            tp_vals = tp_data[skill]
            tn_vals = tn_data[skill]
            for i in range(n_sessions):
                bx = ((plot_x + int(((i + 0.5) * (plot_w / n_sessions)))) - (bar_w // 2))
                base_y = (cur_y + sub_h)
                tp_h = int((tp_vals[i] * (sub_h - 6)))
                tn_h = int((tn_vals[i] * (sub_h - 6)))
                if (tp_h > 0):
                    cv2.rectangle(img, (bx, (base_y - tp_h)), ((bx + bar_w), base_y), color_tp, (- 1))
                if (tn_h > 0):
                    cv2.rectangle(img, (bx, ((base_y - tp_h) - tn_h)), ((bx + bar_w), (base_y - tp_h)), color_tn, (- 1))
            if (s_idx == (n_skills - 1)):
                tick_sessions = ([1] + [s for s in range(10, (n_sessions + 1), 10)])
                for s_num in tick_sessions:
                    bx = (plot_x + int(((s_num - 0.5) * (plot_w / n_sessions))))
                    cv2.line(img, (bx, (cur_y + sub_h)), (bx, ((cur_y + sub_h) + 5)), self.renderer.BORDER_COLOR, 1, cv2.LINE_AA)
                    cv2.putText(img, str(s_num), ((bx - 8), ((cur_y + sub_h) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        return img

    def _resolve_file_path(self) -> (Path | None):
        if ((not self.session_path) or (not os.path.exists(self.session_path))):
            return None
        target = (Path(self.session_path) / 'execution_statistics.txt')
        return (target if target.exists() else None)

    def run(self):
        file_path = self._resolve_file_path()
        if (not file_path):
            return
        try:
            current_dir = Path(self.session_path)
            logs_root = Path(self.logs_root)
            chain_dirs = self.collect_chain_folders(current_dir, logs_root)
            n_sessions = len(chain_dirs)
            all_skills: list[str] = []
            session_stats: list[dict] = []
            for s_dir in chain_dirs:
                stats = self.parse_execution_statistics((s_dir / 'execution_statistics.txt'))
                session_stats.append(stats)
                for skill in (list(stats['collected_data'].keys()) + list(stats['executions'].keys())):
                    if (skill not in all_skills):
                        all_skills.append(skill)
            data_nodes_by_skill = {skill: [] for skill in all_skills}
            tp_by_skill = {skill: [] for skill in all_skills}
            tn_by_skill = {skill: [] for skill in all_skills}
            correct_by_skill = {skill: [] for skill in all_skills}
            incorrect_by_skill = {skill: [] for skill in all_skills}
            for stats in session_stats:
                for skill in all_skills:
                    data_nodes_by_skill[skill].append(stats['collected_data'].get(skill, 0))
                    tp_by_skill[skill].append(stats['tp'].get(skill, 0.0))
                    tn_by_skill[skill].append(stats['tn'].get(skill, 0.0))
                    correct_by_skill[skill].append(stats['correct_counts'].get(skill, 0))
                    incorrect_by_skill[skill].append(stats['incorrect_counts'].get(skill, 0))
            out_dir = (self.get_pictures_dir() or '.')
            img_nodes = self.render_trend_plot(n_sessions, data_nodes_by_skill, 'Collected Data Nodes Trend', 'Nodes Count')
            out_path_nodes = os.path.join(out_dir, 'skills_collected_data_trend.png')
            cv2.imwrite(out_path_nodes, img_nodes)
            img_exec_outcomes = self.render_stacked_outcomes_count(n_sessions, all_skills, correct_by_skill, incorrect_by_skill, title='Skill Executions Count')
            out_path_exec = os.path.join(out_dir, 'skills_executions_trend.png')
            cv2.imwrite(out_path_exec, img_exec_outcomes)
            img_stacked = self.render_stacked_expected_outcome(n_sessions, all_skills, tp_by_skill, tn_by_skill, 'Expected Outcome Executions')
            out_path_stacked = os.path.join(out_dir, 'skills_expected_outcome_executions.png')
            cv2.imwrite(out_path_stacked, img_stacked)
            print(f'[ExecStatsGenerator] Saved execution stats plots to {out_dir}')
        except Exception as e:
            print(f'[ExecStatsGenerator] Failed to generate execution statistics plots: {e}')
class LiveMonitor(Renderer):

    def __init__(self, bus: Communicator, window_name: str='window_live', timeout: float=0.1, logs_root='logs'):
        super().__init__()
        self.bus = bus
        self.timeout = timeout
        self.window_name = window_name
        self._window_created = False
        self.session_path = get_session(logs_root)
        self._last_mtime = None

    def _ensure_window(self):
        if (not self._window_created):
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self._window_created = True

    def _safe_call(self, topic: str):
        try:
            try:
                return self.bus.call_service(topic, timeout=self.timeout)
            except TypeError:
                return self.bus.call_service(topic)
        except Exception:
            return None

    def setup(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
def get_session(logs_root, session_dir=None):
    if (not os.path.exists(logs_root)):
        return None
    if ((session_dir is not None) and os.path.exists(os.path.join(logs_root, session_dir))):
        return os.path.join(logs_root, session_dir)

    def parse_session_timestamp(dir_name):
        try:
            return datetime.strptime(dir_name, '%Y-%m-%d_%H-%M')
        except ValueError:
            return None
    valid_sessions = []
    for entry in os.scandir(logs_root):
        if (entry.is_dir() and parse_session_timestamp(entry.name)):
            valid_sessions.append(entry.path)
    return max(valid_sessions, key=os.path.basename, default=None)
class PipelineLiveMonitor(LiveMonitor):

    def setup(self):
        self.monitors = [SymbolsLive(self.bus, 'Current Symbols')]
        for mon in self.monitors:
            mon.setup()

    def update(self):
        for mon in self.monitors:
            mon.update()

    def close(self):
        for mon in self.monitors:
            mon.close()
class PipelinePictureGenerator(PictureGenerator):

    def __init__(self, logs_root='logs', session_dir=None):
        super().__init__(logs_root=logs_root, session_dir=session_dir)
        self.generators = [SymbolIconPictureGenerator(logs_root, session_dir=session_dir), SymbolCategoryPictureGenerator(logs_root, session_dir=session_dir), SymbolStatsGenerator(logs_root, session_dir=session_dir), UpdateFrequencyPictureGenerator(logs_root, session_dir=session_dir, bin_size_sec=10.0), HistogramPlotGenerator(logs_root, session_dir=session_dir), ExecStatsGenerator(logs_root, session_dir=session_dir)]

    def run(self):
        for gen in self.generators:
            gen.run()
SKILL_ORDER = ['Start', 'Find_The_Door', 'Go_To_The_Door', 'Open_The_Door', 'Find_The_Goal', 'Go_To_The_Goal', 'Finish']
SKILL_ORDER = [skill.lower() for skill in SKILL_ORDER]
def lower_median(arr, axis=None):
    a = np.asarray(arr)
    if (axis is None):
        flat = a.ravel()
        n = flat.size
        if (n == 0):
            return np.nan
        k = ((n - 1) // 2)
        return np.partition(flat, k)[k]
    else:
        n = a.shape[axis]
        if (n == 0):
            return np.nan
        k = ((n - 1) // 2)
        return np.take(np.sort(a, axis=axis), k, axis=axis)
class LiveSensimotorMonitor(LiveMonitor):

    def setup(self):
        pass

    def update(self):
        self._ensure_window()
        cv2.waitKey(1)
        lidar_readings = []
        for direction in self.LIDAR_ANGLES:
            val = self._safe_call(f'/lidar/{direction}/ask/value')
            if (val is None):
                return
            lidar_readings.append(val)
        raw_coded = self._safe_call('/camera/ask/coded')
        if (raw_coded is None):
            return
        cam_array = np.asarray(raw_coded, dtype=float).flatten()
        if (len(cam_array) < ((6 * 8) * 5)):
            return
        speeds = self._safe_call('/wheels/ask/speed')
        max_vel = self._safe_call('/get/wheels/max_velocity')
        if ((speeds is None) or (max_vel is None)):
            return
        img = self.create_base_canvas('LIVE SENSIMOTOR MONITOR')
        self.draw_lidar(img, lidar_readings, mode='live')
        self.draw_wheels(img, speeds[0], speeds[1], float(max_vel))
        self.draw_grids(img, cam_array, is_live=True)
        cv2.imshow(self.window_name, img)

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False
def _find_pddl_gng_dir(session_path: str):
    if ((not session_path) or (not os.path.isdir(session_path))):
        return None
    for pddl_cand in ['pddl', 'PDDL']:
        pddl_path = os.path.join(session_path, pddl_cand)
        if os.path.isdir(pddl_path):
            for gng_cand in ['gng', 'GNG']:
                gng_path = os.path.join(pddl_path, gng_cand)
                if os.path.isdir(gng_path):
                    return gng_path
    return os.path.join(session_path, 'PDDL', 'GNG')
def _load_symbol_data(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    gng_dict = data.get('gng', {})
    raw_nodes = gng_dict.get('nodes', [])
    if (not raw_nodes):
        return (None, None, None)
    nodes_arr = np.array(raw_nodes, dtype=np.float32)
    active_dims = data.get('active_dimensions')
    if (active_dims is None):
        active_dims = data.get('active_dimensions', list(range(nodes_arr.shape[1])))
    active_set = set(active_dims)
    n_nodes = nodes_arr.shape[0]
    full_nodes = np.zeros((n_nodes, 248), dtype=np.float32)
    for (col_idx, feat_idx) in enumerate(active_dims):
        if ((col_idx < nodes_arr.shape[1]) and (feat_idx < 248)):
            full_nodes[(:, feat_idx)] = nodes_arr[(:, col_idx)]
    return (full_nodes, active_set, nodes_arr)
def _build_masks(active_set):
    lidar_active = [(i in active_set) for i in range(8)]
    active_ganglion = []
    active_cones = []
    for c in range(48):
        offset = (8 + (c * 5))
        ganglion_active = ((offset in active_set) or ((offset + 1) in active_set))
        active_ganglion.append(ganglion_active)
        cone_active = all((((offset + ch) in active_set) for ch in (2, 3, 4)))
        active_cones.append(cone_active)
    return (lidar_active, active_ganglion, active_cones)
class SymbolStatsGenerator(PictureGenerator):
    LINE_COLOR_PRIMARY = (195, 105, 30)
    LINE_COLOR_SECONDARY = (71, 90, 224)
    SHADOW_COLOR = (245, 155, 80)
    TREND_COLOR_PRIMARY = (130, 60, 10)
    TREND_COLOR_SECONDARY = (40, 50, 180)

    def __init__(self, logs_root: str='logs', session_dir: (str | None)=None):
        super().__init__(logs_root=logs_root, session_dir=session_dir)

    @staticmethod
    def parse_continuation(stats_file: Path) -> (Path | None):
        if (not stats_file.exists()):
            return None
        pattern = re.compile('CONTINUATION FROM SESSION:\\s*(.+)', re.IGNORECASE)
        with stats_file.open('r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line.strip())
                if match:
                    return Path(match.group(1).strip())
        return None

    def collect_chain_folders(self, current_dir: Path, logs_root: Path) -> list[Path]:
        visited_dirs = {current_dir.resolve()}
        chain_dirs: list[Path] = [current_dir]
        curr = current_dir
        while True:
            stats_file = (curr / 'execution_statistics.txt')
            continued_rel = self.parse_continuation(stats_file)
            if (not continued_rel):
                break
            if (logs_root.parent / continued_rel).exists():
                next_dir = (logs_root.parent / continued_rel)
            elif (logs_root / continued_rel.name).exists():
                next_dir = (logs_root / continued_rel.name)
            else:
                break
            next_resolved = next_dir.resolve()
            if (next_resolved in visited_dirs):
                break
            visited_dirs.add(next_resolved)
            chain_dirs.append(next_dir)
            curr = next_dir
        chain_dirs.reverse()
        return chain_dirs

    @staticmethod
    def _extract_dim_val(val_str: str) -> (float | None):
        clean = val_str.strip().strip('"').strip("'")
        if (clean.upper() in {'N/A', 'NA', 'NONE', 'NULL', ''}):
            return None
        match = re.match('^([\\d.]+)', clean)
        return (float(match.group(1)) if match else None)

    @staticmethod
    def _extract_float_safely(val_str: str) -> (float | None):
        clean = str(val_str).strip().strip('"').strip("'")
        if (clean.upper() in {'N/A', 'NA', 'NONE', 'NULL', '', 'NAN', 'INF', '-INF'}):
            return None
        try:
            val = float(clean)
            return (None if (math.isnan(val) or math.isinf(val)) else val)
        except ValueError:
            return None

    def parse_symbol_stats_file(self, file_path: Path) -> list[dict]:
        if (not file_path.exists()):
            return []
        records = []
        with file_path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = self._extract_float_safely(row.get('timestamp', ''))
                tot = self._extract_float_safely(row.get('total_symbols', ''))
                if ((ts is None) or (tot is None)):
                    continue
                records.append({'timestamp': ts, 'total_symbols': tot, 'total_skills': self._extract_float_safely(row.get('total_skills', 0)), 'min_dims': self._extract_dim_val(row.get('min_dims', 0)), 'dominant_dims': self._extract_dim_val(row.get('dominant_dims', 0)), 'max_dims': self._extract_dim_val(row.get('max_dims', '')), 'avg_pre_symbols': self._extract_float_safely(row.get('avg_pre_symbols', 0)), 'avg_eff_symbols': self._extract_float_safely(row.get('avg_eff_symbols', 0)), 'reusability_ratio': self._extract_float_safely(row.get('reusability_ratio', '')), 'raw_row': row})
        return records

    def cumulate_stats(self, chain_dirs: list[Path], target_dir: Path) -> list[dict]:
        out_path = (target_dir / 'cumulated_symbol_stats.txt')
        if out_path.exists():
            print(f'[SymbolStatsGenerator] cumulated_symbol_stats.txt already exists at {out_path}, skipping.')
            return self.parse_symbol_stats_file(out_path)
        cumulated: list[dict] = []
        fieldnames = ['timestamp', 'total_symbols', 'total_skills', 'min_dims', 'dominant_dims', 'max_dims', 'avg_pre_symbols', 'avg_eff_symbols', 'reusability_ratio']
        for s_dir in reversed(chain_dirs):
            if (s_dir.resolve() != target_dir.resolve()):
                prev_cumulative = (s_dir / 'cumulated_symbol_stats.txt')
                if prev_cumulative.exists():
                    print(f'[SymbolStatsGenerator] Found preceding cumulative file at {prev_cumulative}. Merging.')
                    base_records = self.parse_symbol_stats_file(prev_cumulative)
                    cumulated.extend(base_records)
                    break
            for fname in ('symbol_stats_log.txt', 'symbol_stats.txt'):
                fpath = (s_dir / fname)
                if fpath.exists():
                    records = self.parse_symbol_stats_file(fpath)
                    cumulated.extend(records)
                    break
        cumulated.sort(key=(lambda x: x['timestamp']))
        with out_path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in cumulated:
                writer.writerow({k: r['raw_row'].get(k, r.get(k, '')) for k in fieldnames})
        return cumulated

    @staticmethod
    def _collapse_time_gaps(timestamps: list[float], gap_threshold_factor: float=5.0) -> list[float]:
        if (len(timestamps) <= 2):
            return timestamps[:]
        deltas = [(timestamps[i] - timestamps[(i - 1)]) for i in range(1, len(timestamps))]
        positive_deltas = [d for d in deltas if (d > 0)]
        if (not positive_deltas):
            return timestamps[:]
        mean_delta = float(np.mean(positive_deltas))
        gap_limit = max((mean_delta * gap_threshold_factor), 600.0)
        compressed_times = [0.0]
        for d in deltas:
            effective_dt = (mean_delta if (d > gap_limit) else max(d, 0.0))
            compressed_times.append((compressed_times[(- 1)] + effective_dt))
        return compressed_times

    @staticmethod
    def _calculate_moving_average(y_vals: list[float], window_size: int=100) -> list[float]:
        if (not y_vals):
            return []
        ma = []
        for i in range(len(y_vals)):
            start_idx = max(0, ((i - window_size) + 1))
            window = y_vals[start_idx:(i + 1)]
            ma.append(float(np.mean(window)))
        return ma

    def _draw_trend_line(self, img: np.ndarray, x_vals: list[float], y_vals: list[float], get_x: callable, get_y: callable, color: tuple[(int, int, int)], thickness: int=2, window_size: int=1000):
        if ((len(x_vals) < 2) or (len(x_vals) != len(y_vals))):
            return
        ma_vals = self._calculate_moving_average(y_vals, window_size=window_size)
        pts = [(get_x(x), get_y(y)) for (x, y) in zip(x_vals, ma_vals)]
        for i in range((len(pts) - 1)):
            cv2.line(img, pts[i], pts[(i + 1)], color, thickness, cv2.LINE_AA)

    def _setup_plot_area(self, title: str, x_vals: list[float], y_min: float, y_max: float, y_label: str, width: int=1000, height: int=620) -> tuple[(np.ndarray, int, int, int, int, callable, callable)]:
        img = self.renderer.create_base_canvas(title.upper(), width=width, height=height)
        (plot_x, plot_y) = (90, 110)
        (plot_w, plot_h) = (710, 430)
        cv2.rectangle(img, (plot_x, plot_y), ((plot_x + plot_w), (plot_y + plot_h)), self.renderer.PANEL_BORDER, 1)
        t_min = (min(x_vals) if x_vals else 0.0)
        t_max = (max(x_vals) if x_vals else 1.0)
        t_span = max((t_max - t_min), 1e-06)
        y_range = max((y_max - y_min), 1e-06)
        margin = (y_range * 0.08)
        plot_y_min = (y_min - margin)
        plot_y_max = (y_max + margin)
        plot_y_span = (plot_y_max - plot_y_min)
        grid_steps = 4
        for i in range((grid_steps + 1)):
            y_pixel = ((plot_y + plot_h) - int((i * (plot_h / grid_steps))))
            val = (plot_y_min + (i * (plot_y_span / grid_steps)))
            cv2.line(img, (plot_x, y_pixel), ((plot_x + plot_w), y_pixel), self.renderer.tertiary_color, 1, cv2.LINE_AA)
            cv2.putText(img, (f'{val:.2f}' if (plot_y_span < 10) else f'{int(round(val))}'), ((plot_x - 55), (y_pixel + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.neutral_gray, 1, cv2.LINE_AA)
        x_steps = 5
        for i in range((x_steps + 1)):
            x_pixel = (plot_x + int((i * (plot_w / x_steps))))
            t_val = (i * (t_span / x_steps))
            cv2.line(img, (x_pixel, (plot_y + plot_h)), (x_pixel, ((plot_y + plot_h) + 5)), self.renderer.BORDER_COLOR, 1, cv2.LINE_AA)
            cv2.putText(img, f'+{int(t_val)}s', ((x_pixel - 15), ((plot_y + plot_h) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.putText(img, 'Time Elapsed', (((plot_x + (plot_w // 2)) - 40), ((plot_y + plot_h) + 45)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.putText(img, y_label, ((plot_x - 30), (plot_y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

        def get_x_px(val: float) -> int:
            return (plot_x + int((((val - t_min) / t_span) * plot_w)))

        def get_y_px(val: float) -> int:
            norm = ((val - plot_y_min) / plot_y_span)
            return ((plot_y + plot_h) - int((np.clip(norm, 0.0, 1.0) * plot_h)))
        return (img, plot_x, plot_y, plot_w, plot_h, get_x_px, get_y_px)

    def render_total_symbols_plot(self, timestamps: list[float], total_symbols: list[float]) -> np.ndarray:
        y_min = (min(total_symbols) if total_symbols else 0.0)
        y_max = (max(total_symbols) if total_symbols else 1.0)
        (img, px, py, pw, ph, get_x, get_y) = self._setup_plot_area('Total Number of Symbols', timestamps, y_min, y_max, 'Symbols (count)')
        self._draw_trend_line(img, timestamps, total_symbols, get_x, get_y, self.TREND_COLOR_PRIMARY, thickness=2)
        for (t, v) in zip(timestamps, total_symbols):
            cv2.circle(img, (get_x(t), get_y(v)), 1, self.LINE_COLOR_PRIMARY, (- 1), cv2.LINE_AA)
        (leg_x, leg_y) = (((px + pw) + 15), (py + 25))
        cv2.circle(img, ((leg_x + 6), (leg_y - 2)), 4, self.LINE_COLOR_PRIMARY, (- 1), cv2.LINE_AA)
        cv2.putText(img, 'Symbols Count (points)', ((leg_x + 18), (leg_y + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.line(img, (leg_x, (leg_y + 20)), ((leg_x + 14), (leg_y + 20)), self.TREND_COLOR_PRIMARY, 2, cv2.LINE_AA)
        cv2.putText(img, 'Moving Average', ((leg_x + 18), (leg_y + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        self.renderer.draw_sample_badge(img, count=len(timestamps), label='Data Points', position=(px, 595))
        return img

    def render_dimensionality_plot(self, timestamps: list[float], min_dims: list[float], dominant_dims: list[float], max_dims: list[float]) -> np.ndarray:
        all_vals = ((min_dims + dominant_dims) + max_dims)
        y_min = (min(all_vals) if all_vals else 0.0)
        y_max = (max(all_vals) if all_vals else 1.0)
        (img, px, py, pw, ph, get_x, get_y) = self._setup_plot_area('Dimensionality of Symbols', timestamps, y_min, y_max, 'Dimensions')
        upper_pts = [(get_x(t), get_y(v)) for (t, v) in zip(timestamps, max_dims)]
        lower_pts = [(get_x(t), get_y(v)) for (t, v) in zip(timestamps, min_dims)]
        poly_pts = np.array((upper_pts + lower_pts[::(- 1)]), dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly_pts], self.SHADOW_COLOR)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        for i in range((len(upper_pts) - 1)):
            cv2.line(img, upper_pts[i], upper_pts[(i + 1)], self.renderer.PANEL_BORDER, 1, cv2.LINE_AA)
            cv2.line(img, lower_pts[i], lower_pts[(i + 1)], self.renderer.PANEL_BORDER, 1, cv2.LINE_AA)
        self._draw_trend_line(img, timestamps, dominant_dims, get_x, get_y, self.TREND_COLOR_PRIMARY, thickness=2)
        for (t, v) in zip(timestamps, dominant_dims):
            cv2.circle(img, (get_x(t), get_y(v)), 1, self.LINE_COLOR_PRIMARY, (- 1), cv2.LINE_AA)
        (leg_x, leg_y) = (((px + pw) + 15), (py + 25))
        cv2.circle(img, ((leg_x + 6), (leg_y - 2)), 4, self.LINE_COLOR_PRIMARY, (- 1), cv2.LINE_AA)
        cv2.putText(img, 'Dominant', ((leg_x + 18), (leg_y + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.line(img, (leg_x, (leg_y + 20)), ((leg_x + 14), (leg_y + 20)), self.TREND_COLOR_PRIMARY, 2, cv2.LINE_AA)
        cv2.putText(img, 'Moving Average (dominant)', ((leg_x + 18), (leg_y + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.rectangle(img, (leg_x, (leg_y + 36)), ((leg_x + 12), (leg_y + 48)), self.SHADOW_COLOR, (- 1))
        cv2.putText(img, 'Min - Max', ((leg_x + 18), (leg_y + 46)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        self.renderer.draw_sample_badge(img, count=len(timestamps), label='Data Points', position=(px, 595))
        return img

    def render_avg_symbols_plot(self, pre_timestamps: list[float], avg_pre: list[float], eff_timestamps: list[float], avg_eff: list[float]) -> np.ndarray:
        all_vals = (avg_pre + avg_eff)
        y_min = (min(all_vals) if all_vals else 0.0)
        y_max = (max(all_vals) if all_vals else 1.0)
        all_ts = (pre_timestamps + eff_timestamps)
        (img, px, py, pw, ph, get_x, get_y) = self._setup_plot_area('Average Number of Symbols in Predicate', all_ts, y_min, y_max, 'Symbol Count')
        self._draw_trend_line(img, pre_timestamps, avg_pre, get_x, get_y, self.TREND_COLOR_PRIMARY, thickness=2)
        self._draw_trend_line(img, eff_timestamps, avg_eff, get_x, get_y, self.TREND_COLOR_SECONDARY, thickness=2)
        for (t, v) in zip(pre_timestamps, avg_pre):
            cv2.circle(img, (get_x(t), get_y(v)), 1, self.LINE_COLOR_PRIMARY, (- 1), cv2.LINE_AA)
        for (t, v) in zip(eff_timestamps, avg_eff):
            cv2.circle(img, (get_x(t), get_y(v)), 1, self.LINE_COLOR_SECONDARY, (- 1), cv2.LINE_AA)
        (leg_x, leg_y) = (((px + pw) + 15), (py + 25))
        cv2.circle(img, ((leg_x + 6), (leg_y - 2)), 4, self.LINE_COLOR_PRIMARY, (- 1), cv2.LINE_AA)
        cv2.putText(img, 'Precondition (points)', ((leg_x + 18), (leg_y + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.line(img, (leg_x, (leg_y + 18)), ((leg_x + 14), (leg_y + 18)), self.TREND_COLOR_PRIMARY, 2, cv2.LINE_AA)
        cv2.putText(img, 'Precondition (moving average)', ((leg_x + 18), (leg_y + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.circle(img, ((leg_x + 6), (leg_y + 38)), 4, self.LINE_COLOR_SECONDARY, (- 1), cv2.LINE_AA)
        cv2.putText(img, 'Effect (points)', ((leg_x + 18), (leg_y + 42)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.line(img, (leg_x, (leg_y + 58)), ((leg_x + 14), (leg_y + 58)), self.TREND_COLOR_SECONDARY, 2, cv2.LINE_AA)
        cv2.putText(img, 'Effect (moving average)', ((leg_x + 18), (leg_y + 62)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        self.renderer.draw_sample_badge(img, count=max(len(pre_timestamps), len(eff_timestamps)), label='Data Points', position=(px, 595))
        return img

    def render_reusability_plot(self, timestamps: list[float], reusability_ratios: list[float]) -> np.ndarray:
        y_min = (min(reusability_ratios) if reusability_ratios else 0.0)
        y_max = (max(reusability_ratios) if reusability_ratios else 1.0)
        (img, px, py, pw, ph, get_x, get_y) = self._setup_plot_area('Reusability Ratio', timestamps, y_min, y_max, 'Ratio')
        self._draw_trend_line(img, timestamps, reusability_ratios, get_x, get_y, self.TREND_COLOR_PRIMARY, thickness=2)
        for (t, v) in zip(timestamps, reusability_ratios):
            cv2.circle(img, (get_x(t), get_y(v)), 1, self.LINE_COLOR_PRIMARY, (- 1), cv2.LINE_AA)
        (leg_x, leg_y) = (((px + pw) + 15), (py + 25))
        cv2.circle(img, ((leg_x + 6), (leg_y - 2)), 4, self.LINE_COLOR_PRIMARY, (- 1), cv2.LINE_AA)
        cv2.putText(img, 'Ratio (points)', ((leg_x + 18), (leg_y + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.line(img, (leg_x, (leg_y + 20)), ((leg_x + 14), (leg_y + 20)), self.TREND_COLOR_PRIMARY, 2, cv2.LINE_AA)
        cv2.putText(img, 'Moving Average', ((leg_x + 18), (leg_y + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        self.renderer.draw_sample_badge(img, count=len(timestamps), label='Valid Samples', position=(px, 595))
        return img

    def run(self):
        if ((not self.session_path) or (not os.path.exists(self.session_path))):
            return
        try:
            current_dir = Path(self.session_path)
            logs_root = Path(self.logs_root)
            chain_dirs = self.collect_chain_folders(current_dir, logs_root)
            cumulated = self.cumulate_stats(chain_dirs, current_dir)
            if (not cumulated):
                print('[SymbolStatsGenerator] No symbol statistics found across session dirs.')
                return
            raw_ts = [r['timestamp'] for r in cumulated]
            plot_ts = self._collapse_time_gaps(raw_ts)
            for (r, pts) in zip(cumulated, plot_ts):
                r['plot_timestamp'] = pts
            tot_all = [r['total_symbols'] for r in cumulated]
            out_dir = (self.get_pictures_dir() or str(current_dir))
            img_symbols = self.render_total_symbols_plot(plot_ts, tot_all)
            cv2.imwrite(os.path.join(out_dir, 'symbols_total_trend.png'), img_symbols)
            dim_valid = [r for r in cumulated if ((r['dominant_dims'] is not None) and (r['min_dims'] is not None) and (r['max_dims'] is not None))]
            if dim_valid:
                img_dims = self.render_dimensionality_plot([r['plot_timestamp'] for r in dim_valid], [r['min_dims'] for r in dim_valid], [r['dominant_dims'] for r in dim_valid], [r['max_dims'] for r in dim_valid])
                cv2.imwrite(os.path.join(out_dir, 'symbols_dimensionality_trend.png'), img_dims)
            pre_valid = [r for r in cumulated if (r['avg_pre_symbols'] is not None)]
            eff_valid = [r for r in cumulated if (r['avg_eff_symbols'] is not None)]
            if (pre_valid or eff_valid):
                img_avg = self.render_avg_symbols_plot([r['plot_timestamp'] for r in pre_valid], [r['avg_pre_symbols'] for r in pre_valid], [r['plot_timestamp'] for r in eff_valid], [r['avg_eff_symbols'] for r in eff_valid])
                cv2.imwrite(os.path.join(out_dir, 'symbols_pre_eff_avg_trend.png'), img_avg)
            ratio_valid = [r for r in cumulated if (r['reusability_ratio'] is not None)]
            if ratio_valid:
                img_reusability = self.render_reusability_plot([r['plot_timestamp'] for r in ratio_valid], [r['reusability_ratio'] for r in ratio_valid])
                cv2.imwrite(os.path.join(out_dir, 'symbols_reusability_trend.png'), img_reusability)
            print(f'[SymbolStatsGenerator] Saved cumulated symbol statistics and plots to {out_dir}')
        except Exception as e:
            print(f'[SymbolStatsGenerator] Failed to generate symbol statistics plots: {e}')
class SymbolsLive(LiveMonitor):

    def setup(self):
        pass

    def update(self):
        pass

    def close(self):
        pass
class TransGraphLive(LiveMonitor):

    def setup(self):
        self.trans_filename = 'trans_graph_PDDL.json'

    def update(self):
        self._ensure_window()
        cv2.waitKey(1)
        if (not self.session_path):
            return
        trans_path = os.path.join(self.session_path, 'graphs', self.trans_filename)
        try:
            if (os.path.exists(trans_path) and (os.path.getsize(trans_path) > 0)):
                mtime = os.path.getmtime(trans_path)
                if (self._last_mtime is None):
                    self._last_mtime = mtime
                if (mtime > self._last_mtime):
                    with open(trans_path, 'r') as f:
                        graph_data = json.load(f)
                    img = self.create_base_canvas('LIVE TRANSITION GRAPH', 'based on PDDL domain')
                    self.draw_transition_graph(img, graph_data)
                    cv2.imshow(self.window_name, img)
                    self._last_mtime = mtime
        except Exception as e:
            print(f'Error loading live graph: {e}')

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False
class TransGraphPictureGenerator(PictureGenerator):

    def run(self):
        if (not self.session_path):
            return
        graphs_dir = os.path.join(self.session_path, 'graphs')
        json_path = os.path.join(graphs_dir, 'trans_graph_PDDL.json')
        if (not os.path.exists(json_path)):
            return
        try:
            with open(json_path, 'r') as f:
                graph_data = json.load(f)
            img = self.renderer.create_base_canvas('TRANSITION GRAPH', f'based on PDDL domain')
            self.renderer.draw_transition_graph(img, graph_data)
            out_dir = self.get_pictures_dir()
            if (not out_dir):
                return
            out_path = os.path.join(out_dir, 'trans_graph_PDDL.png')
            cv2.imwrite(out_path, img)
            print(f'[PictureGenerator] Saved Transition Graph image to {out_path}')
        except Exception as e:
            print(f'[PictureGenerator] Failed to generate TransGraph: {e}')
@dataclass
class SkillFinalStats():
    nodes: int
    ratio: (float | None)
    pos_ratio: (float | None)
    neg_ratio: (float | None)
class HistogramPlotGenerator(PictureGenerator):

    def __init__(self, logs_root: str='logs', session_dir: (str | None)=None):
        super().__init__(logs_root=logs_root, session_dir=session_dir)

    @staticmethod
    def parse_continuation(stats_file: Path) -> (Path | None):
        if (not stats_file.exists()):
            return None
        pattern = re.compile('CONTINUATION FROM SESSION:\\s*(.+)', re.IGNORECASE)
        with stats_file.open('r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line.strip())
                if match:
                    return Path(match.group(1).strip())
        return None

    @staticmethod
    def parse_timestamp_line(line: str) -> (float | None):
        line = line.strip()
        if (not line):
            return None
        try:
            return float(line)
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(line).timestamp()
        except ValueError:
            return None

    def collect_chain_folders(self, current_dir: Path, logs_root: Path) -> list[Path]:
        visited_dirs = {current_dir.resolve()}
        chain_dirs: list[Path] = [current_dir]
        curr = current_dir
        while True:
            stats_file = (curr / 'execution_statistics.txt')
            continued_rel = self.parse_continuation(stats_file)
            if (not continued_rel):
                break
            if (logs_root.parent / continued_rel).exists():
                next_dir = (logs_root.parent / continued_rel)
            elif (logs_root / continued_rel.name).exists():
                next_dir = (logs_root / continued_rel.name)
            else:
                break
            next_resolved = next_dir.resolve()
            if (next_resolved in visited_dirs):
                break
            visited_dirs.add(next_resolved)
            chain_dirs.append(next_dir)
            curr = next_dir
        chain_dirs.reverse()
        return chain_dirs

    def load_session_timestamps(self, session_dir: Path) -> np.ndarray:
        ts_file = (session_dir / 'timestamps.txt')
        if (not ts_file.exists()):
            return np.array([], dtype=float)
        timestamps: list[float] = []
        with ts_file.open('r', encoding='utf-8') as f:
            for line in f:
                val = self.parse_timestamp_line(line)
                if (val is not None):
                    timestamps.append(val)
        return (np.sort(np.array(timestamps, dtype=float)) if timestamps else np.array([], dtype=float))

    def build_and_glue_normalized_timeline(self, chain_dirs: list[Path], intra_session_max_gap_sec: float=3600.0, stitch_gap_sec: float=60.0) -> tuple[(np.ndarray, list[dict])]:
        raw_segments: list[dict] = []
        for (s_idx, s_dir) in enumerate(chain_dirs):
            ts = self.load_session_timestamps(s_dir)
            if (len(ts) < 2):
                continue
            session_num = (s_idx + 1)
            dt = np.diff(ts, prepend=ts[0])
            gap_indices = np.where((dt > intra_session_max_gap_sec))[0]
            split_points = (([0] + list(gap_indices)) + [len(ts)])
            num_splits = (len(split_points) - 1)
            for sub_i in range(num_splits):
                sub_ts = ts[split_points[sub_i]:split_points[(sub_i + 1)]]
                if (len(sub_ts) >= 2):
                    session_tag = (f'S{session_num}' if (sub_i == 0) else None)
                    subsession_tag = (f'S{session_num}.{(sub_i + 1)}' if (num_splits > 1) else None)
                    raw_segments.append({'session_tag': session_tag, 'subsession_tag': subsession_tag, 'timestamps': sub_ts})
        if (not raw_segments):
            return (np.array([], dtype=float), [])
        glued_times: list[np.ndarray] = []
        segment_boundaries: list[dict] = []
        current_offset_sec = 0.0
        for seg in raw_segments:
            rel_ts = (seg['timestamps'] - seg['timestamps'][0])
            stitched_part = (current_offset_sec + rel_ts)
            glued_times.append(stitched_part)
            segment_boundaries.append({'session_tag': seg['session_tag'], 'subsession_tag': seg['subsession_tag'], 'start_min': (stitched_part[0] / 60.0), 'end_min': (stitched_part[(- 1)] / 60.0)})
            current_offset_sec = (stitched_part[(- 1)] + stitch_gap_sec)
        return ((np.concatenate(glued_times) / 60.0), segment_boundaries)

    def draw_kde_canvas(self, timeline_min: np.ndarray, width: int=1000, height: int=620) -> np.ndarray:
        img = self.renderer.create_base_canvas('BRAIN UPDATES FREQUENCY (KDE)', width=width, height=height)
        if (len(timeline_min) < 10):
            cv2.putText(img, 'Insufficient data points for KDE estimation.', (320, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.renderer.neutral_gray, 1, cv2.LINE_AA)
            return img
        bw = ('scott' if (np.std(timeline_min) > 1e-05) else 0.5)
        kde = gaussian_kde(timeline_min, bw_method=bw)
        (t_min, t_max) = (timeline_min[0], timeline_min[(- 1)])
        x_margin = max(((t_max - t_min) * 0.02), 1.0)
        (x_start, x_end) = (max(0.0, (t_min - x_margin)), (t_max + x_margin))
        (plot_x, plot_y) = (90, 110)
        (plot_w, plot_h) = (860, 420)
        cv2.rectangle(img, (plot_x, plot_y), ((plot_x + plot_w), (plot_y + plot_h)), self.renderer.PANEL_BORDER, 1)
        x_grid = np.linspace(x_start, x_end, 600)
        density = (kde(x_grid) * len(timeline_min))
        max_density = max((float(np.max(density)) * 1.15), 0.0001)
        grid_steps_y = 4
        for i in range((grid_steps_y + 1)):
            y_val = ((plot_y + plot_h) - int((i * (plot_h / grid_steps_y))))
            cv2.line(img, (plot_x, y_val), ((plot_x + plot_w), y_val), self.renderer.tertiary_color, 1, cv2.LINE_AA)
            val_label = f'{(i * (max_density / grid_steps_y)):.2f}'
            cv2.putText(img, val_label, ((plot_x - 55), (y_val + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.renderer.neutral_gray, 1, cv2.LINE_AA)
        time_span = (x_end - x_start)
        step = max(1.0, round((time_span / 6.0)))
        first_tick = (np.ceil((x_start / step)) * step)
        tick_vals = np.arange(first_tick, x_end, step)
        for t_val in tick_vals:
            tx = (plot_x + int((((t_val - x_start) / (x_end - x_start)) * plot_w)))
            cv2.line(img, (tx, plot_y), (tx, (plot_y + plot_h)), self.renderer.tertiary_color, 1, cv2.LINE_AA)
            cv2.line(img, (tx, (plot_y + plot_h)), (tx, ((plot_y + plot_h) + 5)), self.renderer.BORDER_COLOR, 1, cv2.LINE_AA)
            cv2.putText(img, f'{int(t_val)}m', ((tx - 12), ((plot_y + plot_h) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        pts = []
        for (x_val, dens) in zip(x_grid, density):
            px = (plot_x + int((((x_val - x_start) / (x_end - x_start)) * plot_w)))
            py = ((plot_y + plot_h) - int(((dens / max_density) * plot_h)))
            pts.append((px, py))
        fill_pts = np.array((([(plot_x, (plot_y + plot_h))] + pts) + [((plot_x + plot_w), (plot_y + plot_h))]), dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [fill_pts], self.renderer.secondary_color)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], isClosed=False, color=self.renderer.BORDER_COLOR, thickness=2, lineType=cv2.LINE_AA)
        cv2.rectangle(img, (plot_x, plot_y), ((plot_x + plot_w), (plot_y + plot_h)), self.renderer.PANEL_BORDER, 1)
        cv2.putText(img, 'Continuous Time [minutes]', (((plot_x + (plot_w // 2)) - 90), ((plot_y + plot_h) + 45)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.putText(img, 'Density (events/min)', ((plot_x - 30), (plot_y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        self.renderer.draw_sample_badge(img, count=len(timeline_min), label='Data Points', position=(plot_x, 595))
        return img

    def _resolve_file_path(self) -> (Path | None):
        if ((not self.session_path) or (not os.path.exists(self.session_path))):
            return None
        target = (Path(self.session_path) / 'timestamps.txt')
        return (target if target.exists() else None)

    def run(self):
        file_path = self._resolve_file_path()
        if (not file_path):
            return
        try:
            current_dir = Path(self.session_path)
            logs_root = Path(self.logs_root)
            chain_dirs = self.collect_chain_folders(current_dir, logs_root)
            (unified_timeline_min, _) = self.build_and_glue_normalized_timeline(chain_dirs)
            img = self.draw_kde_canvas(unified_timeline_min)
            out_dir = (self.get_pictures_dir() or '.')
            out_path = os.path.join(out_dir, 'brain_updates_frequency_kde.png')
            cv2.imwrite(out_path, img)
            print(f'[HistogramPlotGenerator] Saved KDE plot to {out_path}')
        except Exception as e:
            print(f'[HistogramPlotGenerator] Failed to generate KDE plot: {e}')
class UpdateFrequencyPictureGenerator(PictureGenerator):

    def __init__(self, logs_root='logs', session_dir=None, timestamp_file='timestamps.txt', bin_size_sec=60.0):
        super().__init__(logs_root, session_dir=session_dir)
        self.timestamp_file = timestamp_file
        self.bin_size_sec = bin_size_sec

    def _resolve_file_path(self):
        if self.session_path:
            sess_path = os.path.join(self.session_path, self.timestamp_file)
            if os.path.exists(sess_path):
                return sess_path
        if os.path.exists(self.timestamp_file):
            return self.timestamp_file
        return None

    def run(self):
        file_path = self._resolve_file_path()
        if (not file_path):
            return
        try:
            with open(file_path, 'r') as f:
                timestamps = [float(line.strip()) for line in f if line.strip()]
            img = self.renderer.create_base_canvas('UPDATE FREQUENCY OVER TIME')
            self.renderer.draw_frequency_plot(img, timestamps, bin_size_sec=self.bin_size_sec)
            self.renderer.draw_sample_badge(img, count=len(timestamps), position=(35, 595))
            out_dir = (self.get_pictures_dir() or '.')
            out_path = os.path.join(out_dir, 'update_frequency.png')
            cv2.imwrite(out_path, img)
            print(f'[PictureGenerator] Saved frequency plot to {out_path}')
        except Exception as e:
            print(f'[PictureGenerator] Failed to generate frequency plot: {e}')
class PictureGenerator():

    def __init__(self, logs_root='logs', session_dir=None):
        self.logs_root = logs_root
        self.session_path = get_session(logs_root, session_dir)
        self.renderer = Renderer()

    def get_pictures_dir(self):
        out_dir = os.path.join(self.session_path, 'pictures')
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def run(self):
        raise NotImplementedError
class Renderer():

    def __init__(self):
        self.primary_color = (195, 105, 30)
        self.secondary_color = (235, 185, 90)
        self.tertiary_color = (248, 235, 215)
        self.BG_COLOR = (248, 249, 250)
        self.BORDER_COLOR = (40, 44, 52)
        self.TEXT_MAIN = (30, 30, 30)
        self.ROBOT_BODY = (40, 44, 52)
        self.PANEL_BORDER = (210, 215, 220)
        self.neutral_gray = (130, 135, 140)
        self.status_green = (45, 160, 50)
        self.status_red = (45, 50, 205)
        self.LIDAR_ANGLES = [17, 50, 90, 150, 210, 270, 310, 343]

    def compute_hue_median_stats(self, r_vals, g_vals, b_vals):
        max_v = max(np.max(r_vals), np.max(g_vals), np.max(b_vals))
        if (max_v > 1.0):
            r = (np.array(r_vals, dtype=np.float32) / 255.0)
            g = (np.array(g_vals, dtype=np.float32) / 255.0)
            b = (np.array(b_vals, dtype=np.float32) / 255.0)
        else:
            r = np.array(r_vals, dtype=np.float32)
            g = np.array(g_vals, dtype=np.float32)
            b = np.array(b_vals, dtype=np.float32)
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = (cmax - cmin)
        v = cmax
        s = np.zeros_like(v)
        nonzero_v = (v > 1e-06)
        s[nonzero_v] = (delta[nonzero_v] / v[nonzero_v])
        h = np.zeros_like(r)
        nonzero_delta = (delta > 1e-06)
        mask = (nonzero_delta & (cmax == r))
        h[mask] = (60.0 * (((g[mask] - b[mask]) / delta[mask]) % 6))
        mask = (nonzero_delta & (cmax == g))
        h[mask] = (60.0 * (((b[mask] - r[mask]) / delta[mask]) + 2))
        mask = (nonzero_delta & (cmax == b))
        h[mask] = (60.0 * (((r[mask] - g[mask]) / delta[mask]) + 4))
        angles = np.deg2rad(h)
        sin_mean = np.mean(np.sin(angles))
        cos_mean = np.mean(np.cos(angles))
        R = np.hypot(sin_mean, cos_mean)
        circ_var = float((1.0 - R))
        mean_angle = np.arctan2(sin_mean, cos_mean)
        diff_angles = np.arctan2(np.sin((angles - mean_angle)), np.cos((angles - mean_angle)))
        median_diff = lower_median(diff_angles)
        median_angle = (((mean_angle + median_diff) + (2 * np.pi)) % (2 * np.pi))
        median_angle_deg = np.rad2deg(median_angle)
        median_s = float(lower_median(s))
        median_v = float(lower_median(v))
        h_cv = int(np.clip((median_angle_deg / 2.0), 0, 179))
        s_cv = int(np.clip((median_s * 255.0), 0, 255))
        v_cv = int(np.clip((median_v * 255.0), 0, 255))
        hsv_pixel = np.uint8([[[h_cv, s_cv, v_cv]]])
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        median_bgr = (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))
        return (median_bgr, circ_var)

    def draw_sample_badge(self, img, count: int, label: str='Nodes', position=(90, 595)):
        text = f'{label}: {count}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1
        ((text_w, text_h), baseline) = cv2.getTextSize(text, font, scale, thickness)
        (x, y) = position
        (pad_x, pad_y) = (10, 6)
        cv2.rectangle(img, (x, ((y - text_h) - pad_y)), (((x + text_w) + (2 * pad_x)), ((y + baseline) + pad_y)), self.tertiary_color, (- 1))
        cv2.rectangle(img, (x, ((y - text_h) - pad_y)), (((x + text_w) + (2 * pad_x)), ((y + baseline) + pad_y)), self.primary_color, 1)
        cv2.putText(img, text, ((x + pad_x), y), font, scale, self.TEXT_MAIN, thickness, cv2.LINE_AA)

    def create_base_canvas(self, title, subtitle=None, width=1000, height=620):
        img = np.ones((height, width, 3), dtype=np.uint8)
        img[:] = self.BG_COLOR
        cv2.rectangle(img, (0, 0), ((width - 1), (height - 1)), self.BORDER_COLOR, 3)
        cv2.putText(img, title, (35, 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, self.TEXT_MAIN, 2, cv2.LINE_AA)
        if subtitle:
            cv2.putText(img, subtitle, (35, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 70, 20), 1, cv2.LINE_AA)
        y_line = (85 if subtitle else 65)
        cv2.line(img, (35, y_line), ((width - 35), y_line), self.PANEL_BORDER, 1, cv2.LINE_AA)
        return img

    def draw_lidar(self, img, lidar_data, center=(230, 280), max_radius=150, mode='live', active_flags=None, custom_color=None):
        cv2.putText(img, 'LIDAR', (45, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)
        color_main = (custom_color if (custom_color is not None) else self.primary_color)
        color_sec = (self.secondary_color if (custom_color is None) else tuple((int(((0.6 * c) + (0.4 * 255))) for c in color_main)))
        for ang in self.LIDAR_ANGLES:
            rad = np.deg2rad((ang - 90))
            x_end = int((center[0] + (max_radius * np.cos(rad))))
            y_end = int((center[1] + (max_radius * np.sin(rad))))
            cv2.line(img, center, (x_end, y_end), self.tertiary_color, 2, cv2.LINE_AA)
        for (i, ang) in enumerate(self.LIDAR_ANGLES):
            if ((active_flags is not None) and (not active_flags[i])):
                continue
            rad = np.deg2rad((ang - 90))
            if (mode == 'live'):
                val = max(min(float(lidar_data[i]), 0.1), 0.0)
                r_val = ((val / 0.1) * max_radius)
                x_val = int((center[0] + (r_val * np.cos(rad))))
                y_val = int((center[1] + (r_val * np.sin(rad))))
                cv2.line(img, center, (x_val, y_val), color_main, 3, cv2.LINE_AA)
                cv2.circle(img, (x_val, y_val), 4, color_main, (- 1), cv2.LINE_AA)
            elif (mode == 'iconic'):
                val = float(lidar_data[i])
                norm_val = ((val / 0.1) if (val <= 0.1) else val)
                r_val = (max(min(norm_val, 1.0), 0.0) * max_radius)
                x_val = int((center[0] + (r_val * np.cos(rad))))
                y_val = int((center[1] + (r_val * np.sin(rad))))
                cv2.circle(img, (x_val, y_val), 5, color_main, (- 1), cv2.LINE_AA)
            elif (mode == 'categorical'):
                (val_min, val_max, val_median) = lidar_data[i]
                norm_min = ((val_min / 0.1) if (val_min <= 0.1) else val_min)
                norm_max = ((val_max / 0.1) if (val_max <= 0.1) else val_max)
                norm_median = ((val_median / 0.1) if (val_median <= 0.1) else val_median)
                r_min = (max(min(norm_min, 1.0), 0.0) * max_radius)
                r_max = (max(min(norm_max, 1.0), 0.0) * max_radius)
                r_med = (max(min(norm_median, 1.0), 0.0) * max_radius)
                x_min = int((center[0] + (r_min * np.cos(rad))))
                y_min = int((center[1] + (r_min * np.sin(rad))))
                x_max = int((center[0] + (r_max * np.cos(rad))))
                y_max = int((center[1] + (r_max * np.sin(rad))))
                x_med = int((center[0] + (r_med * np.cos(rad))))
                y_med = int((center[1] + (r_med * np.sin(rad))))
                cv2.line(img, (x_min, y_min), (x_max, y_max), color_sec, 4, cv2.LINE_AA)
                cv2.circle(img, (x_med, y_med), 5, color_main, (- 1), cv2.LINE_AA)
        cv2.circle(img, center, 34, self.ROBOT_BODY, (- 1), cv2.LINE_AA)
        cv2.circle(img, center, 36, (100, 105, 115), 2, cv2.LINE_AA)
        cv2.line(img, (center[0], (center[1] - 34)), (center[0], (center[1] - 18)), (255, 255, 255), 2, cv2.LINE_AA)

    def draw_wheels(self, img, left, right, max_vel, center=(230, 520)):
        cv2.putText(img, 'Wheels', (100, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.putText(img, 'L', (165, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.putText(img, 'R', (265, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.line(img, (140, center[1]), (290, center[1]), self.PANEL_BORDER, 2, cv2.LINE_AA)
        max_h = 40
        max_vel = max(max_vel, 0.001)
        l_h = int(np.clip(((left / max_vel) * max_h), (- max_h), max_h))
        r_h = int(np.clip(((right / max_vel) * max_h), (- max_h), max_h))
        cv2.rectangle(img, (150, (center[1] - max_h)), (190, (center[1] + max_h)), self.tertiary_color, (- 1))
        cv2.rectangle(img, (250, (center[1] - max_h)), (290, (center[1] + max_h)), self.tertiary_color, (- 1))
        cv2.rectangle(img, (150, (center[1] - l_h)), (190, center[1]), self.primary_color, (- 1))
        cv2.rectangle(img, (250, (center[1] - r_h)), (290, center[1]), self.primary_color, (- 1))

    def draw_grids(self, img, camera_data, is_live=True, active_ganglion=None, active_cones=None, origin=(520, 120)):
        (rows, cols) = ((120 // 20), (160 // 20))
        cell_size = (210 // rows)
        (rf_start_x, rf_start_y) = (origin[0], origin[1])
        (c_start_x, c_start_y) = (origin[0], (origin[1] + 260))
        cv2.putText(img, 'Retinal Ganglion Cells Receptive Fields', (rf_start_x, (rf_start_y - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.putText(img, 'Cone Cells', (c_start_x, (c_start_y - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.TEXT_MAIN, 2, cv2.LINE_AA)
        for r in range(rows):
            for c in range(cols):
                cell_idx = ((r * cols) + c)
                rf_x = (rf_start_x + (c * cell_size))
                rf_y = (rf_start_y + (r * cell_size))
                cp_x = (c_start_x + (c * cell_size))
                cp_y = (c_start_y + (r * cell_size))
                ganglion_active = (True if (active_ganglion is None) else active_ganglion[cell_idx])
                cone_active = (True if (active_cones is None) else active_cones[cell_idx])
                if is_live:
                    idx = (cell_idx * 5)
                    (on_val, off_val, red, green, blue) = camera_data[idx:(idx + 5)]
                    if ganglion_active:
                        if ((on_val > 0.5) and (off_val <= 0.5)):
                            rf_color = self.primary_color
                        elif ((off_val > 0.5) and (on_val <= 0.5)):
                            rf_color = self.secondary_color
                        else:
                            rf_color = self.tertiary_color
                        cv2.rectangle(img, (rf_x, rf_y), (((rf_x + cell_size) - 2), ((rf_y + cell_size) - 2)), rf_color, (- 1))
                    else:
                        cv2.rectangle(img, (rf_x, rf_y), (((rf_x + cell_size) - 2), ((rf_y + cell_size) - 2)), self.PANEL_BORDER, 1)
                    if cone_active:
                        if (max(red, green, blue) <= 1.0):
                            (red, green, blue) = ((red * 255.0), (green * 255.0), (blue * 255.0))
                        cp_color = (int(np.clip(blue, 0, 255)), int(np.clip(green, 0, 255)), int(np.clip(red, 0, 255)))
                        cv2.rectangle(img, (cp_x, cp_y), (((cp_x + cell_size) - 2), ((cp_y + cell_size) - 2)), cp_color, (- 1))
                    else:
                        cv2.rectangle(img, (cp_x, cp_y), (((cp_x + cell_size) - 2), ((cp_y + cell_size) - 2)), self.PANEL_BORDER, 1)
                else:
                    (ganglion_medians, cone_medians) = camera_data
                    if ganglion_active:
                        idx = (cell_idx * 2)
                        on_med = ganglion_medians[idx]
                        off_med = ganglion_medians[(idx + 1)]
                        if ((on_med > 0.5) and (off_med <= 0.5)):
                            rf_color = self.primary_color
                        elif ((off_med > 0.5) and (on_med <= 0.5)):
                            rf_color = self.secondary_color
                        else:
                            rf_color = self.tertiary_color
                        cv2.rectangle(img, (rf_x, rf_y), (((rf_x + cell_size) - 2), ((rf_y + cell_size) - 2)), rf_color, (- 1))
                    else:
                        cv2.rectangle(img, (rf_x, rf_y), (((rf_x + cell_size) - 2), ((rf_y + cell_size) - 2)), self.PANEL_BORDER, 1)
                    if cone_active:
                        median_bgr = cone_medians[cell_idx]
                        cv2.rectangle(img, (cp_x, cp_y), (((cp_x + cell_size) - 2), ((cp_y + cell_size) - 2)), median_bgr, (- 1))
                    else:
                        cv2.rectangle(img, (cp_x, cp_y), (((cp_x + cell_size) - 2), ((cp_y + cell_size) - 2)), self.PANEL_BORDER, 1)

    def draw_transition_graph(self, img, graph_data):
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        if (not nodes):
            cv2.putText(img, 'No graph data', (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, self.neutral_gray, 2)
            return
        order_lower = [s.lower() for s in SKILL_ORDER]
        ordered_nodes = [n for n in order_lower if (n in nodes)]
        for n in nodes:
            if (n not in ordered_nodes):
                ordered_nodes.append(n)
        node_indices = {node: i for (i, node) in enumerate(ordered_nodes)}
        n_nodes = len(ordered_nodes)
        (h, w) = img.shape[:2]
        node_radius = 42
        y_center = (h // 2)
        margin_x = 80
        if (n_nodes > 1):
            step_x = ((w - (2 * margin_x)) / (n_nodes - 1))
            positions = {node: (int((margin_x + (i * step_x))), y_center) for (i, node) in enumerate(ordered_nodes)}
        else:
            positions = {ordered_nodes[0]: ((w // 2), y_center)}
        thickness = 2
        max_weight = 1.0
        min_weight = 0.0
        for edge_info in edges:
            source = edge_info.get('source')
            target = edge_info.get('target')
            weight = edge_info.get('weight', 1)
            if ((weight is None) or (not isinstance(weight, (int, float))) or math.isnan(weight) or math.isinf(weight)):
                weight = 0.0
            if ((source not in positions) or (target not in positions)):
                continue
            pt1 = positions[source]
            pt2 = positions[target]
            idx_src = node_indices[source]
            idx_tgt = node_indices[target]
            if (weight >= 1.0):
                edge_color = self.status_green
            else:
                t = (max(0.0, min(1.0, ((weight - min_weight) / (max_weight - min_weight)))) if (max_weight != min_weight) else 1.0)
                faint_color = tuple((int(((0.85 * bg) + (0.15 * pri))) for (bg, pri) in zip(self.BG_COLOR, self.primary_color)))
                edge_color = tuple((int((((1.0 - t) * faint_color[c]) + (t * self.primary_color[c]))) for c in range(3)))
            if (source == target):
                P1 = np.array([(pt1[0] - 12.0), (pt1[1] - node_radius)], dtype=float)
                P2 = np.array([(pt1[0] + 12.0), (pt1[1] - node_radius)], dtype=float)
                C = np.array([pt1[0], ((pt1[1] - node_radius) - 65.0)], dtype=float)
                t_samples = np.linspace(0, 1, 25).reshape((- 1), 1)
                curve_pts = (((((1 - t_samples) ** 2) * P1) + (((2 * (1 - t_samples)) * t_samples) * C)) + ((t_samples ** 2) * P2)).astype(np.int32)
                cv2.polylines(img, [curve_pts], False, edge_color, thickness, cv2.LINE_AA)
                continue
            is_forward = (idx_src < idx_tgt)
            y_direction = ((- 1.0) if is_forward else 1.0)
            P1 = np.array([pt1[0], (pt1[1] + (y_direction * node_radius))], dtype=float)
            P2 = np.array([pt2[0], (pt2[1] + (y_direction * node_radius))], dtype=float)
            node_step_diff = abs((idx_tgt - idx_src))
            curve_height = (45.0 + ((node_step_diff - 1) * 26.0))
            M = ((P1 + P2) / 2.0)
            C = np.array([M[0], (M[1] + (y_direction * curve_height))])
            t_samples = np.linspace(0, 1, 35).reshape((- 1), 1)
            curve_pts = (((((1 - t_samples) ** 2) * P1) + (((2 * (1 - t_samples)) * t_samples) * C)) + ((t_samples ** 2) * P2)).astype(np.int32)
            cv2.polylines(img, [curve_pts], False, edge_color, thickness, cv2.LINE_AA)
            tangent = (P2 - C)
            t_len = math.hypot(tangent[0], tangent[1])
            if (t_len > 0):
                (tx, ty) = ((tangent[0] / t_len), (tangent[1] / t_len))
                arrow_start = (int((P2[0] - (tx * 16))), int((P2[1] - (ty * 16))))
                arrow_end = (int(P2[0]), int(P2[1]))
                cv2.arrowedLine(img, arrow_start, arrow_end, edge_color, thickness, tipLength=0.45, line_type=cv2.LINE_AA)
            apex = (((P1 + (2 * C)) + P2) / 4.0)
            label_pos_y = (apex[1] + (y_direction * 14.0))
            label_text = (f'{weight:.2f}' if isinstance(weight, float) else str(weight))
            font_lbl = cv2.FONT_HERSHEY_SIMPLEX
            ((lbl_w, lbl_h), _) = cv2.getTextSize(label_text, font_lbl, 0.38, 1)
            lbl_x = int((apex[0] - (lbl_w / 2)))
            lbl_y = int((label_pos_y + ((lbl_h / 2) if (y_direction > 0) else ((- lbl_h) / 2))))
            cv2.putText(img, label_text, (lbl_x, lbl_y), font_lbl, 0.38, self.TEXT_MAIN, 1, cv2.LINE_AA)
        for (node, (x, y)) in positions.items():
            cv2.circle(img, (x, y), node_radius, self.tertiary_color, (- 1))
            cv2.circle(img, (x, y), node_radius, self.primary_color, 2)
            display_name = node.replace('_', ' ').title()
            wrapped_lines = textwrap.wrap(display_name, width=9)
            font = cv2.FONT_HERSHEY_SIMPLEX
            line_metrics = [(*cv2.getTextSize(line, font, 0.36, 1), line) for line in wrapped_lines]
            total_height = (sum(((th + b) for ((tw, th), b, line) in line_metrics)) + (max(0, (len(wrapped_lines) - 1)) * 3))
            current_y = (y - (total_height // 2))
            for ((tw, th), baseline, line) in line_metrics:
                cv2.putText(img, line, (int((x - (tw / 2))), int((current_y + th))), font, 0.36, self.TEXT_MAIN, 1, cv2.LINE_AA)
                current_y += ((th + baseline) + 3)

    def draw_positive_GNG_plane(self, img, G_pos, pos_colors, legend_labels):
        if (len(G_pos.nodes) == 0):
            return
        pos = nx.spring_layout(G_pos, seed=42, k=0.25)
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        (min_x, max_x) = (min(xs), max(xs))
        (min_y, max_y) = (min(ys), max(ys))
        raw_w = ((max_x - min_x) if (max_x > min_x) else 1.0)
        raw_h = ((max_y - min_y) if (max_y > min_y) else 1.0)
        max_r_px = max([max(int((G_pos.nodes[n].get('radius', 0.0) * 35)), 5) for n in G_pos.nodes()], default=5)
        pad_left = (max_r_px + 30)
        pad_right = (max_r_px + 230)
        pad_y_top = (max_r_px + 100)
        pad_y_bot = (max_r_px + 30)
        avail_w = (1000 - (pad_left + pad_right))
        avail_h = (620 - (pad_y_top + pad_y_bot))
        scale = min((avail_w / raw_w), (avail_h / raw_h))
        center_x = ((pad_left + (avail_w / 2)) - (((min_x + max_x) / 2) * scale))
        center_y = ((pad_y_top + (avail_h / 2)) - (((min_y + max_y) / 2) * scale))
        for node in G_pos.nodes():
            pt = (int((center_x + (pos[node][0] * scale))), int((center_y + (pos[node][1] * scale))))
            base_color = pos_colors.get(node, self.neutral_gray)
            light_color = tuple((int(((0.75 * bg) + (0.25 * bc))) for (bg, bc) in zip(self.BG_COLOR, base_color)))
            r_val = G_pos.nodes[node].get('radius', 0.0)
            if (r_val > 0):
                r_px = max(int((r_val * 35)), 5)
                cv2.circle(img, pt, r_px, light_color, (- 1), cv2.LINE_AA)
        for (u, v) in G_pos.edges():
            pt1 = (int((center_x + (pos[u][0] * scale))), int((center_y + (pos[u][1] * scale))))
            pt2 = (int((center_x + (pos[v][0] * scale))), int((center_y + (pos[v][1] * scale))))
            cv2.line(img, pt1, pt2, pos_colors.get(u, self.neutral_gray), 1, cv2.LINE_AA)
        for node in G_pos.nodes():
            pt = (int((center_x + (pos[node][0] * scale))), int((center_y + (pos[node][1] * scale))))
            cv2.circle(img, pt, 5, pos_colors.get(node, self.neutral_gray), (- 1), cv2.LINE_AA)
            cv2.circle(img, pt, 5, self.TEXT_MAIN, 1, cv2.LINE_AA)
        leg_x = 720
        leg_y = 110
        cv2.putText(img, 'Positive Manifolds (TRUE):', (leg_x, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.TEXT_MAIN, 1, cv2.LINE_AA)
        for (i, (pred_name, col)) in enumerate(legend_labels.items()):
            y_pos = ((leg_y + 25) + (i * 20))
            if (y_pos > 600):
                break
            cv2.circle(img, ((leg_x + 10), (y_pos - 4)), 5, col, (- 1), cv2.LINE_AA)
            cv2.putText(img, pred_name[:24], ((leg_x + 25), y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.TEXT_MAIN, 1, cv2.LINE_AA)

    def draw_frequency_plot(self, img, timestamps, bin_size_sec=60.0):
        if ((not timestamps) or (len(timestamps) < 2)):
            cv2.putText(img, 'Insufficient timestamp data', (350, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.neutral_gray, 2, cv2.LINE_AA)
            return
        ts = np.sort(np.array(timestamps, dtype=float))
        (t_start, t_end) = (ts[0], ts[(- 1)])
        duration = max((t_end - t_start), 1.0)
        n_bins = max(int(np.ceil((duration / bin_size_sec))), 1)
        bin_edges = np.linspace(t_start, (t_start + (n_bins * bin_size_sec)), (n_bins + 1))
        (counts, _) = np.histogram(ts, bins=bin_edges)
        (plot_x, plot_y) = (90, 140)
        (plot_w, plot_h) = (820, 380)
        cv2.rectangle(img, (plot_x, plot_y), ((plot_x + plot_w), (plot_y + plot_h)), self.PANEL_BORDER, 1)
        avg_freq = (len(ts) / (duration / 60.0))
        stat_text = f'Total: {len(ts)} updates | Duration: {duration:.1f}s | Avg Rate: {avg_freq:.2f} updates/min'
        cv2.putText(img, stat_text, (plot_x, (plot_y - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_MAIN, 1, cv2.LINE_AA)
        max_count = max(int(np.max(counts)), 1)
        grid_steps = 4
        for i in range((grid_steps + 1)):
            y_val = ((plot_y + plot_h) - int((i * (plot_h / grid_steps))))
            val_label = f'{int((i * (max_count / grid_steps)))}'
            cv2.line(img, (plot_x, y_val), ((plot_x + plot_w), y_val), self.tertiary_color, 1, cv2.LINE_AA)
            cv2.putText(img, val_label, ((plot_x - 35), (y_val + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.neutral_gray, 1, cv2.LINE_AA)
        bar_w = max((int((plot_w / n_bins)) - 3), 2)
        curve_pts = []
        for (i, count) in enumerate(counts):
            bar_h = int(((count / max_count) * (plot_h - 20)))
            bx = ((plot_x + int((i * (plot_w / n_bins)))) + 2)
            by = ((plot_y + plot_h) - bar_h)
            cv2.rectangle(img, (bx, by), ((bx + bar_w), (plot_y + plot_h)), self.secondary_color, (- 1))
            cv2.rectangle(img, (bx, by), ((bx + bar_w), (plot_y + plot_h)), self.primary_color, 1)
            curve_pts.append(((bx + (bar_w // 2)), by))
        if (len(curve_pts) > 1):
            for i in range((len(curve_pts) - 1)):
                cv2.line(img, curve_pts[i], curve_pts[(i + 1)], self.BORDER_COLOR, 2, cv2.LINE_AA)
            for pt in curve_pts:
                cv2.circle(img, pt, 3, self.primary_color, (- 1), cv2.LINE_AA)
        x_label = f'Elapsed Time (Bins of {int(bin_size_sec)}s) ->'
        cv2.putText(img, x_label, (((plot_x + (plot_w // 2)) - 100), ((plot_y + plot_h) + 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_MAIN, 1, cv2.LINE_AA)

    def draw_skill_table(self, img, table_data, threshold=0.75):
        (table_x, table_y) = (40, 115)
        (total_w, total_h) = (1120, 440)
        row_height = 80
        header_col_w = 210
        skills = table_data.get('skills', [])
        n_cols = len(skills)
        if (n_cols == 0):
            cv2.putText(img, 'No skill data available', ((table_x + 100), (table_y + 100)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.neutral_gray, 2)
            return
        col_w = int(((total_w - header_col_w) / n_cols))
        actual_total_w = (header_col_w + (n_cols * col_w))
        cv2.rectangle(img, (table_x, table_y), ((table_x + actual_total_w), (table_y + total_h)), self.BORDER_COLOR, 2)
        header_h = 40
        cv2.rectangle(img, (table_x, table_y), ((table_x + actual_total_w), (table_y + header_h)), self.tertiary_color, (- 1))
        cv2.line(img, (table_x, (table_y + header_h)), ((table_x + actual_total_w), (table_y + header_h)), self.BORDER_COLOR, 2)
        cv2.putText(img, 'Skill Metric / Property', ((table_x + 10), (table_y + 26)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.TEXT_MAIN, 1, cv2.LINE_AA)
        for (i, sk) in enumerate(skills):
            cx = ((table_x + header_col_w) + (i * col_w))
            short_name = sk.replace('_', ' ')
            lines = textwrap.wrap(short_name, width=12)
            for (li, line) in enumerate(lines[:2]):
                cv2.putText(img, line, ((cx + 8), ((table_y + 16) + (li * 16))), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.TEXT_MAIN, 1, cv2.LINE_AA)
        row_labels = ['1. Next Action', f'2. Overlap (Eff -> Init) [>={threshold:.2f}]', '3. Nodes in positive GNG', '4. Nodes in negative GNG']
        curr_y = (table_y + header_h)
        for (r_idx, label) in enumerate(row_labels):
            y1 = (curr_y + (r_idx * row_height))
            y2 = (y1 + row_height)
            cv2.line(img, (table_x, y2), ((table_x + actual_total_w), y2), self.PANEL_BORDER, 1)
            lines = textwrap.wrap(str(label), width=25)
            for (li, l_text) in enumerate(lines[:3]):
                cv2.putText(img, l_text, ((table_x + 6), ((y1 + 25) + (li * 20))), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_MAIN, 1, cv2.LINE_AA)
            for (c_idx, sk) in enumerate(skills):
                cx1 = ((table_x + header_col_w) + (c_idx * col_w))
                cx2 = (cx1 + col_w)
                center_cell_x = (cx1 + 10)
                center_cell_y = (y1 + 45)
                if (r_idx == 0):
                    action_val = table_data.get('next_action', {}).get(sk, '-')
                    lines = textwrap.wrap(str(action_val.replace('_', ' ')), width=15)
                    for (li, l_text) in enumerate(lines[:3]):
                        cv2.putText(img, l_text, ((cx1 + 6), ((y1 + 25) + (li * 20))), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_MAIN, 1, cv2.LINE_AA)
                elif (r_idx == 1):
                    overlap_val = table_data.get('overlap', {}).get(sk, None)
                    if ((overlap_val is not None) and (not math.isnan(overlap_val))):
                        is_pass = (overlap_val >= threshold)
                        text_val = f'{(overlap_val * 100):.1f}%'
                        bg_c = ((215, 245, 215) if is_pass else (230, 230, 245))
                        fg_c = (self.status_green if is_pass else self.status_red)
                        cv2.rectangle(img, ((cx1 + 4), (y1 + 15)), ((cx2 - 4), (y2 - 15)), bg_c, (- 1))
                        cv2.rectangle(img, ((cx1 + 4), (y1 + 15)), ((cx2 - 4), (y2 - 15)), fg_c, 1)
                        cv2.putText(img, text_val, ((cx1 + 14), center_cell_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_MAIN, 1, cv2.LINE_AA)
                    else:
                        cv2.putText(img, 'N/A', (center_cell_x, center_cell_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_MAIN, 1, cv2.LINE_AA)
                elif (r_idx == 2):
                    pos_count = table_data.get('pos_nodes', {}).get(sk, 0)
                    cv2.putText(img, str(pos_count), (center_cell_x, center_cell_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_MAIN, 1, cv2.LINE_AA)
                elif (r_idx == 3):
                    neg_count = table_data.get('neg_nodes', {}).get(sk, 0)
                    cv2.putText(img, str(neg_count), (center_cell_x, center_cell_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.line(img, ((table_x + header_col_w), table_y), ((table_x + header_col_w), (table_y + total_h)), self.BORDER_COLOR, 2)
        for i in range(1, n_cols):
            vx = ((table_x + header_col_w) + (i * col_w))
            cv2.line(img, (vx, table_y), (vx, (table_y + total_h)), self.PANEL_BORDER, 1)
class SymbolIconPictureGenerator(PictureGenerator):

    def run(self):
        if (not self.session_path):
            return
        gng_dir = _find_pddl_gng_dir(self.session_path)
        if ((not gng_dir) or (not os.path.exists(gng_dir))):
            return
        out_dir = self.get_pictures_dir()
        file_pattern = os.path.join(gng_dir, 'sym_*.json')
        for json_path in glob.glob(file_pattern):
            try:
                (full_nodes, active_set, _) = _load_symbol_data(json_path)
                if ((full_nodes is None) or (len(full_nodes) == 0)):
                    continue
                file_base = os.path.splitext(os.path.basename(json_path))[0]
                symbol_num = file_base.split('_')[(- 1)]
                centroid = np.mean(full_nodes, axis=0)
                dists = np.linalg.norm((full_nodes - centroid), axis=1)
                rep_node = full_nodes[int(np.argmin(dists))]
                (lidar_active, active_ganglion, active_cones) = _build_masks(active_set)
                display_title = f'Symbol {symbol_num}'
                img = self.renderer.create_base_canvas(display_title, f'Symbol Icon')
                self.renderer.draw_lidar(img, rep_node[:8], center=(230, 280), mode='iconic', active_flags=lidar_active, custom_color=self.renderer.primary_color)
                self.renderer.draw_grids(img, rep_node[8:248], is_live=True, active_ganglion=active_ganglion, active_cones=active_cones, origin=(520, 120))
                self.renderer.draw_sample_badge(img, count=len(full_nodes), position=(35, 595))
                self.renderer.draw_sample_badge(img, count=len(active_set), label='Active Dimentions', position=(135, 595))
                out_path = os.path.join(out_dir, f'{file_base}_icon.png')
                cv2.imwrite(out_path, img)
                print(f'[SymbolIconPictureGenerator] Saved icon to {out_path}')
            except Exception as e:
                print(f'[SymbolIconPictureGenerator] Error processing {json_path}: {e}')
class SymbolCategoryPictureGenerator(PictureGenerator):

    def run(self):
        if (not self.session_path):
            return
        gng_dir = _find_pddl_gng_dir(self.session_path)
        if ((not gng_dir) or (not os.path.exists(gng_dir))):
            return
        out_dir = self.get_pictures_dir()
        file_pattern = os.path.join(gng_dir, 'sym_*.json')
        for json_path in glob.glob(file_pattern):
            try:
                (full_nodes, active_set, _) = _load_symbol_data(json_path)
                if ((full_nodes is None) or (len(full_nodes) == 0)):
                    continue
                file_base = os.path.splitext(os.path.basename(json_path))[0]
                symbol_num = file_base.split('_')[(- 1)]
                median_vals = lower_median(full_nodes, axis=0)
                mad = np.median(np.abs((full_nodes - median_vals)), axis=0)
                lower_limit_vals = (median_vals - mad)
                upper_limit_vals = (median_vals + mad)
                (lidar_active, active_ganglion, active_cones) = _build_masks(active_set)
                display_title = f'Symbol {symbol_num}'
                img = self.renderer.create_base_canvas(display_title, f'Symbol Category')
                lidar_data = list(zip(lower_limit_vals[:8], upper_limit_vals[:8], median_vals[:8]))
                self.renderer.draw_lidar(img, lidar_data, mode='categorical', active_flags=lidar_active)
                ganglion_medians = []
                for cell_idx in range(48):
                    idx_on = (8 + (cell_idx * 5))
                    idx_off = (idx_on + 1)
                    ganglion_medians.append(median_vals[idx_on])
                    ganglion_medians.append(median_vals[idx_off])
                cone_medians = []
                for cell_idx in range(48):
                    offset = (8 + (cell_idx * 5))
                    r_p = full_nodes[(:, (offset + 2))]
                    g_p = full_nodes[(:, (offset + 3))]
                    b_p = full_nodes[(:, (offset + 4))]
                    (median_bgr, _) = self.renderer.compute_hue_median_stats(r_p, g_p, b_p)
                    cone_medians.append(median_bgr)
                self.renderer.draw_grids(img, (ganglion_medians, cone_medians), is_live=False, active_ganglion=active_ganglion, active_cones=active_cones, origin=(520, 120))
                self.renderer.draw_sample_badge(img, count=len(full_nodes), position=(35, 595))
                self.renderer.draw_sample_badge(img, count=len(active_set), label='Active Dimentions', position=(135, 595))
                out_path = os.path.join(out_dir, f'{file_base}_category.png')
                cv2.imwrite(out_path, img)
                print(f'[SymbolCategoryPictureGenerator] Saved category image to {out_path}')
            except Exception as e:
                print(f'[SymbolCategoryPictureGenerator] Error processing {json_path}: {e}')
@dataclass
class CumulativeExecutionStats():
    num_sessions: int = 0
    total_duration_sec: int = 0
    total_runs: int = 0
    total_brain_updates: int = 0
    total_completed_plans: int = 0
    final_skills_data: dict[(str, SkillFinalStats)] = field(default_factory=dict)
class CumulativeStatsScraper():

    def __init__(self, logs_root: str='logs', session_dir: (str | None)=None):
        self.logs_root = logs_root
        self.session_path = session_dir

    @staticmethod
    def parse_continuation(stats_file: Path) -> (Path | None):
        if (not stats_file.exists()):
            return None
        pattern = re.compile('CONTINUATION FROM SESSION:\\s*(.+)', re.IGNORECASE)
        with stats_file.open('r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line.strip())
                if match:
                    return Path(match.group(1).strip())
        return None

    def collect_chain_folders(self, current_dir: Path, logs_root: Path) -> list[Path]:
        visited_dirs = {current_dir.resolve()}
        chain_dirs: list[Path] = [current_dir]
        curr = current_dir
        while True:
            stats_file = (curr / 'execution_statistics.txt')
            continued_rel = self.parse_continuation(stats_file)
            if (not continued_rel):
                break
            if (logs_root.parent / continued_rel).exists():
                next_dir = (logs_root.parent / continued_rel)
            elif (logs_root / continued_rel.name).exists():
                next_dir = (logs_root / continued_rel.name)
            else:
                break
            next_resolved = next_dir.resolve()
            if (next_resolved in visited_dirs):
                break
            visited_dirs.add(next_resolved)
            chain_dirs.append(next_dir)
            curr = next_dir
        chain_dirs.reverse()
        return chain_dirs

    @staticmethod
    def parse_duration_to_seconds(duration_str: str) -> int:
        parts = list(map(int, duration_str.strip().split(':')))
        if (len(parts) == 3):
            return (((parts[0] * 3600) + (parts[1] * 60)) + parts[2])
        if (len(parts) == 2):
            return ((parts[0] * 60) + parts[1])
        return parts[0]

    @staticmethod
    def format_total_seconds(total_seconds: int) -> str:
        hours = (total_seconds // 3600)
        remainder = (total_seconds % 3600)
        minutes = (remainder // 60)
        seconds = (remainder % 60)
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

    def parse_session_aggregates(self, stats_file: Path) -> tuple[(int, int, int, int)]:
        duration_sec = 0
        runs = 0
        brain_updates = 0
        completed_plans = 0
        duration_pat = re.compile('duration:\\s*([0-9:]+)', re.IGNORECASE)
        stats_pat = re.compile('runs:\\s*(\\d+);\\s*brain updates:\\s*(\\d+);\\s*completed plans:\\s*(\\d+)', re.IGNORECASE)
        with stats_file.open('r', encoding='utf-8') as f:
            for line in f:
                d_match = duration_pat.search(line)
                if d_match:
                    duration_sec = self.parse_duration_to_seconds(d_match.group(1))
                s_match = stats_pat.search(line)
                if s_match:
                    runs = int(s_match.group(1))
                    brain_updates = int(s_match.group(2))
                    completed_plans = int(s_match.group(3))
        return (duration_sec, runs, brain_updates, completed_plans)

    def parse_final_session_skills(self, stats_file: Path) -> dict[(str, SkillFinalStats)]:
        skills_nodes: dict[(str, int)] = {}
        skills_ratios: dict[(str, (float | None))] = {}
        skills_positive_ratios: dict[(str, (float | None))] = {}
        skills_negative_ratios: dict[(str, (float | None))] = {}
        mode = None
        with stats_file.open('r', encoding='utf-8') as f:
            for line in f:
                raw_line = line.strip()
                if ((not raw_line) or raw_line.startswith('---')):
                    continue
                if ('Collected Data:' in raw_line):
                    mode = 'collected_data'
                    continue
                elif ('Executions:' in raw_line):
                    mode = 'executions'
                    continue
                parts = [p.strip() for p in raw_line.split('|')]
                if ((len(parts) < 2) or (parts[0].lower() == 'skill name')):
                    continue
                skill_name = parts[0]
                if ((mode == 'collected_data') and (len(parts) >= 2)):
                    try:
                        skills_nodes[skill_name] = int(parts[1])
                    except ValueError as e:
                        print(f'[CS] {e}')
                        continue
                elif ((mode == 'executions') and (len(parts) >= 5)):
                    try:
                        tp = float(parts[1])
                        tn = float(parts[2])
                        fp = float(parts[3])
                        fn = float(parts[4])
                        okay = ((((fn + tp) + fp) + tn) > 0)
                        skills_ratios[skill_name] = (((tp + tn) / (((fn + tp) + fp) + tn)) if okay else None)
                        okay = ((tp + fn) > 0)
                        skills_positive_ratios[skill_name] = ((tp / (tp + fn)) if okay else None)
                        okay = ((tn + fp) > 0)
                        skills_negative_ratios[skill_name] = ((tn / (tn + fp)) if okay else None)
                    except ValueError as e:
                        print(f'[CS] {e}')
                        continue
        all_skills = set(skills_nodes.keys()).union(skills_ratios.keys())
        results: dict[(str, SkillFinalStats)] = {}
        for skill in all_skills:
            results[skill] = SkillFinalStats(nodes=skills_nodes.get(skill, 0), ratio=skills_ratios.get(skill, None), pos_ratio=skills_positive_ratios.get(skill, None), neg_ratio=skills_negative_ratios.get(skill, None))
        return results

    def scrape(self) -> (CumulativeExecutionStats | None):
        if ((not self.session_path) or (not os.path.exists(self.session_path))):
            print('[CS] Session path do not exists')
            return None
        current_dir = Path(self.session_path)
        logs_root = Path(self.logs_root)
        chain_dirs = self.collect_chain_folders(current_dir, logs_root)
        if (not chain_dirs):
            print('[CS] No chain of dirs found')
            return None
        cumulative = CumulativeExecutionStats(num_sessions=len(chain_dirs))
        for s_dir in chain_dirs:
            s_file = (s_dir / 'execution_statistics.txt')
            if (not s_file.exists()):
                print(f"[CS] No 'execution_statistics.txt' in {s_dir}")
                continue
            (duration_sec, runs, updates, plans) = self.parse_session_aggregates(s_file)
            cumulative.total_duration_sec += duration_sec
            cumulative.total_runs += runs
            cumulative.total_brain_updates += updates
            cumulative.total_completed_plans += plans
        final_stats_file = (chain_dirs[(- 1)] / 'execution_statistics.txt')
        if final_stats_file.exists():
            print(f"[CS] No 'execution_statistics.txt' in {s_dir}")
            cumulative.final_skills_data = self.parse_final_session_skills(final_stats_file)
        return cumulative

    def save_to_txt(self, stats: CumulativeExecutionStats, output_path: (str | Path)) -> None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        duration_formatted = self.format_total_seconds(stats.total_duration_sec)
        lines: list[str] = [('-' * 60), 'CUMULATIVE EXECUTION STATISTICS', ('-' * 60), f'number of sessions: {stats.num_sessions}', f'total duration: {duration_formatted}', f'total number of runs: {stats.total_runs}', f'total number of brain updates: {stats.total_brain_updates}', f'total number of completed plans: {stats.total_completed_plans}', ('-' * 60), 'FINAL SESSION SKILLS', ('-' * 100), f"{'Skill Name':<20} | {'Nodes':<10} | {'true/all ratio':<20} | {'true success ratio':<20} | {'true fail ratio':<20} ", ('-' * 100)]
        for (skill, data) in stats.final_skills_data.items():
            ratio_str = (f'{data.ratio:.3f}' if (data.ratio is not None) else 'N/A')
            pos_ratio_str = (f'{data.pos_ratio:.3f}' if (data.pos_ratio is not None) else 'N/A')
            neg_ratio_str = (f'{data.neg_ratio:.3f}' if (data.neg_ratio is not None) else 'N/A')
            eps = 0.01
            tag = (" \t(●'◡'●)" if ((data.ratio is not None) and (data.ratio >= (1 - eps)) and (data.pos_ratio is not None) and (data.pos_ratio >= (1 - eps))) else '')
            lines.append(f'{skill:<20} | {data.nodes:<10} | {ratio_str:<20} | {pos_ratio_str:<20} | {neg_ratio_str:<20}{tag}')
        lines.append((('-' * 100) + '\n'))
        with output_file.open('w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def export_stats(self, output_filename: str='cumulative_statistics.txt') -> (Path | None):
        stats = self.scrape()
        if ((not stats) or (not self.session_path)):
            print('[Cumulative Stats] No stats or session path.')
            return None
        out_path = (Path(self.session_path) / output_filename)
        self.save_to_txt(stats, out_path)
        return out_path
if (__name__ == '__main__'):
    generator = ExecStatsGenerator()
    generator.run()
    sd = None
    gen = PipelinePictureGenerator(session_dir=sd)
    gen.run()
    generator = SymbolStatsGenerator()
    generator.run()
    scraper = CumulativeStatsScraper(logs_root='logs', session_dir='logs\\2026-11-04_07-00')
    scraper.export_stats()
    generator = HistogramPlotGenerator()
    generator.run()
