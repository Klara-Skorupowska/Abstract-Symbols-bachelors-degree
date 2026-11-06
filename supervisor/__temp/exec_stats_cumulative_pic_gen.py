from __future__ import annotations

import os
from pathlib import Path
import re

import cv2
import numpy as np

from supervisor.picture_generator import PictureGenerator


class ExecStatsGenerator(PictureGenerator):
    """Generates execution trends and stacked outcome plots using OpenCV and Renderer style."""

    SKILL_PALETTE = [
        (195, 105, 30),   # Slate Blue
        (71, 90, 224),    # Coral Red
        (74, 168, 43),    # Green
        (6, 119, 217),    # Amber
        (246, 92, 139),   # Purple
        (212, 182, 6),    # Cyan
        (153, 72, 236),   # Pink
        (139, 116, 100),  # Slate
    ]

    def __init__(self, logs_root: str = "logs", session_dir: str | None = None):
        super().__init__(logs_root=logs_root, session_dir=session_dir)

    @staticmethod
    def parse_continuation(stats_file: Path) -> Path | None:
        if not stats_file.exists():
            return None
        pattern = re.compile(r"CONTINUATION FROM SESSION:\s*(.+)", re.IGNORECASE)
        with stats_file.open("r", encoding="utf-8") as f:
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
            stats_file = curr / "execution_statistics.txt"
            continued_rel = self.parse_continuation(stats_file)
            if not continued_rel:
                break

            if (logs_root.parent / continued_rel).exists():
                next_dir = logs_root.parent / continued_rel
            elif (logs_root / continued_rel.name).exists():
                next_dir = logs_root / continued_rel.name
            else:
                break

            next_resolved = next_dir.resolve()
            if next_resolved in visited_dirs:
                break

            visited_dirs.add(next_resolved)
            chain_dirs.append(next_dir)
            curr = next_dir

        chain_dirs.reverse()
        return chain_dirs

    @staticmethod
    def parse_execution_statistics(file_path: Path) -> dict:
        if not file_path.exists():
            return {
                "collected_data": {},
                "executions": {},
                "tp": {},
                "tn": {},
                "correct_counts": {},
                "incorrect_counts": {},
            }

        collected_data = {}
        executions = {}
        tp = {}
        tn = {}
        correct_counts = {}
        incorrect_counts = {}
        mode = None

        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue

                if "Collected Data:" in line_str:
                    mode = "collected_data"
                    continue
                elif "Executions:" in line_str:
                    mode = "executions"
                    continue
                elif line_str.startswith("---") or line_str.startswith("start time:") or line_str.startswith("CONTINUATION"):
                    continue

                parts = [p.strip() for p in line_str.split("|")]
                if len(parts) >= 2 and parts[0].lower().startswith("skill name"):
                    continue

                if mode == "collected_data" and len(parts) >= 2:
                    skill = parts[0]
                    try:
                        collected_data[skill] = int(parts[1])
                    except ValueError:
                        pass
                elif mode == "executions" and len(parts) >= 6:
                    # Skill Name | True Positive | True Negative | False Positive | False Negative | Total
                    skill = parts[0]
                    try:
                        tp_val = int(parts[1])
                        tn_val = int(parts[2])
                        fp_val = int(parts[3])
                        fn_val = int(parts[4])
                        total_val = int(parts[5])

                        both = max(tp_val + tn_val, 1)
                        tp[skill] = tp_val / both
                        tn[skill] = tn_val / both
                        executions[skill] = total_val

                        # Raw counts for (TP + TN) vs (FP + FN)
                        correct_counts[skill] = tp_val + tn_val
                        incorrect_counts[skill] = fp_val + fn_val
                    except ValueError:
                        pass

        return {
            "collected_data": collected_data,
            "executions": executions,
            "tp": tp,
            "tn": tn,
            "correct_counts": correct_counts,
            "incorrect_counts": incorrect_counts,
        }

    def render_trend_plot(
        self,
        n_sessions: int,
        skills_data: dict[str, list[int]],
        title: str,
        y_label: str,
        width: int = 1000,
        height: int = 620,
    ) -> np.ndarray:
        img = self.renderer.create_base_canvas(title.upper(), width=width, height=height)

        plot_x, plot_y = 90, 110
        plot_w, plot_h = 710, 430
        cv2.rectangle(img, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), self.renderer.PANEL_BORDER, 1)

        all_vals = [v for vals in skills_data.values() for v in vals]
        max_val = max(max(all_vals) * 1.05, 1.0) if all_vals else 1.0

        # Horizontal gridlines & Y ticks
        grid_steps = 4
        for i in range(grid_steps + 1):
            y_val = plot_y + plot_h - int(i * (plot_h / grid_steps))
            cv2.line(img, (plot_x, y_val), (plot_x + plot_w, y_val), self.renderer.tertiary_color, 1, cv2.LINE_AA)
            cv2.putText(img, f"{int(i * (max_val / grid_steps))}", (plot_x - 45, y_val + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.renderer.neutral_gray, 1, cv2.LINE_AA)

        # X axis: continuous mapping labeled every 10 units (1, 10, 20, 30...)
        def get_x_px(s_idx: int) -> int:
            if n_sessions <= 1:
                return plot_x + plot_w // 2
            return plot_x + int((s_idx / (n_sessions - 1)) * plot_w)

        # Draw session ticks & labels every 10 steps
        tick_sessions = [1] + [s for s in range(10, n_sessions + 1, 10)]
        for s_num in tick_sessions:
            tx = get_x_px(s_num - 1)
            cv2.line(img, (tx, plot_y + plot_h), (tx, plot_y + plot_h + 5), self.renderer.BORDER_COLOR, 1, cv2.LINE_AA)
            cv2.putText(img, str(s_num), (tx - 8, plot_y + plot_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

        legend_x = plot_x + plot_w + 20
        legend_y = plot_y + 20

        # Plot curves
        for idx, (skill_name, vals) in enumerate(skills_data.items()):
            color = self.SKILL_PALETTE[idx % len(self.SKILL_PALETTE)]
            pts = []
            for s_idx, val in enumerate(vals):
                xi = get_x_px(s_idx)
                yi = plot_y + plot_h - int((val / max_val) * plot_h)
                pts.append((xi, yi))

            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(img, pt, 3, color, -1, cv2.LINE_AA)

            ly = legend_y + idx * 24
            if ly < height - 60:
                cv2.rectangle(img, (legend_x, ly - 8), (legend_x + 12, ly + 4), color, -1)
                cv2.putText(img, skill_name[:16], (legend_x + 18, ly + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

        cv2.putText(img, "Session", (plot_x + plot_w // 2 - 25, plot_y + plot_h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.putText(img, y_label, (plot_x - 30, plot_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

        return img

    def render_stacked_outcomes_count(
        self,
        sessions: int | list,
        all_skills: list[str],
        correct_data: dict[str, list[int]],
        incorrect_data: dict[str, list[int]],
        title: str = "Skill Executions Count",
        width: int = 1000,
    ) -> np.ndarray:
        n_sessions = len(sessions) if isinstance(sessions, list) else int(sessions)
        n_skills = len(all_skills)
        row_height = 80
        total_h = max(620, 130 + n_skills * row_height)
        img = self.renderer.create_base_canvas(title.upper(), width=width, height=total_h)

        if n_skills == 0:
            return img

        plot_x, start_y = 150, 100
        plot_w = 780
        bar_w = max(int((plot_w / n_sessions) * 0.7), 2)

        color_correct = self.renderer.status_green
        color_incorrect = self.renderer.status_red

        # Top Header Legend
        leg_y = 45
        cv2.rectangle(img, (630, leg_y - 10), (645, leg_y + 2), color_correct, -1)
        cv2.putText(img, "Expected Outcome", (652, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.rectangle(img, (800, leg_y - 10), (815, leg_y + 2), color_incorrect, -1)
        cv2.putText(img, "Unexpected Outcome", (822, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

        # Global maximum across all skills and all sessions for a shared Y-axis scale
        global_max = 1.0
        for skill in all_skills:
            c_vals = correct_data.get(skill, [])
            inc_vals = incorrect_data.get(skill, [])
            for c, inc in zip(c_vals, inc_vals):
                total = c + inc
                if total > global_max:
                    global_max = float(total)

        common_y_max = max(global_max * 1.15, 1.0)

        for s_idx, skill in enumerate(all_skills):
            cur_y = start_y + s_idx * row_height
            sub_h = row_height - 20
            cv2.rectangle(img, (plot_x, cur_y), (plot_x + plot_w, cur_y + sub_h), self.renderer.PANEL_BORDER, 1)
            cv2.putText(img, skill[:16], (35, cur_y + sub_h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

            # Common Y-axis boundary labels per subplot
            cv2.putText(img, f"{int(common_y_max)}", (plot_x - 38, cur_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, self.renderer.neutral_gray, 1, cv2.LINE_AA)
            cv2.putText(img, "0", (plot_x - 18, cur_y + sub_h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.32, self.renderer.neutral_gray, 1, cv2.LINE_AA)

            c_vals = correct_data.get(skill, [0] * n_sessions)
            inc_vals = incorrect_data.get(skill, [0] * n_sessions)

            for i in range(n_sessions):
                bx = plot_x + int((i + 0.5) * (plot_w / n_sessions)) - bar_w // 2
                base_y = cur_y + sub_h

                c_h = int((c_vals[i] / common_y_max) * (sub_h - 6))
                inc_h = int((inc_vals[i] / common_y_max) * (sub_h - 6))

                if c_h > 0:
                    cv2.rectangle(img, (bx, base_y - c_h), (bx + bar_w, base_y), color_correct, -1)
                if inc_h > 0:
                    cv2.rectangle(img, (bx, base_y - c_h - inc_h), (bx + bar_w, base_y - c_h), color_incorrect, -1)

            # Continuous tick labels every 10 sessions on bottom row
            if s_idx == n_skills - 1:
                tick_sessions = [1] + [s for s in range(10, n_sessions + 1, 10)]
                for s_num in tick_sessions:
                    bx = plot_x + int((s_num - 0.5) * (plot_w / n_sessions))
                    cv2.line(img, (bx, cur_y + sub_h), (bx, cur_y + sub_h + 5), self.renderer.BORDER_COLOR, 1, cv2.LINE_AA)
                    cv2.putText(img, str(s_num), (bx - 8, cur_y + sub_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

        return img

    def render_stacked_expected_outcome(
        self,
        n_sessions: int,
        all_skills: list[str],
        tp_data: dict[str, list[float]],
        tn_data: dict[str, list[float]],
        title: str,
        width: int = 1000,
    ) -> np.ndarray:
        n_skills = len(all_skills)
        row_height = 80
        total_h = max(620, 130 + n_skills * row_height)
        img = self.renderer.create_base_canvas(title.upper(), width=width, height=total_h)

        if n_skills == 0:
            return img

        plot_x, start_y = 150, 100
        plot_w = 780
        bar_w = max(int((plot_w / n_sessions) * 0.7), 2)

        color_tp = self.renderer.status_green
        color_tn = self.renderer.status_red

        leg_y = 45
        cv2.rectangle(img, (700, leg_y - 10), (715, leg_y + 2), color_tp, -1)
        cv2.putText(img, "True Positive", (722, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.rectangle(img, (820, leg_y - 10), (835, leg_y + 2), color_tn, -1)
        cv2.putText(img, "True Negative", (842, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

        for s_idx, skill in enumerate(all_skills):
            cur_y = start_y + s_idx * row_height
            sub_h = row_height - 20
            cv2.rectangle(img, (plot_x, cur_y), (plot_x + plot_w, cur_y + sub_h), self.renderer.PANEL_BORDER, 1)
            cv2.putText(img, skill[:16], (35, cur_y + sub_h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

            tp_vals = tp_data[skill]
            tn_vals = tn_data[skill]

            for i in range(n_sessions):
                bx = plot_x + int((i + 0.5) * (plot_w / n_sessions)) - bar_w // 2
                base_y = cur_y + sub_h
                tp_h = int(tp_vals[i] * (sub_h - 6))
                tn_h = int(tn_vals[i] * (sub_h - 6))

                if tp_h > 0:
                    cv2.rectangle(img, (bx, base_y - tp_h), (bx + bar_w, base_y), color_tp, -1)
                if tn_h > 0:
                    cv2.rectangle(img, (bx, base_y - tp_h - tn_h), (bx + bar_w, base_y - tp_h), color_tn, -1)

            # Continuous tick labels every 10 sessions on bottom row
            if s_idx == n_skills - 1:
                tick_sessions = [1] + [s for s in range(10, n_sessions + 1, 10)]
                for s_num in tick_sessions:
                    bx = plot_x + int((s_num - 0.5) * (plot_w / n_sessions))
                    cv2.line(img, (bx, cur_y + sub_h), (bx, cur_y + sub_h + 5), self.renderer.BORDER_COLOR, 1, cv2.LINE_AA)
                    cv2.putText(img, str(s_num), (bx - 8, cur_y + sub_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

        return img

    def _resolve_file_path(self) -> Path | None:
        """Resolves the path to execution_statistics.txt in the target session."""
        if not self.session_path or not os.path.exists(self.session_path):
            return None
        target = Path(self.session_path) / "execution_statistics.txt"
        return target if target.exists() else None

    def run(self):
        file_path = self._resolve_file_path()
        if not file_path:
            return

        try:
            current_dir = Path(self.session_path)
            logs_root = Path(self.logs_root)
            chain_dirs = self.collect_chain_folders(current_dir, logs_root)

            n_sessions = len(chain_dirs)
            all_skills: list[str] = []
            session_stats: list[dict] = []

            for s_dir in chain_dirs:
                stats = self.parse_execution_statistics(s_dir / "execution_statistics.txt")
                session_stats.append(stats)
                for skill in list(stats["collected_data"].keys()) + list(stats["executions"].keys()):
                    if skill not in all_skills:
                        all_skills.append(skill)

            data_nodes_by_skill = {skill: [] for skill in all_skills}
            tp_by_skill = {skill: [] for skill in all_skills}
            tn_by_skill = {skill: [] for skill in all_skills}
            correct_by_skill = {skill: [] for skill in all_skills}
            incorrect_by_skill = {skill: [] for skill in all_skills}

            for stats in session_stats:
                for skill in all_skills:
                    data_nodes_by_skill[skill].append(stats["collected_data"].get(skill, 0))
                    tp_by_skill[skill].append(stats["tp"].get(skill, 0.0))
                    tn_by_skill[skill].append(stats["tn"].get(skill, 0.0))
                    correct_by_skill[skill].append(stats["correct_counts"].get(skill, 0))
                    incorrect_by_skill[skill].append(stats["incorrect_counts"].get(skill, 0))

            out_dir = self.get_pictures_dir() or "."

            # 1. Collected Data Nodes Trend
            img_nodes = self.render_trend_plot(
                n_sessions, data_nodes_by_skill, "Collected Data Nodes Trend", "Nodes Count"
            )
            out_path_nodes = os.path.join(out_dir, "skills_collected_data_trend.png")
            cv2.imwrite(out_path_nodes, img_nodes)

            # 2. Executions: (TP + TN) vs (FP + FN) Stacked Counts
            img_exec_outcomes = self.render_stacked_outcomes_count(
                n_sessions,
                all_skills,
                correct_by_skill,
                incorrect_by_skill,
                title="Skill Executions Count",
            )
            out_path_exec = os.path.join(out_dir, "skills_executions_trend.png")
            cv2.imwrite(out_path_exec, img_exec_outcomes)

            # 3. Expected Outcome Executions: TP vs TN relative stacked bars
            img_stacked = self.render_stacked_expected_outcome(
                n_sessions, all_skills, tp_by_skill, tn_by_skill, "Expected Outcome Executions"
            )
            out_path_stacked = os.path.join(out_dir, "skills_expected_outcome_executions.png")
            cv2.imwrite(out_path_stacked, img_stacked)

            print(f"[ExecStatsGenerator] Saved execution stats plots to {out_dir}")
        except Exception as e:
            print(f"[ExecStatsGenerator] Failed to generate execution statistics plots: {e}")

if __name__ == "__main__":
    generator = ExecStatsGenerator()
    generator.run()