from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import re

import cv2
import numpy as np

from supervisor.picture_generator import PictureGenerator


class SymbolStatsGenerator(PictureGenerator):
    """Cumulates symbol statistics across session directories and generates OpenCV plots with trend lines."""

    LINE_COLOR_PRIMARY = (195, 105, 30)     # Slate Blue / Primary
    LINE_COLOR_SECONDARY = (71, 90, 224)    # Coral Red / Secondary
    SHADOW_COLOR = (245, 155, 80)           # Soft slate blue fill for min-max band
    TREND_COLOR_PRIMARY = (130, 60, 10)     # Darker tone for Primary Trend
    TREND_COLOR_SECONDARY = (40, 50, 180)   # Darker tone for Secondary Trend

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
    def _extract_dim_val(val_str: str) -> float | None:
        """Parses dimension strings like '135 (1)' or '1' to float, handling N/A."""
        clean = val_str.strip().strip('"').strip("'")
        if clean.upper() in {"N/A", "NA", "NONE", "NULL", ""}:
            return None
        match = re.match(r"^([\d.]+)", clean)
        return float(match.group(1)) if match else None

    @staticmethod
    def _extract_float_safely(val_str: str) -> float | None:
        """Parses float strings, ignoring N/A, inf, or empty values."""
        clean = str(val_str).strip().strip('"').strip("'")
        if clean.upper() in {"N/A", "NA", "NONE", "NULL", "", "NAN", "INF", "-INF"}:
            return None
        try:
            val = float(clean)
            return None if math.isnan(val) or math.isinf(val) else val
        except ValueError:
            return None

    def parse_symbol_stats_file(self, file_path: Path) -> list[dict]:
        """Parses symbol_stats_log.txt into structured records."""
        if not file_path.exists():
            return []

        records = []
        with file_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = self._extract_float_safely(row.get("timestamp", ""))
                tot = self._extract_float_safely(row.get("total_symbols", ""))
                if ts is None or tot is None:
                    continue

                records.append({
                    "timestamp": ts,
                    "total_symbols": tot,
                    "total_skills": self._extract_float_safely(row.get("total_skills", 0)),
                    "min_dims": self._extract_dim_val(row.get("min_dims", 0)),
                    "dominant_dims": self._extract_dim_val(row.get("dominant_dims", 0)),
                    "max_dims": self._extract_dim_val(row.get("max_dims", "")),
                    "avg_pre_symbols": self._extract_float_safely(row.get("avg_pre_symbols", 0)),
                    "avg_eff_symbols": self._extract_float_safely(row.get("avg_eff_symbols", 0)),
                    "reusability_ratio": self._extract_float_safely(row.get("reusability_ratio", "")),
                    "raw_row": row,
                })
        return records

    def cumulate_stats(self, chain_dirs: list[Path], target_dir: Path) -> list[dict]:
        """Collects symbol statistics from chain. If a preceding folder has cumulated_symbol_stats.txt,
        merges it and stops traversing backwards."""
        out_path = target_dir / "cumulated_symbol_stats.txt"

        if out_path.exists():
            print(f"[SymbolStatsGenerator] cumulated_symbol_stats.txt already exists at {out_path}, skipping.")
            return self.parse_symbol_stats_file(out_path)

        cumulated: list[dict] = []
        fieldnames = [
            "timestamp", "total_symbols", "total_skills",
            "min_dims", "dominant_dims", "max_dims",
            "avg_pre_symbols", "avg_eff_symbols", "reusability_ratio"
        ]

        # chain_dirs are chronological: [oldest, ..., previous, current]
        # Iterate from newest (current) backwards to oldest
        for s_dir in reversed(chain_dirs):
            # If a preceding directory already has the cumulated file, merge it and halt chain traversal
            if s_dir.resolve() != target_dir.resolve():
                prev_cumulative = s_dir / "cumulated_symbol_stats.txt"
                if prev_cumulative.exists():
                    print(f"[SymbolStatsGenerator] Found preceding cumulative file at {prev_cumulative}. Merging.")
                    base_records = self.parse_symbol_stats_file(prev_cumulative)
                    cumulated.extend(base_records)
                    break

            # Otherwise, read this directory's raw log
            for fname in ("symbol_stats_log.txt", "symbol_stats.txt"):
                fpath = s_dir / fname
                if fpath.exists():
                    records = self.parse_symbol_stats_file(fpath)
                    cumulated.extend(records)
                    break

        cumulated.sort(key=lambda x: x["timestamp"])

        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in cumulated:
                writer.writerow({k: r["raw_row"].get(k, r.get(k, "")) for k in fieldnames})

        return cumulated

    @staticmethod
    def _collapse_time_gaps(timestamps: list[float], gap_threshold_factor: float = 5.0) -> list[float]:
        """Compresses idle gaps (e.g., > 10 min without data) into the mean delta time for plot rendering."""
        if len(timestamps) <= 2:
            return timestamps[:]

        deltas = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        positive_deltas = [d for d in deltas if d > 0]
        if not positive_deltas:
            return timestamps[:]

        mean_delta = float(np.mean(positive_deltas))
        # Gaps larger than threshold factor * mean_delta (or >= 600s/10m) get clipped
        gap_limit = max(mean_delta * gap_threshold_factor, 600.0)

        compressed_times = [0.0]
        for d in deltas:
            effective_dt = mean_delta if d > gap_limit else max(d, 0.0)
            compressed_times.append(compressed_times[-1] + effective_dt)

        return compressed_times

    @staticmethod
    def _calculate_moving_average(y_vals: list[float], window_size: int = 100) -> list[float]:
        """Calculates moving average with dynamic edge handling for leading points."""
        if not y_vals:
            return []
        ma = []
        for i in range(len(y_vals)):
            start_idx = max(0, i - window_size + 1)
            window = y_vals[start_idx : i + 1]
            ma.append(float(np.mean(window)))
        return ma

    def _draw_trend_line(
        self,
        img: np.ndarray,
        x_vals: list[float],
        y_vals: list[float],
        get_x: callable,
        get_y: callable,
        color: tuple[int, int, int],
        thickness: int = 2,
        window_size: int = 1000,
    ):
        """Computes and renders a moving average trend curve over time."""
        if len(x_vals) < 2 or len(x_vals) != len(y_vals):
            return
        
        ma_vals = self._calculate_moving_average(y_vals, window_size=window_size)
        pts = [(get_x(x), get_y(y)) for x, y in zip(x_vals, ma_vals)]
        
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)

    def _setup_plot_area(
        self,
        title: str,
        x_vals: list[float],
        y_min: float,
        y_max: float,
        y_label: str,
        width: int = 1000,
        height: int = 620,
    ) -> tuple[np.ndarray, int, int, int, int, callable, callable]:
        """Creates canvas, axis bounding boxes, gridlines, and coordinate mappers."""
        img = self.renderer.create_base_canvas(title.upper(), width=width, height=height)

        plot_x, plot_y = 90, 110
        plot_w, plot_h = 710, 430
        cv2.rectangle(img, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), self.renderer.PANEL_BORDER, 1)

        t_min = min(x_vals) if x_vals else 0.0
        t_max = max(x_vals) if x_vals else 1.0
        t_span = max(t_max - t_min, 1e-6)

        y_range = max(y_max - y_min, 1e-6)
        margin = y_range * 0.08
        plot_y_min = y_min - margin
        plot_y_max = y_max + margin
        plot_y_span = plot_y_max - plot_y_min

        # Horizontal Gridlines & Y Ticks
        grid_steps = 4
        for i in range(grid_steps + 1):
            y_pixel = plot_y + plot_h - int(i * (plot_h / grid_steps))
            val = plot_y_min + i * (plot_y_span / grid_steps)
            cv2.line(img, (plot_x, y_pixel), (plot_x + plot_w, y_pixel), self.renderer.tertiary_color, 1, cv2.LINE_AA)
            cv2.putText(
                img,
                f"{val:.2f}" if plot_y_span < 10 else f"{int(round(val))}",
                (plot_x - 55, y_pixel + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                self.renderer.neutral_gray,
                1,
                cv2.LINE_AA,
            )

        # X Ticks
        x_steps = 5
        for i in range(x_steps + 1):
            x_pixel = plot_x + int(i * (plot_w / x_steps))
            t_val = i * (t_span / x_steps)
            cv2.line(img, (x_pixel, plot_y + plot_h), (x_pixel, plot_y + plot_h + 5), self.renderer.BORDER_COLOR, 1, cv2.LINE_AA)
            cv2.putText(
                img,
                f"+{int(t_val)}s",
                (x_pixel - 15, plot_y + plot_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                self.renderer.TEXT_MAIN,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(img, "Time Elapsed", (plot_x + plot_w // 2 - 40, plot_y + plot_h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.putText(img, y_label, (plot_x - 30, plot_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

        def get_x_px(val: float) -> int:
            return plot_x + int(((val - t_min) / t_span) * plot_w)

        def get_y_px(val: float) -> int:
            norm = (val - plot_y_min) / plot_y_span
            return plot_y + plot_h - int(np.clip(norm, 0.0, 1.0) * plot_h)

        return img, plot_x, plot_y, plot_w, plot_h, get_x_px, get_y_px

    def render_total_symbols_plot(self, timestamps: list[float], total_symbols: list[float]) -> np.ndarray:
        """1) Total symbols in time: separated scatter points + linear trend line."""
        y_min = min(total_symbols) if total_symbols else 0.0
        y_max = max(total_symbols) if total_symbols else 1.0

        img, px, py, pw, ph, get_x, get_y = self._setup_plot_area(
            "Total Number of Symbols", timestamps, y_min, y_max, "Symbols (count)"
        )

        # Draw trend line
        self._draw_trend_line(img, timestamps, total_symbols, get_x, get_y, self.TREND_COLOR_PRIMARY, thickness=2)

        # Draw separated points
        for t, v in zip(timestamps, total_symbols):
            cv2.circle(img, (get_x(t), get_y(v)), 1, self.LINE_COLOR_PRIMARY, -1, cv2.LINE_AA)

        # Legend
        leg_x, leg_y = px + pw + 15, py + 25
        cv2.circle(img, (leg_x + 6, leg_y - 2), 4, self.LINE_COLOR_PRIMARY, -1, cv2.LINE_AA)
        cv2.putText(img, "Symbols Count (points)", (leg_x + 18, leg_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

        cv2.line(img, (leg_x, leg_y + 20), (leg_x + 14, leg_y + 20), self.TREND_COLOR_PRIMARY, 2, cv2.LINE_AA)
        cv2.putText(img, "Moving Average", (leg_x + 18, leg_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

        self.renderer.draw_sample_badge(img, count=len(timestamps), label="Data Points", position=(px, 595))
        return img

    def render_dimensionality_plot(
        self,
        timestamps: list[float],
        min_dims: list[float],
        dominant_dims: list[float],
        max_dims: list[float],
    ) -> np.ndarray:
        """2) Dimensionality in time: min-max shadow, dominant separated points + dominant trend line."""
        all_vals = min_dims + dominant_dims + max_dims
        y_min = min(all_vals) if all_vals else 0.0
        y_max = max(all_vals) if all_vals else 1.0

        img, px, py, pw, ph, get_x, get_y = self._setup_plot_area(
            "Dimensionality of Symbols", timestamps, y_min, y_max, "Dimensions"
        )

        # Min-max shadow band
        upper_pts = [(get_x(t), get_y(v)) for t, v in zip(timestamps, max_dims)]
        lower_pts = [(get_x(t), get_y(v)) for t, v in zip(timestamps, min_dims)]
        poly_pts = np.array(upper_pts + lower_pts[::-1], dtype=np.int32)

        overlay = img.copy()
        cv2.fillPoly(overlay, [poly_pts], self.SHADOW_COLOR)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

        for i in range(len(upper_pts) - 1):
            cv2.line(img, upper_pts[i], upper_pts[i + 1], self.renderer.PANEL_BORDER, 1, cv2.LINE_AA)
            cv2.line(img, lower_pts[i], lower_pts[i + 1], self.renderer.PANEL_BORDER, 1, cv2.LINE_AA)

        # Dominant trend line
        self._draw_trend_line(img, timestamps, dominant_dims, get_x, get_y, self.TREND_COLOR_PRIMARY, thickness=2)

        # Dominant separated points
        for t, v in zip(timestamps, dominant_dims):
            cv2.circle(img, (get_x(t), get_y(v)), 1, self.LINE_COLOR_PRIMARY, -1, cv2.LINE_AA)

        # Legend
        leg_x, leg_y = px + pw + 15, py + 25
        cv2.circle(img, (leg_x + 6, leg_y - 2), 4, self.LINE_COLOR_PRIMARY, -1, cv2.LINE_AA)
        cv2.putText(img, "Dominant", (leg_x + 18, leg_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

        cv2.line(img, (leg_x, leg_y + 20), (leg_x + 14, leg_y + 20), self.TREND_COLOR_PRIMARY, 2, cv2.LINE_AA)
        cv2.putText(img, "Moving Average (dominant)", (leg_x + 18, leg_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

        cv2.rectangle(img, (leg_x, leg_y + 36), (leg_x + 12, leg_y + 48), self.SHADOW_COLOR, -1)
        cv2.putText(img, "Min - Max", (leg_x + 18, leg_y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

        self.renderer.draw_sample_badge(img, count=len(timestamps), label="Data Points", position=(px, 595))
        return img

    def render_avg_symbols_plot(
        self,
        pre_timestamps: list[float],
        avg_pre: list[float],
        eff_timestamps: list[float],
        avg_eff: list[float],
    ) -> np.ndarray:
        """3) Avg symbols in time: separated points and trend lines for precondition and effect."""
        all_vals = avg_pre + avg_eff
        y_min = min(all_vals) if all_vals else 0.0
        y_max = max(all_vals) if all_vals else 1.0

        all_ts = pre_timestamps + eff_timestamps
        img, px, py, pw, ph, get_x, get_y = self._setup_plot_area(
            "Average Number of Symbols in Predicate", all_ts, y_min, y_max, "Symbol Count"
        )

        # Trend lines
        self._draw_trend_line(img, pre_timestamps, avg_pre, get_x, get_y, self.TREND_COLOR_PRIMARY, thickness=2)
        self._draw_trend_line(img, eff_timestamps, avg_eff, get_x, get_y, self.TREND_COLOR_SECONDARY, thickness=2)

        # Separated points
        for t, v in zip(pre_timestamps, avg_pre):
            cv2.circle(img, (get_x(t), get_y(v)), 1, self.LINE_COLOR_PRIMARY, -1, cv2.LINE_AA)
        for t, v in zip(eff_timestamps, avg_eff):
            cv2.circle(img, (get_x(t), get_y(v)), 1, self.LINE_COLOR_SECONDARY, -1, cv2.LINE_AA)

        # Legend
        leg_x, leg_y = px + pw + 15, py + 25
        cv2.circle(img, (leg_x + 6, leg_y - 2), 4, self.LINE_COLOR_PRIMARY, -1, cv2.LINE_AA)
        cv2.putText(img, "Precondition (points)", (leg_x + 18, leg_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

        cv2.line(img, (leg_x, leg_y + 18), (leg_x + 14, leg_y + 18), self.TREND_COLOR_PRIMARY, 2, cv2.LINE_AA)
        cv2.putText(img, "Precondition (moving average)", (leg_x + 18, leg_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

        cv2.circle(img, (leg_x + 6, leg_y + 38), 4, self.LINE_COLOR_SECONDARY, -1, cv2.LINE_AA)
        cv2.putText(img, "Effect (points)", (leg_x + 18, leg_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

        cv2.line(img, (leg_x, leg_y + 58), (leg_x + 14, leg_y + 58), self.TREND_COLOR_SECONDARY, 2, cv2.LINE_AA)
        cv2.putText(img, "Effect (moving average)", (leg_x + 18, leg_y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

        self.renderer.draw_sample_badge(img, count=max(len(pre_timestamps), len(eff_timestamps)), label="Data Points", position=(px, 595))
        return img

    def render_reusability_plot(self, timestamps: list[float], reusability_ratios: list[float]) -> np.ndarray:
        """4) Reusability ratio in time: filter N/A points, separated points + trend line."""
        y_min = min(reusability_ratios) if reusability_ratios else 0.0
        y_max = max(reusability_ratios) if reusability_ratios else 1.0

        img, px, py, pw, ph, get_x, get_y = self._setup_plot_area(
            "Reusability Ratio", timestamps, y_min, y_max, "Ratio"
        )

        # Trend line
        self._draw_trend_line(img, timestamps, reusability_ratios, get_x, get_y, self.TREND_COLOR_PRIMARY, thickness=2)

        # Separated points
        for t, v in zip(timestamps, reusability_ratios):
            cv2.circle(img, (get_x(t), get_y(v)), 1, self.LINE_COLOR_PRIMARY, -1, cv2.LINE_AA)

        # Legend
        leg_x, leg_y = px + pw + 15, py + 25
        cv2.circle(img, (leg_x + 6, leg_y - 2), 4, self.LINE_COLOR_PRIMARY, -1, cv2.LINE_AA)
        cv2.putText(img, "Ratio (points)", (leg_x + 18, leg_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

        cv2.line(img, (leg_x, leg_y + 20), (leg_x + 14, leg_y + 20), self.TREND_COLOR_PRIMARY, 2, cv2.LINE_AA)
        cv2.putText(img, "Moving Average", (leg_x + 18, leg_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

        self.renderer.draw_sample_badge(img, count=len(timestamps), label="Valid Samples", position=(px, 595))
        return img

    def run(self):
        if not self.session_path or not os.path.exists(self.session_path):
            return

        try:
            current_dir = Path(self.session_path)
            logs_root = Path(self.logs_root)
            chain_dirs = self.collect_chain_folders(current_dir, logs_root)

            cumulated = self.cumulate_stats(chain_dirs, current_dir)
            if not cumulated:
                print("[SymbolStatsGenerator] No symbol statistics found across session dirs.")
                return

            # Compress large gaps (>= 10 min or > 5x mean delta) for plot display only
            raw_ts = [r["timestamp"] for r in cumulated]
            plot_ts = self._collapse_time_gaps(raw_ts)
            for r, pts in zip(cumulated, plot_ts):
                r["plot_timestamp"] = pts

            tot_all = [r["total_symbols"] for r in cumulated]
            out_dir = self.get_pictures_dir() or str(current_dir)

            # 1. Total symbols trend
            img_symbols = self.render_total_symbols_plot(plot_ts, tot_all)
            cv2.imwrite(os.path.join(out_dir, "symbols_total_trend.png"), img_symbols)

            # 2. Dimensionality (min, dominant, max)
            dim_valid = [r for r in cumulated if r["dominant_dims"] is not None and r["min_dims"] is not None and r["max_dims"] is not None]
            if dim_valid:
                img_dims = self.render_dimensionality_plot(
                    [r["plot_timestamp"] for r in dim_valid],
                    [r["min_dims"] for r in dim_valid],
                    [r["dominant_dims"] for r in dim_valid],
                    [r["max_dims"] for r in dim_valid],
                )
                cv2.imwrite(os.path.join(out_dir, "symbols_dimensionality_trend.png"), img_dims)

            # 3. Precondition vs Effect average symbols
            pre_valid = [r for r in cumulated if r["avg_pre_symbols"] is not None]
            eff_valid = [r for r in cumulated if r["avg_eff_symbols"] is not None]
            if pre_valid or eff_valid:
                img_avg = self.render_avg_symbols_plot(
                    [r["plot_timestamp"] for r in pre_valid],
                    [r["avg_pre_symbols"] for r in pre_valid],
                    [r["plot_timestamp"] for r in eff_valid],
                    [r["avg_eff_symbols"] for r in eff_valid],
                )
                cv2.imwrite(os.path.join(out_dir, "symbols_pre_eff_avg_trend.png"), img_avg)

            # 4. Reusability ratio (guarded against N/A)
            ratio_valid = [r for r in cumulated if r["reusability_ratio"] is not None]
            if ratio_valid:
                img_reusability = self.render_reusability_plot(
                    [r["plot_timestamp"] for r in ratio_valid],
                    [r["reusability_ratio"] for r in ratio_valid],
                )
                cv2.imwrite(os.path.join(out_dir, "symbols_reusability_trend.png"), img_reusability)

            print(f"[SymbolStatsGenerator] Saved cumulated symbol statistics and plots to {out_dir}")
        except Exception as e:
            print(f"[SymbolStatsGenerator] Failed to generate symbol statistics plots: {e}")


if __name__ == "__main__":
    generator = SymbolStatsGenerator()
    generator.run()