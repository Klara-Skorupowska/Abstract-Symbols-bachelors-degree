from __future__ import annotations

import os
from pathlib import Path
import re
from datetime import datetime

import cv2
import numpy as np
from scipy.stats import gaussian_kde

from supervisor.picture_generator import PictureGenerator


class HistogramPlotGenerator(PictureGenerator):
    """Generates continuous Gaussian KDE frequency plots using OpenCV and Renderer style."""

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

    @staticmethod
    def parse_timestamp_line(line: str) -> float | None:
        line = line.strip()
        if not line:
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

    def load_session_timestamps(self, session_dir: Path) -> np.ndarray:
        ts_file = session_dir / "timestamps.txt"
        if not ts_file.exists():
            return np.array([], dtype=float)

        timestamps: list[float] = []
        with ts_file.open("r", encoding="utf-8") as f:
            for line in f:
                val = self.parse_timestamp_line(line)
                if val is not None:
                    timestamps.append(val)

        return np.sort(np.array(timestamps, dtype=float)) if timestamps else np.array([], dtype=float)

    def build_and_glue_normalized_timeline(
        self,
        chain_dirs: list[Path],
        intra_session_max_gap_sec: float = 3600.0,
        stitch_gap_sec: float = 60.0,
    ) -> tuple[np.ndarray, list[dict]]:
        raw_segments: list[dict] = []

        for s_idx, s_dir in enumerate(chain_dirs):
            ts = self.load_session_timestamps(s_dir)
            if len(ts) < 2:
                continue

            session_num = s_idx + 1
            dt = np.diff(ts, prepend=ts[0])
            gap_indices = np.where(dt > intra_session_max_gap_sec)[0]
            split_points = [0] + list(gap_indices) + [len(ts)]

            num_splits = len(split_points) - 1
            for sub_i in range(num_splits):
                sub_ts = ts[split_points[sub_i] : split_points[sub_i + 1]]
                if len(sub_ts) >= 2:
                    session_tag = f"S{session_num}" if sub_i == 0 else None
                    subsession_tag = f"S{session_num}.{sub_i + 1}" if num_splits > 1 else None
                    raw_segments.append({
                        "session_tag": session_tag,
                        "subsession_tag": subsession_tag,
                        "timestamps": sub_ts,
                    })

        if not raw_segments:
            return np.array([], dtype=float), []

        glued_times: list[np.ndarray] = []
        segment_boundaries: list[dict] = []
        current_offset_sec = 0.0

        for seg in raw_segments:
            rel_ts = seg["timestamps"] - seg["timestamps"][0]
            stitched_part = current_offset_sec + rel_ts
            glued_times.append(stitched_part)

            segment_boundaries.append({
                "session_tag": seg["session_tag"],
                "subsession_tag": seg["subsession_tag"],
                "start_min": stitched_part[0] / 60.0,
                "end_min": stitched_part[-1] / 60.0,
            })
            current_offset_sec = stitched_part[-1] + stitch_gap_sec

        return np.concatenate(glued_times) / 60.0, segment_boundaries

    def draw_kde_canvas(self, timeline_min: np.ndarray, width: int = 1000, height: int = 620) -> np.ndarray:
        img = self.renderer.create_base_canvas("BRAIN UPDATES FREQUENCY (KDE)", width=width, height=height)

        if len(timeline_min) < 10:
            cv2.putText(
                img,
                "Insufficient data points for KDE estimation.",
                (320, 310),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                self.renderer.neutral_gray,
                1,
                cv2.LINE_AA,
            )
            return img

        bw = "scott" if np.std(timeline_min) > 1e-5 else 0.5
        kde = gaussian_kde(timeline_min, bw_method=bw)

        t_min, t_max = timeline_min[0], timeline_min[-1]
        x_margin = max((t_max - t_min) * 0.02, 1.0)
        x_start, x_end = max(0.0, t_min - x_margin), t_max + x_margin

        plot_x, plot_y = 90, 110
        plot_w, plot_h = 860, 420
        cv2.rectangle(img, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), self.renderer.PANEL_BORDER, 1)

        x_grid = np.linspace(x_start, x_end, 600)
        density = kde(x_grid) * len(timeline_min)
        max_density = max(float(np.max(density)) * 1.15, 1e-4)

        # -------------------------------------------------------------
        # 1. Full Grid: Horizontal Gridlines & Y-axis labels
        # -------------------------------------------------------------
        grid_steps_y = 4
        for i in range(grid_steps_y + 1):
            y_val = plot_y + plot_h - int(i * (plot_h / grid_steps_y))
            # Horizontal grid line spanning full plot width
            cv2.line(img, (plot_x, y_val), (plot_x + plot_w, y_val), self.renderer.tertiary_color, 1, cv2.LINE_AA)
            val_label = f"{(i * (max_density / grid_steps_y)):.2f}"
            cv2.putText(
                img,
                val_label,
                (plot_x - 55, y_val + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                self.renderer.neutral_gray,
                1,
                cv2.LINE_AA,
            )

        # -------------------------------------------------------------
        # 2. Full Grid: Vertical Gridlines & X-axis minute labels
        # -------------------------------------------------------------
        time_span = x_end - x_start
        step = max(1.0, round(time_span / 6.0))
        first_tick = np.ceil(x_start / step) * step
        tick_vals = np.arange(first_tick, x_end, step)

        for t_val in tick_vals:
            tx = plot_x + int(((t_val - x_start) / (x_end - x_start)) * plot_w)
            # Vertical grid line spanning full plot height
            cv2.line(img, (tx, plot_y), (tx, plot_y + plot_h), self.renderer.tertiary_color, 1, cv2.LINE_AA)
            # Tick mark below axis
            cv2.line(img, (tx, plot_y + plot_h), (tx, plot_y + plot_h + 5), self.renderer.BORDER_COLOR, 1, cv2.LINE_AA)
            cv2.putText(
                img,
                f"{int(t_val)}m",
                (tx - 12, plot_y + plot_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                self.renderer.TEXT_MAIN,
                1,
                cv2.LINE_AA,
            )

        # -------------------------------------------------------------
        # 3. KDE Density Curve & Fill
        # -------------------------------------------------------------
        pts = []
        for x_val, dens in zip(x_grid, density):
            px = plot_x + int(((x_val - x_start) / (x_end - x_start)) * plot_w)
            py = plot_y + plot_h - int((dens / max_density) * plot_h)
            pts.append((px, py))

        fill_pts = np.array([(plot_x, plot_y + plot_h)] + pts + [(plot_x + plot_w, plot_y + plot_h)], dtype=np.int32)

        # Draw semi-transparent fill using addWeighted to keep grid lines subtly visible underneath
        overlay = img.copy()
        cv2.fillPoly(overlay, [fill_pts], self.renderer.secondary_color)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        # KDE Top Contour Line
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], isClosed=False, color=self.renderer.BORDER_COLOR, thickness=2, lineType=cv2.LINE_AA)

        # Re-stroke plot outer boundary
        cv2.rectangle(img, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), self.renderer.PANEL_BORDER, 1)

        # -------------------------------------------------------------
        # 4. Axis Titles & Badge
        # -------------------------------------------------------------
        cv2.putText(
            img,
            "Continuous Time [minutes]",
            (plot_x + plot_w // 2 - 90, plot_y + plot_h + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            self.renderer.TEXT_MAIN,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "Density (events/min)",
            (plot_x - 30, plot_y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            self.renderer.TEXT_MAIN,
            1,
            cv2.LINE_AA,
        )

        self.renderer.draw_sample_badge(img, count=len(timeline_min), label="Data Points", position=(plot_x, 595))
        return img

    def _resolve_file_path(self) -> Path | None:
        """Resolves the session chain starting from timestamps.txt in the target session."""
        if not self.session_path or not os.path.exists(self.session_path):
            return None
        target = Path(self.session_path) / "timestamps.txt"
        return target if target.exists() else None

    def run(self):
        file_path = self._resolve_file_path()
        if not file_path:
            return

        try:
            current_dir = Path(self.session_path)
            logs_root = Path(self.logs_root)
            chain_dirs = self.collect_chain_folders(current_dir, logs_root)

            unified_timeline_min, _ = self.build_and_glue_normalized_timeline(chain_dirs)
            img = self.draw_kde_canvas(unified_timeline_min)

            out_dir = self.get_pictures_dir() or "."
            out_path = os.path.join(out_dir, "brain_updates_frequency_kde.png")
            cv2.imwrite(out_path, img)
            print(f"[HistogramPlotGenerator] Saved KDE plot to {out_path}")
        except Exception as e:
            print(f"[HistogramPlotGenerator] Failed to generate KDE plot: {e}")


if __name__ == "__main__":
    generator = HistogramPlotGenerator()
    generator.run()