import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from supervisor.core.picture_generator import PictureGenerator
from supervisor.utils.dir_utils import collect_chain_folders, get_latest_epoch_dir


def _parse_timestamp_line(line: str) -> float | None:
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


def _load_timestamps(ts_file: Path) -> np.ndarray:
    if not ts_file.exists():
        return np.array([], dtype=float)
    with ts_file.open('r', encoding='utf-8') as f:
        timestamps = [val for line in f if (val := _parse_timestamp_line(line)) is not None]
    return np.sort(np.array(timestamps, dtype=float)) if timestamps else np.array([], dtype=float)


def _resolve_session_dir(logs_root: str, session_dir: str | None, session_path: str | None) -> Path | None:
    """Resolves the actual session directory whether passed as a full/relative path or session name."""
    candidates = []
    if session_path:
        candidates.append(Path(session_path))
    if session_dir:
        p = Path(session_dir)
        candidates.append(p)
        if logs_root:
            candidates.append(Path(logs_root) / p.name)
            candidates.append(Path(logs_root) / p)

    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand.resolve()
    return None


class HistogramPlotGenerator(PictureGenerator):
    """
    Renders a Kernel Density Estimation (KDE) plot comparing update density
    across chained epoch directories using Renderer.draw_kde_plot.
    """
    def __init__(self, logs_root='logs', session_dir=None, timestamp_file='timestamps.txt', bw_method=None):
        super().__init__(logs_root, session_dir)
        self.raw_session_dir = session_dir
        self.timestamp_file = timestamp_file
        self.bw_method = bw_method

    def run(self):
        session_dir = _resolve_session_dir(self.logs_root, self.raw_session_dir, self.session_path)
        if session_dir is None:
            print(f"[HistogramPlotGenerator] Session directory not found (logs_root={self.logs_root!r}, session_dir={self.raw_session_dir!r}, session_path={self.session_path!r})")
            return

        self.session_path = str(session_dir)

        try:
            logs_root_path = Path(self.logs_root).resolve() if self.logs_root and os.path.exists(self.logs_root) else session_dir.parent
            chain_dirs = collect_chain_folders(session_dir, logs_root_path)
            if not chain_dirs:
                chain_dirs = [session_dir]

            epoch_datasets = []
            prev_max_ts = -np.inf

            for folder in chain_dirs:
                folder_path = Path(folder)
                ts_file = folder_path / self.timestamp_file
                if not ts_file.exists():
                    latest_ep = get_latest_epoch_dir(str(folder_path))
                    if latest_ep:
                        ts_file = Path(latest_ep) / self.timestamp_file

                ts = _load_timestamps(ts_file)
                if len(ts) == 0:
                    continue

                # Keep only timestamps newly added in this epoch
                new_ts = ts[ts > prev_max_ts]
                prev_max_ts = max(prev_max_ts, float(ts[-1]))

                if len(new_ts) >= 2:
                    epoch_datasets.append((folder_path.name, new_ts))

            if not epoch_datasets:
                print(f"[HistogramPlotGenerator] No valid '{self.timestamp_file}' with >=2 timestamps found in {chain_dirs}")
                return

            img = self.renderer.create_base_canvas(
                "Update Density (KDE)",
                f"Session: {session_dir.name} | Epochs: {len(epoch_datasets)}"
            )
            self.renderer.draw_kde_plot(img, epoch_datasets, bw_method=self.bw_method)

            total_samples = sum(len(ts) for _, ts in epoch_datasets)
            self.renderer.draw_sample_badge(img, count=total_samples, label="Timestamps", position=(35, 595))
            self.renderer.draw_sample_badge(img, count=len(epoch_datasets), label="Epochs", position=(210, 595))

            out_dir = self.get_pictures_dir()
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "update_kde.png")
            cv2.imwrite(out_path, img)
            print(f"[HistogramPlotGenerator] Saved KDE plot to {out_path}")

        except Exception as e:
            print(f"[HistogramPlotGenerator] Error: {e}")


class UpdateFrequencyPictureGenerator(PictureGenerator):
    """
    Renders a binned update frequency histogram with trend line for the active
    session using Renderer.draw_frequency_plot.
    """
    def __init__(self, logs_root='logs', session_dir=None, timestamp_file='timestamps.txt', bin_size_sec=60.0):
        super().__init__(logs_root, session_dir)
        self.raw_session_dir = session_dir
        self.timestamp_file = timestamp_file
        self.bin_size_sec = bin_size_sec

    def run(self):
        session_dir = _resolve_session_dir(self.logs_root, self.raw_session_dir, self.session_path)
        if session_dir is None:
            print(f"[UpdateFrequencyPictureGenerator] Session directory not found (logs_root={self.logs_root!r}, session_dir={self.raw_session_dir!r}, session_path={self.session_path!r})")
            return

        self.session_path = str(session_dir)

        ts_file = session_dir / self.timestamp_file
        if not ts_file.exists():
            latest_ep = get_latest_epoch_dir(str(session_dir))
            if latest_ep and (Path(latest_ep) / self.timestamp_file).exists():
                ts_file = Path(latest_ep) / self.timestamp_file
            else:
                print(f"[UpdateFrequencyPictureGenerator] '{self.timestamp_file}' not found in {session_dir}")
                return

        try:
            timestamps = _load_timestamps(ts_file)
            if len(timestamps) < 2:
                print(f"[UpdateFrequencyPictureGenerator] Insufficient timestamps ({len(timestamps)}) in {ts_file}")
                return

            img = self.renderer.create_base_canvas(
                "Update Frequency Histogram",
                f"Session: {session_dir.name} | Bin Size: {int(self.bin_size_sec)}s"
            )
            self.renderer.draw_frequency_plot(img, timestamps.tolist(), bin_size_sec=self.bin_size_sec)
            self.renderer.draw_sample_badge(img, count=len(timestamps), label="Timestamps", position=(35, 595))

            out_dir = self.get_pictures_dir()
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "update_frequency.png")
            cv2.imwrite(out_path, img)
            print(f"[UpdateFrequencyPictureGenerator] Saved histogram to {out_path}")

        except Exception as e:
            print(f"[UpdateFrequencyPictureGenerator] Error: {e}")


def _load_updatefreq_lines(freq_file: Path) -> list[str]:
    if not freq_file.exists():
        return []
    with freq_file.open('r', encoding='utf-8') as f:
        return [line.strip().upper() for line in f if line.strip()]


def _bin_updatefreq_lines(lines: list[str], bin_size: int = 100) -> list[dict]:
    """
    Groups consecutive non-empty log lines into bins of `bin_size`.
    Each line is expected to contain mode tokens ('EXPLORE' or 'PLAN') and optional 'COLLECTED' tag.
    """
    if not lines:
        return []

    bins = []
    for i in range(0, len(lines), bin_size):
        chunk = lines[i:i + bin_size]
        explore_cnt = 0
        plan_cnt = 0
        collected_cnt = 0

        for entry in chunk:
            tokens = set(entry.split())
            if 'EXPLORE' in tokens:
                explore_cnt += 1
            elif 'PLAN' in tokens:
                plan_cnt += 1

            if 'COLLECTED' in tokens:
                collected_cnt += 1

        bins.append({
            'total': len(chunk),
            'explore': explore_cnt,
            'plan': plan_cnt,
            'collected': collected_cnt,
        })

    return bins


class LogModeDistributionGenerator(PictureGenerator):
    """
    Renders a binned stacked bar + trend plot of updatefreq.txt across all chained
    epoch directories, showing how many logs per N consecutive entries are 'EXPLORE',
    'PLAN', and tagged 'COLLECTED', with vertical markers separating epochs.
    """
    def __init__(self, logs_root='logs', session_dir=None, freq_file='updatefreq.txt', bin_size=100):
        super().__init__(logs_root, session_dir)
        self.raw_session_dir = session_dir
        self.freq_file = freq_file
        self.bin_size = max(int(bin_size), 1)

    def run(self):
        session_dir = _resolve_session_dir(self.logs_root, self.raw_session_dir, self.session_path)
        if session_dir is None:
            print(
                f"[LogModeDistributionGenerator] Session directory not found "
                f"(logs_root={self.logs_root!r}, session_dir={self.raw_session_dir!r}, session_path={self.session_path!r})"
            )
            return

        self.session_path = str(session_dir)

        try:
            logs_root_path = (
                Path(self.logs_root).resolve()
                if self.logs_root and os.path.exists(self.logs_root)
                else session_dir.parent
            )
            chain_dirs = collect_chain_folders(session_dir, logs_root_path)
            if not chain_dirs:
                chain_dirs = [session_dir]

            all_lines: list[str] = []
            epoch_boundaries: list[tuple[int, str, int]] = []  # (start_log_idx, folder_name, new_count)
            prev_raw_lines: list[str] = []

            for folder in chain_dirs:
                folder_path = Path(folder)
                target_file = folder_path / self.freq_file
                if not target_file.exists():
                    latest_ep = get_latest_epoch_dir(str(folder_path))
                    if latest_ep:
                        target_file = Path(latest_ep) / self.freq_file

                raw_lines = _load_updatefreq_lines(target_file)
                if not raw_lines:
                    continue

                # If this epoch's file carried over prior lines cumulatively, keep only newly added lines
                if (
                    prev_raw_lines
                    and len(raw_lines) >= len(prev_raw_lines)
                    and raw_lines[:len(prev_raw_lines)] == prev_raw_lines
                ):
                    new_lines = raw_lines[len(prev_raw_lines):]
                else:
                    new_lines = raw_lines

                prev_raw_lines = raw_lines

                if new_lines:
                    epoch_boundaries.append((len(all_lines), folder_path.name, len(new_lines)))
                    all_lines.extend(new_lines)

            if not all_lines:
                print(f"[LogModeDistributionGenerator] No valid '{self.freq_file}' entries found in {chain_dirs}")
                return

            bins_data = _bin_updatefreq_lines(all_lines, bin_size=self.bin_size)
            total_logs = len(all_lines)
            total_collected = sum(b['collected'] for b in bins_data)

            img = self.renderer.create_base_canvas(
                "Log Mode & Collection Distribution",
                f"Session: {session_dir.name} | Epochs: {len(epoch_boundaries)} | Bin Size: {self.bin_size} logs"
            )
            self.renderer.draw_log_mode_bins(
                img,
                bins_data,
                bin_size=self.bin_size,
                epoch_boundaries=epoch_boundaries
            )

            self.renderer.draw_sample_badge(img, count=total_logs, label="Total Logs", position=(35, 595))
            self.renderer.draw_sample_badge(img, count=total_collected, label="Collected", position=(225, 595))
            self.renderer.draw_sample_badge(img, count=len(epoch_boundaries), label="Epochs", position=(410, 595))

            out_dir = self.get_pictures_dir()
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "log_mode_distribution.png")
            cv2.imwrite(out_path, img)
            print(f"[LogModeDistributionGenerator] Saved plot to {out_path}")

        except Exception as e:
            print(f"[LogModeDistributionGenerator] Error: {e}")