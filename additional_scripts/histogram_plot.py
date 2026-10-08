from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def get_newest_folder(logs_dir: Path) -> Path:
    """Finds the most recently modified directory in the given path."""
    folders = [p for p in logs_dir.iterdir() if p.is_dir()]
    if not folders:
        raise FileNotFoundError(f"No subdirectories found in: {logs_dir}")
    return max(folders, key=lambda p: p.stat().st_mtime)


def parse_continuation(stats_file: Path) -> Path | None:
    """Parses the previous session path from execution_statistics.txt."""
    if not stats_file.exists():
        return None

    pattern = re.compile(r"CONTINUATION FROM SESSION:\s*(.+)", re.IGNORECASE)
    with stats_file.open("r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line.strip())
            if match:
                raw_path = match.group(1).strip()
                return Path(raw_path)
    return None


def parse_timestamp_line(line: str) -> float | None:
    """Parses a timestamp line to a float representing Unix epoch seconds."""
    line = line.strip()
    if not line:
        return None
    try:
        return float(line)
    except ValueError:
        pass

    try:
        dt = datetime.fromisoformat(line)
        return dt.timestamp()
    except ValueError:
        return None


def collect_chain_folders(newest_dir: Path, logs_root: Path) -> list[Path]:
    """Traverses backwards through the session chain using execution_statistics.txt."""
    current_dir = newest_dir
    visited_dirs = {current_dir.resolve()}
    chain_dirs: list[Path] = [current_dir]

    while True:
        stats_file = current_dir / "execution_statistics.txt"
        continued_rel_path = parse_continuation(stats_file)

        if not continued_rel_path:
            break

        if (logs_root.parent / continued_rel_path).exists():
            next_dir = logs_root.parent / continued_rel_path
        elif (logs_root / continued_rel_path.name).exists():
            next_dir = logs_root / continued_rel_path.name
        else:
            print(f"Warning: Continuation folder '{continued_rel_path}' not found. Stopping traversal.")
            break

        next_resolved = next_dir.resolve()
        if next_resolved in visited_dirs:
            print(f"Loop detected at {next_dir}. Stopping traversal.")
            break

        visited_dirs.add(next_resolved)
        chain_dirs.append(next_dir)
        current_dir = next_dir

    chain_dirs.reverse()
    return chain_dirs


def load_session_timestamps(session_dir: Path) -> np.ndarray:
    """Loads timestamps from a single session folder."""
    ts_file = session_dir / "timestamps.txt"
    if not ts_file.exists():
        return np.array([], dtype=float)

    timestamps: list[float] = []
    with ts_file.open("r", encoding="utf-8") as f:
        for line in f:
            val = parse_timestamp_line(line)
            if val is not None:
                timestamps.append(val)

    if not timestamps:
        return np.array([], dtype=float)

    return np.sort(np.array(timestamps, dtype=float))


def build_and_glue_normalized_timeline(
    chain_dirs: list[Path],
    intra_session_max_gap_sec: float = 3600.0,
    stitch_gap_sec: float = 60.0,
) -> tuple[np.ndarray, list[dict]]:
    """
    1. Splits sessions into sub-sessions if an internal gap > intra_session_max_gap_sec.
    2. Glues segments together seamlessly with stitch_gap_sec spacing.
    3. Retains main session tag (e.g. S4) and nested sub-session tags (e.g. S4.1, S4.2).
    """
    raw_segments: list[dict] = []

    for s_idx, s_dir in enumerate(chain_dirs):
        ts = load_session_timestamps(s_dir)
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

        start_min = stitched_part[0] / 60.0
        end_min = stitched_part[-1] / 60.0

        segment_boundaries.append({
            "session_tag": seg["session_tag"],
            "subsession_tag": seg["subsession_tag"],
            "start_min": start_min,
            "end_min": end_min,
        })

        current_offset_sec = stitched_part[-1] + stitch_gap_sec

    unified_timeline_min = np.concatenate(glued_times) / 60.0
    return unified_timeline_min, segment_boundaries


def resolve_safe_output_path(target_path: Path) -> Path | None:
    """Prompt the user before overwriting an existing file, with options to abort, overwrite, or rename."""
    if not target_path.exists():
        return target_path

    print(f"\n[WARNING] Output file already exists: {target_path}")
    while True:
        try:
            choice = input("Overwrite existing file? ([y]es / [n]o / [a]uto-rename): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nOperation cancelled by user. No file written.")
            return None

        if choice in ("y", "yes"):
            return target_path
        elif choice in ("a", "auto", "rename"):
            counter = 1
            stem = target_path.stem
            ext = target_path.suffix
            while True:
                candidate = target_path.with_name(f"{stem}_{counter}{ext}")
                if not candidate.exists():
                    print(f"Saving to new file name: {candidate.name}")
                    return candidate
                counter += 1
        elif choice in ("n", "no"):
            print("Operation cancelled by user. No file written.")
            return None
        else:
            print("Invalid input. Please enter 'y', 'n', or 'a'.")


def generate_glued_kde_plot(
    timeline_min: np.ndarray,
    boundaries: list[dict],
    output_image_path: Path,
) -> None:
    """Generates a KDE plot matching the Renderer visual design system."""
    if len(timeline_min) < 10:
        print("Insufficient data points for KDE estimation.")
        return

    # Palette aligned with Renderer (BGR translated to RGB Hex)
    primary_color = "#1E69C3"       # Slate Blue
    secondary_color = "#5AB9EB"     # Light Accent Blue
    tertiary_color = "#D7EBF8"      # Ice Blue Badge Fill
    dark_border = "#282C34"         # Frame & Dark Accents
    text_main = "#1E1E1E"           # Main Headings / Text
    panel_border = "#D2D7DC"        # Axis frame
    grid_color = "#EBF4FA"          # Subtle Horizontal Rule
    bg_color = "#FAF9F8"

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    fig, ax = plt.subplots(figsize=(8.8, 5.4), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # KDE Density Estimation
    bw = "scott" if np.std(timeline_min) > 1e-5 else 0.5
    kde = gaussian_kde(timeline_min, bw_method=bw)

    x_grid = np.linspace(timeline_min[0], timeline_min[-1], 2000)
    density = kde(x_grid) * len(timeline_min)

    # Plot line & fill
    ax.fill_between(x_grid, density, color=secondary_color, alpha=0.85, zorder=2)
    ax.plot(x_grid, density, color=dark_border, lw=2.0, zorder=3, label="KDE Density")

    y_max = ax.get_ylim()[1]

    # Boundaries
    for idx, b in enumerate(boundaries):
        start_m = b["start_min"]
        session_tag = b["session_tag"]
        subsession_tag = b["subsession_tag"]

        if session_tag is not None:
            if idx > 0:
                ax.axvline(start_m, color=dark_border, linestyle="--", lw=1.2, alpha=0.75, zorder=4)
            ax.text(
                start_m,
                y_max * 0.95,
                f" {session_tag}",
                color=text_main,
                fontsize=9.5,
                fontweight="bold",
                rotation=90,
                verticalalignment="top",
                zorder=5,
            )

        if subsession_tag is not None:
            ax.axvline(start_m, color="#82878C", linestyle=":", lw=1.0, alpha=0.8, zorder=4)
            ax.text(
                start_m,
                y_max * 0.75,
                f" {subsession_tag}",
                color="#555555",
                fontsize=8.5,
                fontweight="normal",
                rotation=90,
                verticalalignment="top",
                zorder=5,
            )

    # Axis Labels & Ticks
    ax.set_xlabel("Effective Continuous Time [minutes]", fontsize=10.5, color=text_main, labelpad=8)
    ax.set_ylabel("Update Density\n(events / min)", fontsize=10.5, color=text_main, labelpad=8)
    ax.tick_params(colors="#82878C", labelsize=9.5, length=0)

    # Spines & Horizontal Gridlines
    for spine in ax.spines.values():
        spine.set_color(panel_border)
        spine.set_linewidth(1.2)
    ax.grid(axis="y", color=grid_color, linestyle="-", linewidth=1.2, zorder=1)
    ax.grid(axis="x", visible=False)

    margin = max((timeline_min[-1] - timeline_min[0]) * 0.02, 1.0)
    ax.set_xlim(timeline_min[0] - margin, timeline_min[-1] + margin)
    ax.set_ylim(bottom=0)

    # Legend
    legend = ax.legend(loc="lower right", frameon=True, fontsize=9.5)
    legend.get_frame().set_edgecolor(panel_border)
    legend.get_frame().set_facecolor("white")
    for text in legend.get_texts():
        text.set_color(text_main)

    # Top Header
    fig.text(
        0.05,
        0.932,
        "BRAIN UPDATES FREQUENCY (KDE)",
        fontsize=15,
        fontweight="bold",
        color=text_main,
        horizontalalignment="left",
        verticalalignment="bottom",
    )

    # Horizontal divider line beneath header
    fig.add_artist(
        plt.Line2D(
            [0.05, 0.95],
            [0.885, 0.885],
            transform=fig.transFigure,
            color="#EAEAEA",
            linewidth=1.2,
        )
    )

    # Bottom-left sample count badge
    ax.text(
        0.05,
        0.035,
        f"N = {len(timeline_min)} samples",
        transform=fig.transFigure,
        fontsize=9,
        color=text_main,
        bbox=dict(boxstyle="square,pad=0.45", facecolor=tertiary_color, edgecolor=primary_color, linewidth=1.0),
    )

    # Canvas framing outline
    fig.patch.set_linewidth(2.0)
    fig.patch.set_edgecolor(dark_border)

    plt.subplots_adjust(top=0.79, bottom=0.15, left=0.12, right=0.94)
    fig.savefig(output_image_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
    plt.close(fig)


def main():
    logs_root = Path("logs")
    if not logs_root.exists():
        raise FileNotFoundError(f"Directory '{logs_root}' does not exist.")

    # 1. Locate newest session directory
    newest_dir = get_newest_folder(logs_root)
    print(f"Newest session: {newest_dir.name}")

    # 2. Collect session chain
    chain_dirs = collect_chain_folders(newest_dir, logs_root)
    print(f"Chronological sessions to process: {[d.name for d in chain_dirs]}")

    # 3. Stitch timeline together and compute sub-session boundaries
    unified_timeline_min, boundaries = build_and_glue_normalized_timeline(
        chain_dirs=chain_dirs,
        intra_session_max_gap_sec=3600.0,
        stitch_gap_sec=60.0,
    )

    if len(unified_timeline_min) == 0:
        print("No valid timestamps found to process.")
        return

    # 4. Check overwrite safeguard
    target_image_path = newest_dir / "brain_updates_frequency_kde.png"
    safe_output_path = resolve_safe_output_path(target_image_path)
    if safe_output_path is None:
        return

    # 5. Generate plot
    generate_glued_kde_plot(unified_timeline_min, boundaries, safe_output_path)

    print("\n[SUCCESS] Plot generated and saved to:")
    print(f"  -> Relative path: {safe_output_path}")
    print(f"  -> Absolute path: {safe_output_path.resolve()}")


if __name__ == "__main__":
    main()