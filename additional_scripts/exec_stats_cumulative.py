from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# Style & Design System (spójny z histogram_plot.py)
# ---------------------------------------------------------
DARK_BORDER = "#282C34"
TEXT_MAIN = "#1E1E1E"
PANEL_BORDER = "#D2D7DC"
GRID_COLOR = "#EBF4FA"
BG_COLOR = "#FAF9F8"
TERTIARY_COLOR = "#D7EBF8"
PRIMARY_COLOR = "#1E69C3"

# Kolory dla Expected Outcome (Stacked: TP zielony na dole, TN czerwony na górze)
COLOR_TP = "#2BA84A"  # Zielony dla True Positive
COLOR_TN = "#E05A47"  # Czerwony dla True Negative

# Paleta kolorów dla poszczególnych skilli (wykresy 1 i 2)
SKILL_COLORS = [
    "#1E69C3",  # Deep Blue
    "#E05A47",  # Coral Red
    "#2BA84A",  # Emerald Green
    "#D97706",  # Amber / Orange
    "#8B5CF6",  # Purple
    "#06B6D4",  # Cyan / Teal
    "#EC4899",  # Pink
    "#64748B",  # Slate
]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]


# ---------------------------------------------------------
# Session Chain & Parsing
# ---------------------------------------------------------
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


def parse_execution_statistics(file_path: Path) -> dict:
    """
    Parses 'Collected Data' and 'Executions' from execution_statistics.txt.
    Returns:
        {
            "collected_data": {skill_name: count},
            "executions": {skill_name: total_executions},
            "tp": {skill_name: tp_count},
            "tn": {skill_name: tn_count}
        }
    """
    if not file_path.exists():
        return {"collected_data": {}, "executions": {}, "tp": {}, "tn": {}}

    collected_data = {}
    executions = {}
    tp = {}
    tn = {}
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
                    val = int(parts[1])
                    collected_data[skill] = val
                except ValueError:
                    pass

            elif mode == "executions" and len(parts) >= 6:
                # Skill Name | True Positive | True Negative | False Positive | False Negative | Total
                skill = parts[0]
                try:
                    both = max(int(parts[1]) + int(parts[2]), 1)
                    tp_val = int(parts[1]) / both
                    tn_val = int(parts[2]) / both
                    total_val = int(parts[5])
                    tp[skill] = tp_val
                    tn[skill] = tn_val
                    executions[skill] = total_val
                except ValueError:
                    pass

    return {
        "collected_data": collected_data,
        "executions": executions,
        "tp": tp,
        "tn": tn,
    }


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


# ---------------------------------------------------------
# Plotting Routines
# ---------------------------------------------------------
def render_trend_plot(
    session_labels: list[str],
    skills_data: dict[str, list[int]],
    title_main: str,
    y_label: str,
    output_path: Path,
) -> None:
    """Creates a styled line plot matching the visual aesthetic of the renderer."""
    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    x_indices = np.arange(len(session_labels))

    for idx, (skill_name, values) in enumerate(skills_data.items()):
        color = SKILL_COLORS[idx % len(SKILL_COLORS)]
        ax.plot(
            x_indices,
            values,
            marker="o",
            markersize=6,
            linewidth=2.2,
            color=color,
            label=skill_name,
            zorder=4,
        )

    ax.set_xticks(x_indices)
    ax.set_xticklabels(session_labels, fontsize=10, fontweight="bold", color=TEXT_MAIN)
    ax.set_xlabel("Session", fontsize=10.5, color=TEXT_MAIN, labelpad=8)
    ax.set_ylabel(y_label, fontsize=10.5, color=TEXT_MAIN, labelpad=8)
    ax.tick_params(colors="#82878C", labelsize=9.5, length=0)

    for spine in ax.spines.values():
        spine.set_color(PANEL_BORDER)
        spine.set_linewidth(1.2)
    ax.grid(axis="y", color=GRID_COLOR, linestyle="-", linewidth=1.2, zorder=1)
    ax.grid(axis="x", visible=False)

    ax.set_xlim(-0.35, len(session_labels) - 0.65 if len(session_labels) > 1 else 0.5)
    ax.set_ylim(bottom=0)

    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=9.0)
    legend.get_frame().set_edgecolor(PANEL_BORDER)
    legend.get_frame().set_facecolor("white")
    for text in legend.get_texts():
        text.set_color(TEXT_MAIN)

    fig.text(
        0.06,
        0.932,
        title_main.upper(),
        fontsize=14.5,
        fontweight="bold",
        color=TEXT_MAIN,
        horizontalalignment="left",
        verticalalignment="bottom",
    )

    fig.add_artist(
        plt.Line2D(
            [0.06, 0.94],
            [0.885, 0.885],
            transform=fig.transFigure,
            color="#EAEAEA",
            linewidth=1.2,
        )
    )

    total_sessions = len(session_labels)
    ax.text(
        0.06,
        0.035,
        f"Sessions: {total_sessions} | Runs / Session: 1000",
        transform=fig.transFigure,
        fontsize=8.5,
        color=TEXT_MAIN,
        bbox=dict(boxstyle="square,pad=0.45", facecolor=TERTIARY_COLOR, edgecolor=PRIMARY_COLOR, linewidth=1.0),
    )

    fig.patch.set_linewidth(2.0)
    fig.patch.set_edgecolor(DARK_BORDER)

    plt.subplots_adjust(top=0.79, bottom=0.15, left=0.10, right=0.78)
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
    plt.close(fig)


def render_stacked_expected_outcome_plots(
    session_labels: list[str],
    all_skills: list[str],
    tp_data: dict[str, list[int]],
    tn_data: dict[str, list[int]],
    title_main: str,
    output_path: Path,
) -> None:
    """
    Renders subplots (one below the other) for each skill, sharing the X-axis.
    Each subplot features stacked bars (TP on bottom, TN on top).
    """
    num_skills = len(all_skills)
    if num_skills == 0:
        return

    # Dynamiczne dopasowanie wysokości figury w zależności od liczby skilli
    fig_height = max(7.0, 1.8 * num_skills + 1.8)
    fig, axes = plt.subplots(
        nrows=num_skills,
        ncols=1,
        sharex=True,
        figsize=(12, fig_height),
        facecolor=BG_COLOR,
    )

    if num_skills == 1:
        axes = [axes]

    x_indices = np.arange(len(session_labels))
    bar_width = 0.45

    for idx, (skill_name, ax) in enumerate(zip(all_skills, axes)):
        ax.set_facecolor(BG_COLOR)

        tp_vals = np.array(tp_data[skill_name])
        tn_vals = np.array(tn_data[skill_name])

        # Stacked bar: TP (zielony) na dole, TN (czerwony) na górze
        ax.bar(
            x_indices,
            tp_vals,
            width=bar_width,
            color=COLOR_TP,
            edgecolor=DARK_BORDER,
            linewidth=0.8,
            label="True Positive" if idx == 0 else "",
            zorder=3,
        )
        ax.bar(
            x_indices,
            tn_vals,
            bottom=tp_vals,
            width=bar_width,
            color=COLOR_TN,
            edgecolor=DARK_BORDER,
            linewidth=0.8,
            label="True Negative" if idx == 0 else "",
            zorder=3,
        )

        # Wygląd osi i obramowanie
        for spine in ax.spines.values():
            spine.set_color(PANEL_BORDER)
            spine.set_linewidth(1.1)

        ax.grid(axis="y", color=GRID_COLOR, linestyle="-", linewidth=1.0, zorder=1)
        ax.grid(axis="x", visible=False)
        ax.tick_params(colors="#82878C", labelsize=9.0, length=0)
        ax.set_ylabel(skill_name, fontsize=9.5, fontweight="bold", color=TEXT_MAIN, labelpad=6)
        ax.set_ylim(bottom=0)

        # Górna granica z marginesem
        max_val = np.max(tp_vals + tn_vals) if len(tp_vals) > 0 else 1
        ax.set_ylim(0, max_val * 1.1)

    # Konfiguracja dolnej wspólnej osi X
    axes[-1].set_xticks(x_indices)
    axes[-1].set_xticklabels(session_labels, fontsize=10, fontweight="bold", color=TEXT_MAIN)
    axes[-1].set_xlabel("Session", fontsize=10.5, color=TEXT_MAIN, labelpad=8)
    axes[-1].set_xlim(-0.5, len(session_labels) - 0.5)

    # Legenda dla TP i TN
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.94, 0.965),
        ncol=2,
        frameon=True,
        fontsize=9.0,
    )
    legend.get_frame().set_edgecolor(PANEL_BORDER)
    legend.get_frame().set_facecolor("white")
    for text in legend.get_texts():
        text.set_color(TEXT_MAIN)

    # Nagłówek (Header)
    fig.text(
        0.06,
        0.96,
        title_main.upper(),
        fontsize=13.5,
        fontweight="bold",
        color=TEXT_MAIN,
        horizontalalignment="left",
        verticalalignment="bottom",
    )

    # Pozioma linia pod nagłówkiem
    fig.add_artist(
        plt.Line2D(
            [0.06, 0.94],
            [0.935, 0.935],
            transform=fig.transFigure,
            color="#EAEAEA",
            linewidth=1.2,
        )
    )

    # Badge w lewym dolnym rogu
    total_sessions = len(session_labels)
    fig.text(
        0.06,
        0.02,
        f"Sessions: {total_sessions} | Runs / Session: 1000 | Stacked: TP + TN",
        fontsize=8.5,
        color=TEXT_MAIN,
        bbox=dict(boxstyle="square,pad=0.45", facecolor=TERTIARY_COLOR, edgecolor=PRIMARY_COLOR, linewidth=1.0),
    )

    # Ramka canvas
    fig.patch.set_linewidth(2.0)
    fig.patch.set_edgecolor(DARK_BORDER)

    plt.subplots_adjust(top=0.91, bottom=0.07, left=0.15, right=0.94, hspace=0.35)
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
    plt.close(fig)


def main():
    logs_root = Path("logs")
    if not logs_root.exists():
        raise FileNotFoundError(f"Directory '{logs_root}' does not exist.")

    # 1. Pobranie najnowszego folderu i śledzenie łańcucha sesji
    newest_dir = get_newest_folder(logs_root)
    chain_dirs = collect_chain_folders(newest_dir, logs_root)
    print(f"Collected {len(chain_dirs)} sessions: {[d.name for d in chain_dirs]}")

    # 2. Parsowanie statystyk z każdej sesji
    session_labels = [f"{i+1}" for i in range(len(chain_dirs))]
    
    all_skills = []
    session_stats = []

    for s_dir in chain_dirs:
        stats = parse_execution_statistics(s_dir / "execution_statistics.txt")
        session_stats.append(stats)
        for skill in list(stats["collected_data"].keys()) + list(stats["executions"].keys()):
            if skill not in all_skills:
                all_skills.append(skill)

    # 3. Przygotowanie wektorów danych dla wykresów
    data_nodes_by_skill = {skill: [] for skill in all_skills}
    executions_by_skill = {skill: [] for skill in all_skills}
    tp_by_skill = {skill: [] for skill in all_skills}
    tn_by_skill = {skill: [] for skill in all_skills}

    for stats in session_stats:
        for skill in all_skills:
            data_nodes_by_skill[skill].append(stats["collected_data"].get(skill, 0))
            executions_by_skill[skill].append(stats["executions"].get(skill, 0))
            tp_by_skill[skill].append(stats["tp"].get(skill, 0))
            tn_by_skill[skill].append(stats["tn"].get(skill, 0))

    # 4. Wykres 1: Collected Data (Nodes per Skill)
    path_collected = newest_dir / "skills_collected_data_trend.png"
    safe_path_collected = resolve_safe_output_path(path_collected)
    if safe_path_collected:
        render_trend_plot(
            session_labels=session_labels,
            skills_data=data_nodes_by_skill,
            title_main="Collected Data Nodes Trend",
            y_label="Nodes Count",
            output_path=safe_path_collected,
        )
        print(f"[SUCCESS] Saved: {safe_path_collected.resolve()}")

    # 5. Wykres 2: Executions (Total Executions per Skill)
    path_executions = newest_dir / "skills_executions_trend.png"
    safe_path_executions = resolve_safe_output_path(path_executions)
    if safe_path_executions:
        render_trend_plot(
            session_labels=session_labels,
            skills_data=executions_by_skill,
            title_main="Skill Executions Trend",
            y_label="Total Executions",
            output_path=safe_path_executions,
        )
        print(f"[SUCCESS] Saved: {safe_path_executions.resolve()}")

    # 6. Wykres 3: Expected Outcome Executions (Stacked TP + TN per Skill)
    path_expected = newest_dir / "skills_expected_outcome_executions.png"
    safe_path_expected = resolve_safe_output_path(path_expected)
    if safe_path_expected:
        render_stacked_expected_outcome_plots(
            session_labels=session_labels,
            all_skills=all_skills,
            tp_data=tp_by_skill,
            tn_data=tn_by_skill,
            title_main="Expected Outcome Executions (TP + TN)",
            output_path=safe_path_expected,
        )
        print(f"[SUCCESS] Saved: {safe_path_expected.resolve()}")


if __name__ == "__main__":
    main()
