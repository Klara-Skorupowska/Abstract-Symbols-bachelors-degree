from __future__ import annotations

from collections import Counter
from datetime import timedelta
from pathlib import Path
import re
import statistics


def get_newest_folder(logs_dir: Path) -> Path:
    folders = [p for p in logs_dir.iterdir() if p.is_dir()]
    if not folders:
        raise FileNotFoundError(f"No subdirectories found in {logs_dir}")
    return max(folders, key=lambda p: p.stat().st_mtime)


def parse_continuation(stats_file: Path) -> Path | None:
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


def parse_duration_to_seconds(dur_str: str) -> float:
    """Converts a duration string formatted as HH:MM:SS or H:M:S into total seconds."""
    parts = [float(p) for p in dur_str.strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return float(dur_str)


def format_seconds(seconds: float) -> str:
    td = timedelta(seconds=int(round(seconds)))
    return str(td)


def get_dominant(items: list) -> str:
    """Returns the most frequent item (mode) as a string, with tie-break handling."""
    if not items:
        return "N/A"
    counts = Counter(items)
    dominant_val, _ = counts.most_common(1)[0]
    return str(dominant_val)


def parse_execution_statistics(file_path: Path):
    """Parses execution_statistics.txt into structured metrics."""
    with file_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    duration_sec = 0.0
    runs = 0
    brain_updates = 0

    dur_match = re.search(r"duration:\s*([0-9:]+)", "\n".join(lines), re.IGNORECASE)
    if dur_match:
        duration_sec = parse_duration_to_seconds(dur_match.group(1))

    runs_match = re.search(
        r"runs:\s*(\d+);\s*brain updates:\s*(\d+)", "\n".join(lines), re.IGNORECASE
    )
    if runs_match:
        runs = int(runs_match.group(1))
        brain_updates = int(runs_match.group(2))

    skills_data: dict[str, list[int]] = {}
    plan_lengths: list[int] = []
    steps_executed: list[int] = []
    test_durations: list[float] = []
    results: list[str] = []

    mode = None
    for line in lines:
        if not line or set(line) <= {"-"}:
            continue

        if "Skill Name" in line:
            mode = "skills"
            continue
        if "Test" in line and "Plan Length" in line:
            mode = "tests"
            continue

        cols = [col.strip() for col in line.split("|")]

        if mode == "skills" and len(cols) >= 6:
            skill_name = cols[0]
            try:
                counts = [int(v) for v in cols[1:6]]
                skills_data[skill_name] = counts
            except ValueError:
                continue

        elif mode == "tests" and len(cols) >= 6:
            try:
                plan_len = int(cols[1])
                steps = int(cols[2])
                dur = float(cols[4])
                res = cols[5]

                plan_lengths.append(plan_len)
                steps_executed.append(steps)
                test_durations.append(dur)
                results.append(res)
            except ValueError:
                continue

    return {
        "duration_sec": duration_sec,
        "runs": runs,
        "brain_updates": brain_updates,
        "skills_data": skills_data,
        "plan_lengths": plan_lengths,
        "steps_executed": steps_executed,
        "test_durations": test_durations,
        "results": results,
    }


def main():
    logs_root = Path("logs")
    if not logs_root.exists():
        raise FileNotFoundError(f"'{logs_root}' directory does not exist.")

    # 1. Locate the newest session directory
    newest_dir = get_newest_folder(logs_root)
    print(f"Newest session: {newest_dir.name}")

    # 2. Traverse backward through continuation chains
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

    # 3. Aggregate statistics across all chained sessions
    total_runs = 0
    total_brain_updates = 0
    total_learning_duration_sec = 0.0

    aggregated_skills: dict[str, list[int]] = {}
    all_plan_lengths: list[int] = []
    all_steps_executed: list[int] = []
    all_durations: list[float] = []
    all_results: list[str] = []

    for d in chain_dirs:
        stats_file = d / "execution_statistics.txt"
        if not stats_file.exists():
            print(f"Warning: Missing '{stats_file}'")
            continue

        data = parse_execution_statistics(stats_file)
        total_runs += data["runs"]
        total_brain_updates += data["brain_updates"]
        total_learning_duration_sec += data["duration_sec"]

        for skill, counts in data["skills_data"].items():
            if skill not in aggregated_skills:
                aggregated_skills[skill] = [0, 0, 0, 0, 0]
            for i in range(5):
                aggregated_skills[skill][i] += counts[i]

        all_plan_lengths.extend(data["plan_lengths"])
        all_steps_executed.extend(data["steps_executed"])
        all_durations.extend(data["test_durations"])
        all_results.extend(data["results"])

    total_tests = len(all_plan_lengths)

    # 4. Compute Test Statistics
    mean_plan_length = statistics.mean(all_plan_lengths) if all_plan_lengths else 0.0
    dominant_plan_length = get_dominant(all_plan_lengths)

    mean_steps = statistics.mean(all_steps_executed) if all_steps_executed else 0.0
    dominant_steps = get_dominant(all_steps_executed)

    mean_duration = statistics.mean(all_durations) if all_durations else 0.0
    dominant_result = get_dominant(all_results)

    try:
        numeric_results = [float(r) for r in all_results]
        mean_result = f"{statistics.mean(numeric_results):.4f}"
    except (ValueError, statistics.StatisticsError):
        mean_result = "N/A"

    # 5. Format and write out all_execution_statistics.txt
    sep = "-" * 100
    output_lines = [
        "AGGREGATED EXECUTION STATISTICS",
        sep,
        f"Sessions Included: {len(chain_dirs)}",
        f"Total Learning Duration: {format_seconds(total_learning_duration_sec)} ({total_learning_duration_sec:.2f}s)",
        f"Total Runs: {total_runs}",
        f"Total Brain Updates: {total_brain_updates}",
        sep,
        "AGGREGATED SKILLS EVALUATION",
        f"{'Skill Name':<20} | {'True Positive':<15} | {'True Negative':<15} | {'False Positive':<15} | {'False Negative':<15} | {'Total':<10}",
        sep,
    ]

    for skill, counts in aggregated_skills.items():
        output_lines.append(
            f"{skill:<20} | {counts[0]:<15} | {counts[1]:<15} | {counts[2]:<15} | {counts[3]:<15} | {counts[4]:<10}"
        )

    output_lines.extend(
        [
            sep,
            "AGGREGATED TESTS SUMMARY",
            f"Total Tests Executed: {total_tests}",
            f"Plan Length:        Mean = {mean_plan_length:.3f}, Dominant = {dominant_plan_length}",
            f"Steps Executed:     Mean = {mean_steps:.3f}, Dominant = {dominant_steps}",
            f"Test Duration [s]:  Mean = {mean_duration:.3f}s",
            f"Result:             Mean = {mean_result}, Dominant = {dominant_result}",
            sep,
        ]
    )

    out_file = newest_dir / "all_execution_statistics.txt"
    out_file.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    # Output print showing both relative and full absolute path
    print(f"\n[SUCCESS] Aggregated statistics saved to:")
    print(f"  -> Relative path: {out_file}")
    print(f"  -> Absolute path: {out_file.resolve()}")


if __name__ == "__main__":
    main()