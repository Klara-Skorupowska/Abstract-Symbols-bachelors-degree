from __future__ import annotations

import os
from pathlib import Path
import re
from dataclasses import dataclass, field


@dataclass
class SkillFinalStats:
    nodes: int
    ratio: float | None  # (TP + TN) / (FP + FN), None if zero-division
    pos_ratio: float | None # TP / (TP + FN)
    neg_ratio: float | None # TN / (TN + FP)


@dataclass
class CumulativeExecutionStats:
    num_sessions: int = 0
    total_duration_sec: int = 0
    total_runs: int = 0
    total_brain_updates: int = 0
    total_completed_plans: int = 0
    final_skills_data: dict[str, SkillFinalStats] = field(default_factory=dict)


class CumulativeStatsScraper:
    """Scrapes cumulative execution statistics across chained log session files."""

    def __init__(self, logs_root: str = "logs", session_dir: str | None = None):
        self.logs_root = logs_root
        self.session_path = session_dir

    @staticmethod
    def parse_continuation(stats_file: Path) -> Path | None:
        """Parses the continuation path from an execution statistics file."""
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
        """Resolves the chain of session directories in chronological order."""
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
    def parse_duration_to_seconds(duration_str: str) -> int:
        """Converts HH:MM:SS string directly into integer total seconds."""
        parts = list(map(int, duration_str.strip().split(":")))
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return parts[0]

    @staticmethod
    def format_total_seconds(total_seconds: int) -> str:
        """Formats total accumulated seconds strictly as hours:minutes:seconds without days."""
        hours = total_seconds // 3600
        remainder = total_seconds % 3600
        minutes = remainder // 60
        seconds = remainder % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def parse_session_aggregates(self, stats_file: Path) -> tuple[int, int, int, int]:
        """Extracts and returns (duration_seconds, runs, brain_updates, completed_plans) from a session."""
        duration_sec = 0
        runs = 0
        brain_updates = 0
        completed_plans = 0

        duration_pat = re.compile(r"duration:\s*([0-9:]+)", re.IGNORECASE)
        stats_pat = re.compile(
            r"runs:\s*(\d+);\s*brain updates:\s*(\d+);\s*completed plans:\s*(\d+)",
            re.IGNORECASE,
        )

        with stats_file.open("r", encoding="utf-8") as f:
            for line in f:
                d_match = duration_pat.search(line)
                if d_match:
                    duration_sec = self.parse_duration_to_seconds(d_match.group(1))

                s_match = stats_pat.search(line)
                if s_match:
                    runs = int(s_match.group(1))
                    brain_updates = int(s_match.group(2))
                    completed_plans = int(s_match.group(3))

        return duration_sec, runs, brain_updates, completed_plans

    def parse_final_session_skills(self, stats_file: Path) -> dict[str, SkillFinalStats]:
        """Extracts single node counts and (TP+TN)/(FP+FN) ratio for each skill in final session."""
        skills_nodes: dict[str, int] = {}
        skills_ratios: dict[str, float | None] = {}
        skills_positive_ratios: dict[str, float | None] = {}
        skills_negative_ratios: dict[str, float | None] = {}

        mode = None
        with stats_file.open("r", encoding="utf-8") as f:
            for line in f:
                raw_line = line.strip()
                if not raw_line or raw_line.startswith("---"):
                    continue

                if "Collected Data:" in raw_line:
                    mode = "collected_data"
                    continue
                elif "Executions:" in raw_line:
                    mode = "executions"
                    continue

                parts = [p.strip() for p in raw_line.split("|")]
                if len(parts) < 2 or parts[0].lower() == "skill name":
                    continue

                skill_name = parts[0]

                if mode == "collected_data" and len(parts) >= 2:
                    try:
                        # Precondition and effect nodes are identical, extract one
                        skills_nodes[skill_name] = int(parts[1])
                    except ValueError as e:
                        print(f"[CS] {e}")
                        continue

                elif mode == "executions" and len(parts) >= 5:
                    try:
                        tp = float(parts[1])
                        tn = float(parts[2])
                        fp = float(parts[3])
                        fn = float(parts[4])
                        okay = (fn+tp+fp+tn > 0)
                        skills_ratios[skill_name] = (tp + tn)/(fn+tp+fp+tn) if okay else None
                        okay = (tp + fn > 0)
                        skills_positive_ratios[skill_name] = tp / (tp + fn) if okay else None
                        okay = (tn + fp > 0)
                        skills_negative_ratios[skill_name] = tn / (tn + fp) if okay else None
                    except ValueError as e:
                        print(f"[CS] {e}")
                        continue

        all_skills = set(skills_nodes.keys()).union(skills_ratios.keys())
        results: dict[str, SkillFinalStats] = {}
        for skill in all_skills:
            results[skill] = SkillFinalStats(
                nodes=skills_nodes.get(skill, 0),
                ratio=skills_ratios.get(skill, None),
                pos_ratio=skills_positive_ratios.get(skill, None),
                neg_ratio=skills_negative_ratios.get(skill, None),
            )

        return results

    def scrape(self) -> CumulativeExecutionStats | None:
        """Scrapes cumulative stats across the chained sessions ending at target session."""
        if not self.session_path or not os.path.exists(self.session_path):
            print("[CS] Session path do not exists")
            return None

        current_dir = Path(self.session_path)
        logs_root = Path(self.logs_root)
        chain_dirs = self.collect_chain_folders(current_dir, logs_root)

        if not chain_dirs:
            print("[CS] No chain of dirs found")
            return None

        cumulative = CumulativeExecutionStats(num_sessions=len(chain_dirs))

        # Sum durations and aggregates across all sessions in the chain
        for s_dir in chain_dirs:
            s_file = s_dir / "execution_statistics.txt"
            if not s_file.exists():
                print(f"[CS] No 'execution_statistics.txt' in {s_dir}")
                continue
            duration_sec, runs, updates, plans = self.parse_session_aggregates(s_file)
            cumulative.total_duration_sec += duration_sec
            cumulative.total_runs += runs
            cumulative.total_brain_updates += updates
            cumulative.total_completed_plans += plans

        # Parse skill node counts and ratios from the final session
        final_stats_file = chain_dirs[-1] / "execution_statistics.txt"
        if final_stats_file.exists():
            print(f"[CS] No 'execution_statistics.txt' in {s_dir}")
            cumulative.final_skills_data = self.parse_final_session_skills(final_stats_file)

        return cumulative

    def save_to_txt(self, stats: CumulativeExecutionStats, output_path: str | Path) -> None:
        """Writes cumulative statistics and final skill node/ratio metrics to a text file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        duration_formatted = self.format_total_seconds(stats.total_duration_sec)

        lines: list[str] = [
            "-" * 60,
            "CUMULATIVE EXECUTION STATISTICS",
            "-" * 60,
            f"number of sessions: {stats.num_sessions}",
            f"total duration: {duration_formatted}",
            f"total number of runs: {stats.total_runs}",
            f"total number of brain updates: {stats.total_brain_updates}",
            f"total number of completed plans: {stats.total_completed_plans}",
            "-" * 60,
            "FINAL SESSION SKILLS",
            "-" * 100,
            f"{'Skill Name':<20} | {'Nodes':<10} | {'true/all ratio':<20} | {'true success ratio':<20} | {'true fail ratio':<20} ",
            "-" * 100,
        ]

        for skill, data in stats.final_skills_data.items():
            ratio_str = f"{data.ratio:.3f}" if data.ratio is not None else "N/A"
            pos_ratio_str = f"{data.pos_ratio:.3f}" if data.pos_ratio is not None else "N/A"
            neg_ratio_str = f"{data.neg_ratio:.3f}" if data.neg_ratio is not None else "N/A"
            eps = 0.01  # 1% error accepted
            tag = " \t(●'◡'●)" if data.ratio is not None and data.ratio >= 1-eps and data.pos_ratio is not None and data.pos_ratio >= 1-eps  else ""

            lines.append(f"{skill:<20} | {data.nodes:<10} | {ratio_str:<20} | {pos_ratio_str:<20} | {neg_ratio_str:<20}{tag}")

        lines.append("-" * 100 + "\n")

        with output_file.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def export_stats(self, output_filename: str = "cumulative_statistics.txt") -> Path | None:
        """Runs the scraper and outputs the text file directly into the current session folder."""
        stats = self.scrape()
        if not stats or not self.session_path:
            print("[Cumulative Stats] No stats or session path.")
            return None

        out_path = Path(self.session_path) / output_filename
        self.save_to_txt(stats, out_path)
        return out_path


if __name__ == "__main__":
    scraper = CumulativeStatsScraper(logs_root=r"logs", session_dir=r"logs\2026-11-04_07-00")
    scraper.export_stats()