import argparse
import re
import traceback
from pathlib import Path

from supervisor.generators.pipeline import PipelinePictureGenerator


def find_experiment_dirs(experiments_root: Path, only: list[str] | None = None) -> list[Path]:
    """Returns exp_<number> directories sorted numerically."""
    if not experiments_root.exists():
        raise FileNotFoundError(f"Experiments directory not found: {experiments_root}")

    dirs: list[tuple[int, Path]] = []
    only_set = set(only) if only else None

    for entry in experiments_root.iterdir():
        if not entry.is_dir():
            continue
        match = re.match(r"^exp_(\d+)$", entry.name)
        if not match:
            continue
        if only_set and entry.name not in only_set:
            continue
        dirs.append((int(match.group(1)), entry))

    dirs.sort(key=lambda item: item[0])
    return [path for _, path in dirs]


def regenerate_all_pictures(
    logs_root: str | Path = "logs",
    experiments_subdir: str = "experiments",
    only: list[str] | None = None,
) -> None:
    logs_root_path = Path(logs_root)
    experiments_root = logs_root_path / experiments_subdir
    exp_dirs = find_experiment_dirs(experiments_root, only=only)

    if not exp_dirs:
        print(f"[Regenerate] No matching exp_* directories found in {experiments_root}.")
        return

    print(f"[Regenerate] Found {len(exp_dirs)} experiment(s) in {experiments_root}.")
    failed: list[str] = []

    for idx, exp_dir in enumerate(exp_dirs, start=1):
        print(f"\n[{idx}/{len(exp_dirs)}] Generating pictures for {exp_dir.name}...")
        try:
            pics = PipelinePictureGenerator(
                logs_root=str(logs_root_path),
                session_dir=str(exp_dir),
            )
            pics.run()
            print(f"[Regenerate] Done: {exp_dir / 'pictures'}")
        except Exception as e:
            print(f"[Regenerate] ERROR in {exp_dir.name}: {e}")
            traceback.print_exc()
            failed.append(exp_dir.name)

    print(f"\n[Regenerate] Finished. Succeeded: {len(exp_dirs) - len(failed)}/{len(exp_dirs)}.")
    if failed:
        print(f"[Regenerate] Failed experiments: {', '.join(failed)}")


if __name__ == "__main__":
    LOGS_ROOT = "logs"
    EXPERIMENTS_SUBDIR = "experiments"
    ONLY_EXPERIMENTS = None  # e.g., ["exp_001", "exp_003"], or None for all

    regenerate_all_pictures(
        logs_root=LOGS_ROOT,
        experiments_subdir=EXPERIMENTS_SUBDIR,
        only=ONLY_EXPERIMENTS,
    )