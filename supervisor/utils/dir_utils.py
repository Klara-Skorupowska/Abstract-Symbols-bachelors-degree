import os
import re
from datetime import datetime
from pathlib import Path


def parse_continuation(stats_file: Path) -> Path | None:
    if not stats_file.exists():
        return None
    pattern = re.compile(r'CONTINUATION FROM SESSION:\s*(.+)', re.IGNORECASE)
    with stats_file.open('r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line.strip())
            if match:
                return Path(match.group(1).strip())
    return None


def get_epoch_dirs(session_dir: Path) -> list[Path]:
    if not session_dir.exists():
        return []
    epochs = [entry for entry in session_dir.iterdir() if entry.is_dir() and entry.name.startswith("epoch_")]
    epochs.sort(key=lambda p: p.name)
    if not epochs:
        return [session_dir]
    return epochs


def collect_chain_folders(current_dir: Path, logs_root: Path) -> list[Path]:
    visited_sessions = {current_dir.resolve()}
    session_chain = [current_dir]
    curr = current_dir

    while True:
        epochs = get_epoch_dirs(curr)
        continued_rel = None
        
        for cand_dir in reversed(epochs + [curr]):
            stats_file = cand_dir / "execution_statistics.txt"
            continued_rel = parse_continuation(stats_file)
            if continued_rel:
                break
        
        if not continued_rel:
            break
            
        next_dir = logs_root.parent / continued_rel
        if not next_dir.exists():
            next_dir = logs_root / continued_rel.name
            if not next_dir.exists():
                break
                
        next_resolved = next_dir.resolve()
        if next_resolved in visited_sessions:
            break
        
        visited_sessions.add(next_resolved)
        session_chain.append(next_dir)
        curr = next_dir

    session_chain.reverse()
    
    all_dirs = []
    for sess in session_chain:
        all_dirs.extend(get_epoch_dirs(sess))
        
    return all_dirs


def parse_session_timestamp(dir_name: str) -> datetime | None:
    try:
        return datetime.strptime(dir_name, "%Y-%m-%d_%H-%M")
    except ValueError:
        return None


def get_latest_session(logs_root: str | Path = "logs", exclude: str | Path | None = None) -> Path | None:
    """Finds the most recent session folder in logs_root, optionally excluding a specific session."""
    root = Path(logs_root)
    if not root.exists():
        return None

    exclude_resolved = Path(exclude).resolve() if exclude else None
    valid_sessions: list[Path] = []
    
    for entry in root.iterdir():
        if entry.is_dir() and parse_session_timestamp(entry.name) is not None:
            if exclude_resolved and entry.resolve() == exclude_resolved:
                continue
            valid_sessions.append(entry)

    if not valid_sessions:
        return None
    return max(valid_sessions, key=lambda p: p.name)


def get_latest_epoch_dir(session_dir: Path | str) -> Path | None:
    """Finds the most recent epoch folder within a session directory."""
    sess_path = Path(session_dir)
    if not sess_path.exists():
        return None
    epoch_dirs = [
        p for p in sess_path.iterdir()
        if p.is_dir() and p.name.startswith("epoch_")
    ]
    if not epoch_dirs:
        return None
    return max(epoch_dirs, key=lambda p: p.name)


def resolve_source_directory(
    source_dir: str | None,
    current_session: str | Path | None = None,
    logs_root: str | Path = "logs"
) -> tuple[Path | None, Path | None]:
    """
    Resolves the source directory to load the brain from.
    Ensures that when continuing from previous sessions, the newly initialized
    current session directory is excluded from candidate lookup.

    Returns:
        (latest_epoch_path, parent_session_path)
    """
    if not source_dir or source_dir == 'new':
        return None, None

    root = Path(logs_root)
    session_path: Path | None = None

    if source_dir == 'continue':
        session_path = get_latest_session(root, exclude=current_session)
        if not session_path:
            print("[SessionUtils] No previous session found to continue from.")
            return None, None
        print(f"[SessionUtils] Found previous session to continue from: '{session_path.name}'")
    else:
        cand = Path(source_dir)
        if not cand.is_absolute():
            cand = root / source_dir
        if cand.exists() and cand.is_dir():
            session_path = cand
        else:
            print(f"[SessionUtils] Target session directory '{cand}' does not exist.")
            return None, None

    latest_epoch = get_latest_epoch_dir(session_path)
    if latest_epoch:
        print(f"[SessionUtils] Loaded latest epoch '{latest_epoch.name}' from session '{session_path.name}'")
        return latest_epoch, session_path

    print(f"[SessionUtils] No epoch folders in session '{session_path.name}'. Fallback to session root.")
    return session_path, session_path