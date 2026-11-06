import os, json
import numpy as np
from datetime import datetime

SKILL_ORDER = ['Start', 'Find_The_Door', 'Go_To_The_Door', 'Open_The_Door', 'Find_The_Goal', 'Go_To_The_Goal', 'Finish']
SKILL_ORDER = [skill.lower() for skill in SKILL_ORDER]

def get_session(logs_root, session_dir=None):
    if not os.path.exists(logs_root):
        return None
    if session_dir is not None and os.path.exists(os.path.join(logs_root, session_dir)):
        return os.path.join(logs_root, session_dir)

    def parse_session_timestamp(dir_name):
        try:
            return datetime.strptime(dir_name, '%Y-%m-%d_%H-%M')
        except ValueError:
            return None
    valid_sessions = []
    for entry in os.scandir(logs_root):
        if entry.is_dir() and parse_session_timestamp(entry.name):
            valid_sessions.append(entry.path)
    return max(valid_sessions, key=os.path.basename, default=None)

def lower_median(arr, axis=None):
    a = np.asarray(arr)
    if axis is None:
        flat = a.ravel()
        n = flat.size
        if n == 0: return np.nan
        k = (n - 1) // 2
        return np.partition(flat, k)[k]
    else:
        n = a.shape[axis]
        if n == 0: return np.nan
        k = (n - 1) // 2
        return np.take(np.sort(a, axis=axis), k, axis=axis)

def _find_pddl_gng_dir(session_path: str):
    if not session_path or not os.path.isdir(session_path):
        return None
    for pddl_cand in ['pddl', 'PDDL']:
        pddl_path = os.path.join(session_path, pddl_cand)
        if os.path.isdir(pddl_path):
            for gng_cand in ['gng', 'GNG']:
                gng_path = os.path.join(pddl_path, gng_cand)
                if os.path.isdir(gng_path):
                    return gng_path
    return os.path.join(session_path, 'PDDL', 'GNG')

def _load_symbol_data(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    gng_dict = data.get('gng', {})
    raw_nodes = gng_dict.get('nodes', [])
    if not raw_nodes:
        return None, None, None
    nodes_arr = np.array(raw_nodes, dtype=np.float32)
    active_dims = data.get('active_dimensions')
    if active_dims is None:
        active_dims = data.get('active_dimensions', list(range(nodes_arr.shape[1])))
    active_set = set(active_dims)
    n_nodes = nodes_arr.shape[0]
    full_nodes = np.zeros((n_nodes, 248), dtype=np.float32)
    for col_idx, feat_idx in enumerate(active_dims):
        if col_idx < nodes_arr.shape[1] and feat_idx < 248:
            full_nodes[:, feat_idx] = nodes_arr[:, col_idx]
    return full_nodes, active_set, nodes_arr

def _build_masks(active_set):
    lidar_active = [i in active_set for i in range(8)]
    active_ganglion = []
    active_cones = []
    for c in range(48):
        offset = 8 + c * 5
        ganglion_active = offset in active_set or (offset + 1) in active_set
        active_ganglion.append(ganglion_active)
        cone_active = all((offset + ch) in active_set for ch in (2, 3, 4))
        active_cones.append(cone_active)
    return lidar_active, active_ganglion, active_cones
