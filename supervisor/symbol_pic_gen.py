import os
import glob
import json
import cv2
import numpy as np

from supervisor.picture_generator import PictureGenerator
from supervisor.renderer import lower_median


def _find_pddl_gng_dir(session_path: str):
    """Case-insensitive directory lookup for <session_path>/pddl/gng."""
    if not session_path or not os.path.isdir(session_path):
        return None

    for pddl_cand in ["pddl", "PDDL"]:
        pddl_path = os.path.join(session_path, pddl_cand)
        if os.path.isdir(pddl_path):
            for gng_cand in ["gng", "GNG"]:
                gng_path = os.path.join(pddl_path, gng_cand)
                if os.path.isdir(gng_path):
                    return gng_path

    # Fallback to direct path
    return os.path.join(session_path, "PDDL", "GNG")


def _load_symbol_data(json_path: str):
    """
    Loads sym_<nr>.json and reconstructs active dimensional vectors.
    Total sensimotor feature vector length = 248 (8 lidar + 48 cells * 5 features).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract nodes
    gng_dict = data.get("gng", {})
    raw_nodes = gng_dict.get("nodes", [])
    if not raw_nodes:
        return None, None, None

    nodes_arr = np.array(raw_nodes, dtype=np.float32)

    # Read active dimensions
    active_dims = data.get("active_dimensions")
    if active_dims is None:
        active_dims = data.get("active_dimensions", list(range(nodes_arr.shape[1])))

    active_set = set(active_dims)
    n_nodes = nodes_arr.shape[0]

    # Reconstruct full 248-dim array
    full_nodes = np.zeros((n_nodes, 248), dtype=np.float32)
    for col_idx, feat_idx in enumerate(active_dims):
        if col_idx < nodes_arr.shape[1] and feat_idx < 248:
            full_nodes[:, feat_idx] = nodes_arr[:, col_idx]

    return full_nodes, active_set, nodes_arr


def _build_masks(active_set):
    """Computes active flags for LIDAR, Ganglion cells, and Cone cells."""
    lidar_active = [i in active_set for i in range(8)]

    active_ganglion = []
    active_cones = []
    for c in range(48):
        offset = 8 + (c * 5)
        # Ganglion on/off
        ganglion_active = (offset in active_set) or ((offset + 1) in active_set)
        active_ganglion.append(ganglion_active)

        # Cone R/G/B
        cone_active = all((offset + ch) in active_set for ch in (2, 3, 4))
        active_cones.append(cone_active)

    return lidar_active, active_ganglion, active_cones


class SymbolIconPictureGenerator(PictureGenerator):
    """
    Visualises a single iconic representative state for symbols.
    Selects the prototype node closest to the centroid.
    """
    def run(self):
        if not self.session_path:
            return

        gng_dir = _find_pddl_gng_dir(self.session_path)
        if not gng_dir or not os.path.exists(gng_dir):
            return

        out_dir = self.get_pictures_dir()
        file_pattern = os.path.join(gng_dir, "sym_*.json")

        for json_path in glob.glob(file_pattern):
            try:
                full_nodes, active_set, _ = _load_symbol_data(json_path)
                if full_nodes is None or len(full_nodes) == 0:
                    continue

                file_base = os.path.splitext(os.path.basename(json_path))[0]
                symbol_num = file_base.split("_")[-1]

                # Point closest to manifold centroid
                centroid = np.mean(full_nodes, axis=0)
                dists = np.linalg.norm(full_nodes - centroid, axis=1)
                rep_node = full_nodes[int(np.argmin(dists))]

                lidar_active, active_ganglion, active_cones = _build_masks(active_set)

                display_title = f"Symbol {symbol_num}"
                img = self.renderer.create_base_canvas(
                    display_title,
                    f"Symbol Icon: Center Prototype ({len(active_set)} active dims)"
                )

                self.renderer.draw_lidar(
                    img, rep_node[:8], center=(230, 280),
                    mode="iconic", active_flags=lidar_active,
                    custom_color=self.renderer.primary_color
                )
                self.renderer.draw_grids(
                    img, rep_node[8:248], is_live=True,
                    active_ganglion=active_ganglion, active_cones=active_cones,
                    origin=(520, 120)
                )
                self.renderer.draw_sample_badge(img, n_pos=len(full_nodes), position=(35, 595))

                out_path = os.path.join(out_dir, f"{file_base}_icon.png")
                cv2.imwrite(out_path, img)
                print(f"[SymbolIconPictureGenerator] Saved icon to {out_path}")

            except Exception as e:
                print(f"[SymbolIconPictureGenerator] Error processing {json_path}: {e}")


class SymbolCategoryPictureGenerator(PictureGenerator):
    """
    Visualises category bounds across all prototypes:
    - Min-median-max range for LIDARs.
    - Median activation & color for retinal ganglion & cone cells.
    """
    def run(self):
        if not self.session_path:
            return

        gng_dir = _find_pddl_gng_dir(self.session_path)
        if not gng_dir or not os.path.exists(gng_dir):
            return

        out_dir = self.get_pictures_dir()
        file_pattern = os.path.join(gng_dir, "sym_*.json")

        for json_path in glob.glob(file_pattern):
            try:
                full_nodes, active_set, _ = _load_symbol_data(json_path)
                if full_nodes is None or len(full_nodes) == 0:
                    continue

                file_base = os.path.splitext(os.path.basename(json_path))[0]
                symbol_num = file_base.split("_")[-1]

                # Prototype statistics
                min_vals = np.min(full_nodes, axis=0)
                max_vals = np.max(full_nodes, axis=0)
                median_vals = lower_median(full_nodes, axis=0)

                lidar_active, active_ganglion, active_cones = _build_masks(active_set)

                display_title = f"Symbol {symbol_num}"
                img = self.renderer.create_base_canvas(
                    display_title,
                    f"Symbol Category: Min-Median-Max Manifold ({len(active_set)} active dims)"
                )

                # 1. LIDAR (Min, Max, Median)
                lidar_data = list(zip(min_vals[:8], max_vals[:8], median_vals[:8]))
                self.renderer.draw_lidar(
                    img, lidar_data, mode="categorical", active_flags=lidar_active
                )

                # 2. Retinal Ganglion Cells (Medians for on/off)
                ganglion_medians = []
                for cell_idx in range(48):
                    idx_on = 8 + (cell_idx * 5)
                    idx_off = idx_on + 1
                    ganglion_medians.append(median_vals[idx_on])
                    ganglion_medians.append(median_vals[idx_off])

                # 3. Cone Cells (Median BGR color)
                cone_medians = []
                for cell_idx in range(48):
                    offset = 8 + (cell_idx * 5)
                    r_p = full_nodes[:, offset + 2]
                    g_p = full_nodes[:, offset + 3]
                    b_p = full_nodes[:, offset + 4]
                    median_bgr, _ = self.renderer.compute_hue_median_stats(r_p, g_p, b_p)
                    cone_medians.append(median_bgr)

                self.renderer.draw_grids(
                    img, (ganglion_medians, cone_medians), is_live=False,
                    active_ganglion=active_ganglion, active_cones=active_cones,
                    origin=(520, 120)
                )

                self.renderer.draw_sample_badge(img, n_pos=len(full_nodes), position=(35, 595))

                out_path = os.path.join(out_dir, f"{file_base}_category.png")
                cv2.imwrite(out_path, img)
                print(f"[SymbolCategoryPictureGenerator] Saved category image to {out_path}")

            except Exception as e:
                print(f"[SymbolCategoryPictureGenerator] Error processing {json_path}: {e}")