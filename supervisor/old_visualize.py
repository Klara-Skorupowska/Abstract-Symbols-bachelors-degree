import os
import re
import glob
import json
import time
import cv2
import numpy as np
import math
from communicator import Communicator
import networkx as nx
import textwrap
from datetime import datetime

# Import static deserializer / predicate loaders from the project
from agent.brain_network import BrainNetwork

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def lower_median(arr, axis=None):
    """Computes the lower median: for even n, takes the smaller middle value rather than the mean."""
    a = np.asarray(arr)
    if axis is None:
        flat = a.ravel()
        n = flat.size
        if n == 0:
            return np.nan
        k = (n - 1) // 2
        return np.partition(flat, k)[k]
    else:
        n = a.shape[axis]
        if n == 0:
            return np.nan
        k = (n - 1) // 2
        return np.take(np.sort(a, axis=axis), k, axis=axis)

def get_session(logs_root, session_dir):
    if not os.path.exists(logs_root):
        return None
    if session_dir is not None and os.path.exists(os.path.join(logs_root, session_dir)):
        return os.path.join(logs_root, session_dir)

    def parse_session_timestamp(dir_name):
        try:
            return datetime.strptime(dir_name, "%Y-%m-%d_%H-%M")
        except ValueError:
            return None

    valid_sessions = []
    for entry in os.scandir(logs_root):
        if entry.is_dir() and parse_session_timestamp(entry.name):
            valid_sessions.append(entry.path)

    return max(valid_sessions, key=os.path.basename, default=None)

def parse_predicate_json(data: dict):
    """Normalizes legacy 1-GNG JSON format and modern 3-value logic JSON format."""
    if data.get("version") == "3_val_logic":
        pos_dict = data.get("gng_pos", {})
        neg_dict = data.get("gng_neg", {})
        return pos_dict, neg_dict
    return data, None

SKILL_ORDER = [
    "Start", "Find_The_Door", "Go_To_The_Door", "Open_The_Door",
    "Find_The_Goal", "Go_To_The_Goal", "Finish"
]

def calculate_predicate_overlap(eff_pred_path: str, init_pred_path: str):
    """
    Computes overlap ratio between Eff(source) and Init(target) 
    using the exact logic defined in BrainNetwork.calculate_overlap.
    """
    if not os.path.exists(eff_pred_path) or not os.path.exists(init_pred_path):
        return None
    try:
        eff_pred = BrainNetwork.load_predicate(eff_pred_path)
        init_pred = BrainNetwork.load_predicate(init_pred_path)

        # eff_pred.nodes exposes positive nodes
        if not eff_pred.nodes or len(eff_pred.nodes) < 1:
            return None
        if not init_pred.nodes or len(init_pred.nodes) < 1:
            return None

        overlapping_points = [
            node for node in eff_pred.nodes
            if init_pred.is_active(node)
        ]
        return float(len(overlapping_points) / len(eff_pred.nodes))
    except Exception:
        return None

def load_action_masks(stats_file_path: str):
    """
    Parses dimension_statistics.txt and returns a dictionary:
    {
        'Skill_Name': set([dimension_indices...])
    }
    """
    action_masks = {}
    if not os.path.exists(stats_file_path):
        return action_masks

    current_skill = None
    with open(stats_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Skill:"):
                current_skill = line.split("Skill:")[1].strip()
            elif "Changed Dimensions (Action Mask)" in line and current_skill:
                match = re.search(r"\[(.*?)\]", line)
                if match:
                    dims_str = match.group(1).strip()
                    if dims_str:
                        dims = [int(x.strip()) for x in dims_str.split(",") if x.strip()]
                        action_masks[current_skill] = set(dims)
    return action_masks

# =============================================================================
# SHARED RENDERERS & UTILS
# =============================================================================

class Renderer:
    """Shared drawing logic for sensimotor, topological, and tabular visualizations."""

    def __init__(self):
        self.primary_color = (195, 105, 30)       # Slate / Steel Blue
        self.secondary_color = (235, 185, 90)     # Sky Blue / Cyan Accent
        self.tertiary_color = (248, 235, 215)     # Soft Ice Blue tint

        self.BG_COLOR = (248, 249, 250)
        self.BORDER_COLOR = (40, 44, 52)
        self.TEXT_MAIN = (30, 30, 30)
        self.ROBOT_BODY = (40, 44, 52)
        self.PANEL_BORDER = (210, 215, 220)

        self.neutral_gray = (130, 135, 140)
        self.status_green = (45, 160, 50)
        self.status_red = (45, 50, 205)
        self.LIDAR_ANGLES = [17, 50, 90, 150, 210, 270, 310, 343]

    def compute_hue_median_stats(self, r_vals, g_vals, b_vals):
        max_v = max(np.max(r_vals), np.max(g_vals), np.max(b_vals))
        if max_v > 1.0:
            r = np.array(r_vals, dtype=np.float32) / 255.0
            g = np.array(g_vals, dtype=np.float32) / 255.0
            b = np.array(b_vals, dtype=np.float32) / 255.0
        else:
            r = np.array(r_vals, dtype=np.float32)
            g = np.array(g_vals, dtype=np.float32)
            b = np.array(b_vals, dtype=np.float32)

        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        h = np.zeros_like(r)
        nonzero = delta > 1e-6

        mask = nonzero & (cmax == r)
        h[mask] = (60.0 * (((g[mask] - b[mask]) / delta[mask]) % 6))

        mask = nonzero & (cmax == g)
        h[mask] = (60.0 * (((b[mask] - r[mask]) / delta[mask]) + 2))

        mask = nonzero & (cmax == b)
        h[mask] = (60.0 * (((r[mask] - g[mask]) / delta[mask]) + 4))

        angles = np.deg2rad(h)
        sin_mean = np.mean(np.sin(angles))
        cos_mean = np.mean(np.cos(angles))
        R = np.hypot(sin_mean, cos_mean)
        circ_var = 1.0 - R

        mean_angle = np.arctan2(sin_mean, cos_mean)
        diff_angles = np.arctan2(np.sin(angles - mean_angle), np.cos(angles - mean_angle))

        median_diff = lower_median(diff_angles)
        median_angle = (mean_angle + median_diff + 2 * np.pi) % (2 * np.pi)
        median_angle_deg = np.rad2deg(median_angle)

        hsv_pixel = np.uint8([[[int(median_angle_deg / 2.0), 255, 255]]])
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        median_bgr = (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))

        return median_bgr, circ_var

    def draw_sample_badge(self, img, n_pos, position=(35, 595), overlap_ratio=None):
        text = f"Prototypes: {n_pos} TRUE (+)"
        if overlap_ratio is not None:
            text += f" | Overlap: {overlap_ratio * 100:.1f}%"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = position
        pad_x, pad_y = 10, 6

        cv2.rectangle(img, (x, y - text_h - pad_y), (x + text_w + 2 * pad_x, y + baseline + pad_y), self.tertiary_color, -1)
        cv2.rectangle(img, (x, y - text_h - pad_y), (x + text_w + 2 * pad_x, y + baseline + pad_y), self.primary_color, 1)
        cv2.putText(img, text, (x + pad_x, y), font, scale, self.TEXT_MAIN, thickness, cv2.LINE_AA)

    def create_base_canvas(self, title, subtitle=None, width=1000, height=620):
        img = np.ones((height, width, 3), dtype=np.uint8)
        img[:] = self.BG_COLOR
        cv2.rectangle(img, (0, 0), (width - 1, height - 1), self.BORDER_COLOR, 3)

        cv2.putText(img, title, (35, 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, self.TEXT_MAIN, 2, cv2.LINE_AA)
        if subtitle:
            cv2.putText(img, subtitle, (35, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 70, 20), 1, cv2.LINE_AA)

        y_line = 85 if subtitle else 65
        cv2.line(img, (35, y_line), (width - 35, y_line), self.PANEL_BORDER, 1, cv2.LINE_AA)
        return img

    def draw_lidar(self, img, lidar_data, center=(230, 280), max_radius=150, mode="live", active_flags=None, custom_color=None):
        cv2.putText(img, "LIDAR", (45, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)
        color_main = custom_color if custom_color is not None else self.primary_color
        color_sec = self.secondary_color if custom_color is None else tuple(int(0.6 * c + 0.4 * 255) for c in color_main)

        for ang in self.LIDAR_ANGLES:
            rad = np.deg2rad(ang - 90)
            x_end = int(center[0] + max_radius * np.cos(rad))
            y_end = int(center[1] + max_radius * np.sin(rad))
            cv2.line(img, center, (x_end, y_end), self.tertiary_color, 2, cv2.LINE_AA)

        for i, ang in enumerate(self.LIDAR_ANGLES):
            if active_flags is not None and not active_flags[i]:
                continue

            rad = np.deg2rad(ang - 90)
            if mode == "live":
                val = max(min(float(lidar_data[i]), 0.1), 0.0)
                r_val = (val / 0.1) * max_radius
                x_val = int(center[0] + r_val * np.cos(rad))
                y_val = int(center[1] + r_val * np.sin(rad))
                cv2.line(img, center, (x_val, y_val), color_main, 3, cv2.LINE_AA)
                cv2.circle(img, (x_val, y_val), 4, color_main, -1, cv2.LINE_AA)

            elif mode == "iconic":
                val = float(lidar_data[i])
                norm_val = val / 0.1 if val <= 0.1 else val
                r_val = (max(min(norm_val, 1.0), 0.0)) * max_radius
                x_val = int(center[0] + r_val * np.cos(rad))
                y_val = int(center[1] + r_val * np.sin(rad))
                cv2.circle(img, (x_val, y_val), 5, color_main, -1, cv2.LINE_AA)

            elif mode == "categorical":
                val_min, val_max, val_median = lidar_data[i]
                norm_min = val_min / 0.1 if val_min <= 0.1 else val_min
                norm_max = val_max / 0.1 if val_max <= 0.1 else val_max
                norm_median = val_median / 0.1 if val_median <= 0.1 else val_median

                r_min = (max(min(norm_min, 1.0), 0.0)) * max_radius
                r_max = (max(min(norm_max, 1.0), 0.0)) * max_radius
                r_med = (max(min(norm_median, 1.0), 0.0)) * max_radius

                x_min = int(center[0] + r_min * np.cos(rad))
                y_min = int(center[1] + r_min * np.sin(rad))
                x_max = int(center[0] + r_max * np.cos(rad))
                y_max = int(center[1] + r_max * np.sin(rad))
                x_med = int(center[0] + r_med * np.cos(rad))
                y_med = int(center[1] + r_med * np.sin(rad))

                cv2.line(img, (x_min, y_min), (x_max, y_max), color_sec, 4, cv2.LINE_AA)
                cv2.circle(img, (x_med, y_med), 5, color_main, -1, cv2.LINE_AA)

        cv2.circle(img, center, 34, self.ROBOT_BODY, -1, cv2.LINE_AA)
        cv2.circle(img, center, 36, (100, 105, 115), 2, cv2.LINE_AA)
        cv2.line(img, (center[0], center[1] - 34), (center[0], center[1] - 18), (255, 255, 255), 2, cv2.LINE_AA)

    def draw_wheels(self, img, left, right, max_vel, center=(230, 520)):
        cv2.putText(img, "Wheels", (100, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.putText(img, "L", (165, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.putText(img, "R", (265, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)

        cv2.line(img, (140, center[1]), (290, center[1]), self.PANEL_BORDER, 2, cv2.LINE_AA)
        max_h = 40
        max_vel = max(max_vel, 0.001)

        l_h = int(np.clip((left / max_vel) * max_h, -max_h, max_h))
        r_h = int(np.clip((right / max_vel) * max_h, -max_h, max_h))

        cv2.rectangle(img, (150, center[1] - max_h), (190, center[1] + max_h), self.tertiary_color, -1)
        cv2.rectangle(img, (250, center[1] - max_h), (290, center[1] + max_h), self.tertiary_color, -1)
        cv2.rectangle(img, (150, center[1] - l_h), (190, center[1]), self.primary_color, -1)
        cv2.rectangle(img, (250, center[1] - r_h), (290, center[1]), self.primary_color, -1)

    def draw_grids(self, img, camera_data, is_live=True, active_ganglion=None, active_cones=None, origin=(520, 120)):
        rows, cols = 120 // 20, 160 // 20
        cell_size = 210 // rows

        rf_start_x, rf_start_y = origin[0], origin[1]
        c_start_x, c_start_y = origin[0], origin[1] + 260

        cv2.putText(img, "Retinal Ganglion Cells", (rf_start_x, rf_start_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.putText(img, "Cone Cells", (c_start_x, c_start_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.TEXT_MAIN, 2, cv2.LINE_AA)

        for r in range(rows):
            for c in range(cols):
                cell_idx = r * cols + c
                rf_x = rf_start_x + (c * cell_size)
                rf_y = rf_start_y + (r * cell_size)
                cp_x = c_start_x + (c * cell_size)
                cp_y = c_start_y + (r * cell_size)

                ganglion_active = True if active_ganglion is None else active_ganglion[cell_idx]
                cone_active = True if active_cones is None else active_cones[cell_idx]

                if is_live:
                    idx = cell_idx * 5
                    on_val, off_val, red, green, blue = camera_data[idx: idx + 5]

                    if ganglion_active:
                        if on_val > 0.5 and off_val <= 0.5:
                            rf_color = self.primary_color
                        elif off_val > 0.5 and on_val <= 0.5:
                            rf_color = self.secondary_color
                        else:
                            rf_color = self.tertiary_color
                        cv2.rectangle(img, (rf_x, rf_y), (rf_x + cell_size - 2, rf_y + cell_size - 2), rf_color, -1)
                    else:
                        cv2.rectangle(img, (rf_x, rf_y), (rf_x + cell_size - 2, rf_y + cell_size - 2), self.PANEL_BORDER, 1)

                    if cone_active:
                        if max(red, green, blue) <= 1.0:
                            red, green, blue = red * 255.0, green * 255.0, blue * 255.0
                        cp_color = (int(np.clip(blue, 0, 255)), int(np.clip(green, 0, 255)), int(np.clip(red, 0, 255)))
                        cv2.rectangle(img, (cp_x, cp_y), (cp_x + cell_size - 2, cp_y + cell_size - 2), cp_color, -1)
                    else:
                        cv2.rectangle(img, (cp_x, cp_y), (cp_x + cell_size - 2, cp_y + cell_size - 2), self.PANEL_BORDER, 1)

                else:
                    ganglion_medians, cone_medians = camera_data
                    if ganglion_active:
                        idx = cell_idx * 2
                        on_med = ganglion_medians[idx]
                        off_med = ganglion_medians[idx + 1]

                        if on_med > 0.5 and off_med <= 0.5:
                            rf_color = self.primary_color
                        elif off_med > 0.5 and on_med <= 0.5:
                            rf_color = self.secondary_color
                        else:
                            rf_color = self.tertiary_color
                        cv2.rectangle(img, (rf_x, rf_y), (rf_x + cell_size - 2, rf_y + cell_size - 2), rf_color, -1)
                    else:
                        cv2.rectangle(img, (rf_x, rf_y), (rf_x + cell_size - 2, rf_y + cell_size - 2), self.PANEL_BORDER, 1)

                    if cone_active:
                        median_bgr = cone_medians[cell_idx]
                        cv2.rectangle(img, (cp_x, cp_y), (cp_x + cell_size - 2, cp_y + cell_size - 2), median_bgr, -1)
                    else:
                        cv2.rectangle(img, (cp_x, cp_y), (cp_x + cell_size - 2, cp_y + cell_size - 2), self.PANEL_BORDER, 1)

    def draw_transition_graph(self, img, graph_data):
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        if not nodes:
            cv2.putText(img, "No graph data", (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, self.neutral_gray, 2)
            return

        ordered_nodes = [n for n in SKILL_ORDER if n in nodes]
        for n in nodes:
            if n not in ordered_nodes:
                ordered_nodes.append(n)

        node_indices = {node: i for i, node in enumerate(ordered_nodes)}
        n_nodes = len(ordered_nodes)

        h, w = img.shape[:2]
        node_radius = 42
        y_center = h // 2
        margin_x = 80

        if n_nodes > 1:
            step_x = (w - 2 * margin_x) / (n_nodes - 1)
            positions = {node: (int(margin_x + i * step_x), y_center) for i, node in enumerate(ordered_nodes)}
        else:
            positions = {ordered_nodes[0]: (w // 2, y_center)}

        thickness = 2
        max_weight = 1.0
        min_weight = 0.0

        for edge_info in edges:
            source = edge_info.get("source")
            target = edge_info.get("target")
            weight = edge_info.get("weight", 1)

            if weight is None or not isinstance(weight, (int, float)) or math.isnan(weight) or math.isinf(weight):
                weight = 0.0

            if source not in positions or target not in positions:
                continue

            pt1 = positions[source]
            pt2 = positions[target]
            idx_src = node_indices[source]
            idx_tgt = node_indices[target]

            t = max(0.0, min(1.0, (weight - min_weight) / (max_weight - min_weight))) if max_weight != min_weight else 1.0
            faint_color = tuple(int(0.85 * bg + 0.15 * pri) for bg, pri in zip(self.BG_COLOR, self.primary_color))
            edge_color = tuple(int((1.0 - t) * faint_color[c] + t * self.primary_color[c]) for c in range(3))

            if source == target:
                P1 = np.array([pt1[0] - 12.0, pt1[1] - node_radius], dtype=float)
                P2 = np.array([pt1[0] + 12.0, pt1[1] - node_radius], dtype=float)
                C = np.array([pt1[0], pt1[1] - node_radius - 65.0], dtype=float)
                t_samples = np.linspace(0, 1, 25).reshape(-1, 1)
                curve_pts = ((1 - t_samples) ** 2 * P1 + 2 * (1 - t_samples) * t_samples * C + t_samples**2 * P2).astype(np.int32)
                cv2.polylines(img, [curve_pts], False, edge_color, thickness, cv2.LINE_AA)
                continue

            is_forward = idx_src < idx_tgt
            y_direction = -1.0 if is_forward else 1.0

            P1 = np.array([pt1[0], pt1[1] + (y_direction * node_radius)], dtype=float)
            P2 = np.array([pt2[0], pt2[1] + (y_direction * node_radius)], dtype=float)

            node_step_diff = abs(idx_tgt - idx_src)
            curve_height = 45.0 + (node_step_diff - 1) * 26.0

            M = (P1 + P2) / 2.0
            C = np.array([M[0], M[1] + (y_direction * curve_height)])

            t_samples = np.linspace(0, 1, 35).reshape(-1, 1)
            curve_pts = ((1 - t_samples) ** 2 * P1 + 2 * (1 - t_samples) * t_samples * C + t_samples**2 * P2).astype(np.int32)
            cv2.polylines(img, [curve_pts], False, edge_color, thickness, cv2.LINE_AA)

            tangent = P2 - C
            t_len = math.hypot(tangent[0], tangent[1])
            if t_len > 0:
                tx, ty = tangent[0] / t_len, tangent[1] / t_len
                arrow_start = (int(P2[0] - tx * 16), int(P2[1] - ty * 16))
                arrow_end = (int(P2[0]), int(P2[1]))
                cv2.arrowedLine(img, arrow_start, arrow_end, edge_color, thickness, tipLength=0.45, line_type=cv2.LINE_AA)

            apex = (P1 + 2 * C + P2) / 4.0
            label_pos_y = apex[1] + (y_direction * 14.0)
            label_text = f"{weight:.2f}" if isinstance(weight, float) else str(weight)
            font_lbl = cv2.FONT_HERSHEY_SIMPLEX
            (lbl_w, lbl_h), _ = cv2.getTextSize(label_text, font_lbl, 0.38, 1)

            lbl_x = int(apex[0] - lbl_w / 2)
            lbl_y = int(label_pos_y + (lbl_h / 2 if y_direction > 0 else -lbl_h / 2))
            cv2.putText(img, label_text, (lbl_x, lbl_y), font_lbl, 0.38, self.TEXT_MAIN, 1, cv2.LINE_AA)

        for node, (x, y) in positions.items():
            cv2.circle(img, (x, y), node_radius, self.tertiary_color, -1)
            cv2.circle(img, (x, y), node_radius, self.primary_color, 2)

            display_name = node.replace("_", " ")
            wrapped_lines = textwrap.wrap(display_name, width=9)

            font = cv2.FONT_HERSHEY_SIMPLEX
            line_metrics = [(*cv2.getTextSize(line, font, 0.36, 1), line) for line in wrapped_lines]
            total_height = sum(th + b for (tw, th), b, line in line_metrics) + max(0, len(wrapped_lines) - 1) * 3
            current_y = y - total_height // 2

            for (tw, th), baseline, line in line_metrics:
                cv2.putText(img, line, (int(x - tw / 2), int(current_y + th)), font, 0.36, self.TEXT_MAIN, 1, cv2.LINE_AA)
                current_y += th + baseline + 3

    def draw_positive_GNG_plane(self, img, G_pos, pos_colors, legend_labels):
        """Topological plane visualizing strictly positive GNG manifolds."""
        if len(G_pos.nodes) == 0:
            return

        pos = nx.spring_layout(G_pos, seed=42, k=0.25)
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        raw_w = max_x - min_x if max_x > min_x else 1.0
        raw_h = max_y - min_y if max_y > min_y else 1.0

        max_r_px = max([max(int(G_pos.nodes[n].get("radius", 0.0) * 35), 5) for n in G_pos.nodes()], default=5)
        pad_left = max_r_px + 30
        pad_right = max_r_px + 230
        pad_y_top = max_r_px + 100
        pad_y_bot = max_r_px + 30

        avail_w = 1000 - (pad_left + pad_right)
        avail_h = 620 - (pad_y_top + pad_y_bot)
        scale = min(avail_w / raw_w, avail_h / raw_h)

        center_x = pad_left + avail_w / 2 - ((min_x + max_x) / 2) * scale
        center_y = pad_y_top + avail_h / 2 - ((min_y + max_y) / 2) * scale

        # Positive manifold envelopes
        for node in G_pos.nodes():
            pt = (int(center_x + pos[node][0] * scale), int(center_y + pos[node][1] * scale))
            base_color = pos_colors.get(node, self.neutral_gray)
            light_color = tuple(int(0.75 * bg + 0.25 * bc) for bg, bc in zip(self.BG_COLOR, base_color))
            r_val = G_pos.nodes[node].get("radius", 0.0)
            if r_val > 0:
                r_px = max(int(r_val * 35), 5)
                cv2.circle(img, pt, r_px, light_color, -1, cv2.LINE_AA)

        # Edges
        for u, v in G_pos.edges():
            pt1 = (int(center_x + pos[u][0] * scale), int(center_y + pos[u][1] * scale))
            pt2 = (int(center_x + pos[v][0] * scale), int(center_y + pos[v][1] * scale))
            cv2.line(img, pt1, pt2, pos_colors.get(u, self.neutral_gray), 1, cv2.LINE_AA)

        # Prototype points
        for node in G_pos.nodes():
            pt = (int(center_x + pos[node][0] * scale), int(center_y + pos[node][1] * scale))
            cv2.circle(img, pt, 5, pos_colors.get(node, self.neutral_gray), -1, cv2.LINE_AA)
            cv2.circle(img, pt, 5, self.TEXT_MAIN, 1, cv2.LINE_AA)

        # Legend
        leg_x = 720
        leg_y = 110
        cv2.putText(img, "Positive Manifolds (TRUE):", (leg_x, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.TEXT_MAIN, 1, cv2.LINE_AA)
        for i, (pred_name, col) in enumerate(legend_labels.items()):
            y_pos = leg_y + 25 + (i * 20)
            if y_pos > 600:
                break
            cv2.circle(img, (leg_x + 10, y_pos - 4), 5, col, -1, cv2.LINE_AA)
            cv2.putText(img, pred_name[:24], (leg_x + 25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.TEXT_MAIN, 1, cv2.LINE_AA)

    def draw_frequency_plot(self, img, timestamps, bin_size_sec=60.0):
        if not timestamps or len(timestamps) < 2:
            cv2.putText(img, "Insufficient timestamp data", (350, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.neutral_gray, 2, cv2.LINE_AA)
            return

        ts = np.sort(np.array(timestamps, dtype=float))
        t_start, t_end = ts[0], ts[-1]
        duration = max(t_end - t_start, 1.0)

        n_bins = max(int(np.ceil(duration / bin_size_sec)), 1)
        bin_edges = np.linspace(t_start, t_start + n_bins * bin_size_sec, n_bins + 1)
        counts, _ = np.histogram(ts, bins=bin_edges)

        plot_x, plot_y = 90, 140
        plot_w, plot_h = 820, 380
        cv2.rectangle(img, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), self.PANEL_BORDER, 1)

        avg_freq = len(ts) / (duration / 60.0)
        stat_text = f"Total: {len(ts)} updates | Duration: {duration:.1f}s | Avg Rate: {avg_freq:.2f} updates/min"
        cv2.putText(img, stat_text, (plot_x, plot_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_MAIN, 1, cv2.LINE_AA)

        max_count = max(int(np.max(counts)), 1)
        grid_steps = 4
        for i in range(grid_steps + 1):
            y_val = plot_y + plot_h - int(i * (plot_h / grid_steps))
            val_label = f"{int(i * (max_count / grid_steps))}"
            cv2.line(img, (plot_x, y_val), (plot_x + plot_w, y_val), self.tertiary_color, 1, cv2.LINE_AA)
            cv2.putText(img, val_label, (plot_x - 35, y_val + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.neutral_gray, 1, cv2.LINE_AA)

        bar_w = max(int(plot_w / n_bins) - 3, 2)
        curve_pts = []

        for i, count in enumerate(counts):
            bar_h = int((count / max_count) * (plot_h - 20))
            bx = plot_x + int(i * (plot_w / n_bins)) + 2
            by = plot_y + plot_h - bar_h

            cv2.rectangle(img, (bx, by), (bx + bar_w, plot_y + plot_h), self.secondary_color, -1)
            cv2.rectangle(img, (bx, by), (bx + bar_w, plot_y + plot_h), self.primary_color, 1)
            curve_pts.append((bx + bar_w // 2, by))

        if len(curve_pts) > 1:
            for i in range(len(curve_pts) - 1):
                cv2.line(img, curve_pts[i], curve_pts[i + 1], self.BORDER_COLOR, 2, cv2.LINE_AA)
            for pt in curve_pts:
                cv2.circle(img, pt, 3, self.primary_color, -1, cv2.LINE_AA)

        x_label = f"Elapsed Time (Bins of {int(bin_size_sec)}s) ->"
        cv2.putText(img, x_label, (plot_x + plot_w // 2 - 100, plot_y + plot_h + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_MAIN, 1, cv2.LINE_AA)

    def draw_skill_table(self, img, table_data, threshold=0.75):
        """
        Renders an OpenCV window table:
          Columns: Skills in sequence (ordered transition graph)
          Row 1: Next Action (e.g. Start -> Find_the_Door)
          Row 2: Overlap between Eff(i) and Init(i+1) (green if >= threshold)
          Row 3: Number of nodes in pos_gng
          Row 4: Number of nodes in neg_gng
        """
        table_x, table_y = 40, 115
        total_w, total_h = 1120, 440
        row_height = 80
        header_col_w = 210

        skills = table_data.get("skills", [])
        n_cols = len(skills)
        if n_cols == 0:
            cv2.putText(img, "No skill data available", (table_x + 100, table_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.neutral_gray, 2)
            return

        col_w = int((total_w - header_col_w) / n_cols)
        actual_total_w = header_col_w + n_cols * col_w

        # Outer border
        cv2.rectangle(img, (table_x, table_y), (table_x + actual_total_w, table_y + total_h), self.BORDER_COLOR, 2)

        # Header background
        header_h = 40
        cv2.rectangle(img, (table_x, table_y), (table_x + actual_total_w, table_y + header_h), self.tertiary_color, -1)
        cv2.line(img, (table_x, table_y + header_h), (table_x + actual_total_w, table_y + header_h), self.BORDER_COLOR, 2)

        # Header titles
        cv2.putText(img, "Skill Metric / Property", (table_x + 10, table_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.TEXT_MAIN, 1, cv2.LINE_AA)
        for i, sk in enumerate(skills):
            cx = table_x + header_col_w + i * col_w
            short_name = sk.replace("_", " ")
            lines = textwrap.wrap(short_name, width=12)
            for li, line in enumerate(lines[:2]):
                cv2.putText(img, line, (cx + 8, table_y + 16 + li * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.TEXT_MAIN, 1, cv2.LINE_AA)

        row_labels = [
            "1. Next Action",
            f"2. Overlap (Eff -> Init) [>={threshold:.2f}]",
            "3. Nodes in positive GNG",
            "4. Nodes in negative GNG"
        ]

        curr_y = table_y + header_h
        for r_idx, label in enumerate(row_labels):
            y1 = curr_y + r_idx * row_height
            y2 = y1 + row_height

            # Row divider
            cv2.line(img, (table_x, y2), (table_x + actual_total_w, y2), self.PANEL_BORDER, 1)

            # Row label
            lines = textwrap.wrap(str(label), width=25)
            for li, l_text in enumerate(lines[:3]):
                cv2.putText(img, l_text, (table_x + 6, y1 + 25 + li * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_MAIN, 1, cv2.LINE_AA)

            for c_idx, sk in enumerate(skills):
                cx1 = table_x + header_col_w + c_idx * col_w
                cx2 = cx1 + col_w
                center_cell_x = cx1 + 10
                center_cell_y = y1 + 45

                if r_idx == 0:
                    action_val = table_data.get("next_action", {}).get(sk, "-")
                    lines = textwrap.wrap(str(action_val.replace("_", " ")), width=15)
                    for li, l_text in enumerate(lines[:3]):
                        cv2.putText(img, l_text, (cx1 + 6, y1 + 25 + li * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_MAIN, 1, cv2.LINE_AA)

                elif r_idx == 1:
                    overlap_val = table_data.get("overlap", {}).get(sk, None)
                    if overlap_val is not None and not math.isnan(overlap_val):
                        is_pass = (overlap_val >= threshold)
                        text_val = f"{overlap_val * 100:.1f}%"
                        bg_c = (215, 245, 215) if is_pass else (230, 230, 245)
                        fg_c = self.status_green if is_pass else self.status_red

                        cv2.rectangle(img, (cx1 + 4, y1 + 15), (cx2 - 4, y2 - 15), bg_c, -1)
                        cv2.rectangle(img, (cx1 + 4, y1 + 15), (cx2 - 4, y2 - 15), fg_c, 1)
                        cv2.putText(img, text_val, (cx1 + 14, center_cell_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_MAIN, 1, cv2.LINE_AA)
                    else:
                        cv2.putText(img, "N/A", (center_cell_x, center_cell_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_MAIN, 1, cv2.LINE_AA)

                elif r_idx == 2:
                    pos_count = table_data.get("pos_nodes", {}).get(sk, 0)
                    cv2.putText(img, str(pos_count), (center_cell_x, center_cell_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_MAIN, 1, cv2.LINE_AA)

                elif r_idx == 3:
                    neg_count = table_data.get("neg_nodes", {}).get(sk, 0)
                    cv2.putText(img, str(neg_count), (center_cell_x, center_cell_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT_MAIN, 1, cv2.LINE_AA)

        # Vertical column grid lines
        cv2.line(img, (table_x + header_col_w, table_y), (table_x + header_col_w, table_y + total_h), self.BORDER_COLOR, 2)
        for i in range(1, n_cols):
            vx = table_x + header_col_w + i * col_w
            cv2.line(img, (vx, table_y), (vx, table_y + total_h), self.PANEL_BORDER, 1)


# =============================================================================
# DATA EXTRACTION UTILITY FOR TRANSITION & SKILL TABLE
# =============================================================================

def collect_skill_table_data(latest_session):
    if not latest_session:
        return None

    graphs_dir = os.path.join(latest_session, "graphs")
    symbols_dir = os.path.join(latest_session, "PDDL", "GNG")
    trans_graph_path = os.path.join(graphs_dir, "trans_graph.json")

    skills_present = []
    if os.path.exists(trans_graph_path):
        try:
            with open(trans_graph_path, "r") as f:
                tg = json.load(f)
            nodes = tg.get("nodes", [])
            skills_present = [s for s in SKILL_ORDER if s in nodes]
            for s in nodes:
                if s not in skills_present:
                    skills_present.append(s)
        except Exception:
            pass

    if not skills_present:
        skills_present = list(SKILL_ORDER)

    next_actions = {}
    overlaps = {}
    pos_nodes = {}
    neg_nodes = {}

    for i, sk in enumerate(skills_present):
        # 1. Next action & dynamic overlap calculation
        if i < len(skills_present) - 1:
            nxt = skills_present[i + 1]
            next_actions[sk] = nxt
            
            # Compute directly using static predicate deserializer and overlap logic
            eff_path = os.path.join(symbols_dir, f"{sk}_eff.json")
            init_path = os.path.join(symbols_dir, f"{nxt}_init.json")
            overlaps[sk] = calculate_predicate_overlap(eff_path, init_path)
        else:
            next_actions[sk] = "-"
            overlaps[sk] = None

        # 2. Prototype node counts (init predicate of this skill)
        init_json = os.path.join(symbols_dir, f"{sk}_init.json")
        p_count = 0
        n_count = 0
        if os.path.exists(init_json):
            try:
                with open(init_json, "r") as f:
                    data = json.load(f)
                pos_d, neg_d = parse_predicate_json(data)
                if pos_d and "nodes" in pos_d:
                    p_count = len(pos_d["nodes"])
                if neg_d and "nodes" in neg_d:
                    n_count = len(neg_d["nodes"])
            except Exception:
                pass
        pos_nodes[sk] = p_count
        neg_nodes[sk] = n_count

    return {
        "skills": skills_present,
        "next_action": next_actions,
        "overlap": overlaps,
        "pos_nodes": pos_nodes,
        "neg_nodes": neg_nodes
    }


# =============================================================================
# PICTURE GENERATORS (STATIC)
# =============================================================================

class PictureGenerator:
    def __init__(self, logs_root="logs", session_dir=None):
        self.logs_root = logs_root
        self.session_dir = session_dir
        self.renderer = Renderer()

    def get_pictures_dir(self):
        latest_session = get_session(self.logs_root, self.session_dir)
        if not latest_session:
            return None
        out_dir = os.path.join(latest_session, "pictures")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def run(self):
        raise NotImplementedError

    def _resolve_trans_weight(self, latest_session, filename):
        name = os.path.splitext(os.path.basename(filename))[0]
        match = re.fullmatch(r"sym_(.+?)_enables_(.+?)_pass", name)
        if not match:
            return None
        from_name, to_name = match.groups()
        symbols_dir = os.path.join(latest_session, "PDDL", "GNG")
        eff_path = os.path.join(symbols_dir, f"{from_name}_eff.json")
        init_path = os.path.join(symbols_dir, f"{to_name}_init.json")
        return calculate_predicate_overlap(eff_path, init_path)

class GNGPictureGenerator(PictureGenerator):
    """
    Discriminative Invariants Representation:
    Visualises positive GNG nodes only. Filters dimensions that remain safe/separable
    from the negative GNG space and have low positive internal variance.
    """
    MAX_ALLOWED_INTERNAL_VAR = 0.035
    MAX_INTERVAL_OVERLAP_RATIO = 0.20
    CIRCULAR_VAR_THRESHOLD = 0.12

    def _format_display_name(self, filename):
        name = os.path.splitext(os.path.basename(filename))[0]
        match = re.fullmatch(r"sym_(.+?)_enables_(.+?)_pass", name)
        if match:
            from_name, to_name = match.groups()
            return f"{from_name.replace('_', ' ')} -> {to_name.replace('_', ' ')}"
        if name.endswith("_init"):
            return name[:-5].replace("_", " ") + " - Precondition"
        if name.endswith("_eff"):
            return name[:-4].replace("_", " ") + " - Effect"
        if name.endswith("_pass"):
            return name[:-5].replace("_", " ") + " - Transition"
        return name.replace("_", " ").strip()

    def _is_dimension_invariant(self, pos_col, neg_col=None):
        pos_var = float(np.var(pos_col))
        if pos_var > self.MAX_ALLOWED_INTERNAL_VAR:
            return False

        if neg_col is None or len(neg_col) == 0:
            return True

        p_min, p_max = float(np.min(pos_col)), float(np.max(pos_col))
        n_min, n_max = float(np.min(neg_col)), float(np.max(neg_col))

        pos_width = max(p_max - p_min, 1e-6)
        overlap_min = max(p_min, n_min)
        overlap_max = min(p_max, n_max)

        if overlap_max <= overlap_min:
            return True

        overlap_width = overlap_max - overlap_min
        overlap_fraction = overlap_width / pos_width
        return overlap_fraction < self.MAX_INTERVAL_OVERLAP_RATIO

    def run(self):
        latest_session = get_session(self.logs_root, self.session_dir)
        if not latest_session:
            return
        symbols_dir = os.path.join(latest_session, "PDDL", "GNG")
        if not os.path.exists(symbols_dir):
            return

        # Load action masks from statistics
        stats_path = os.path.join(latest_session, "dimension_statistics.txt")
        if not os.path.exists(stats_path):
            stats_path = "dimension_statistics.txt"
        action_masks = load_action_masks(stats_path)

        # Match regular .json as well as _abs.json
        for json_path in glob.glob(os.path.join(symbols_dir, "*.json")):
            try:
                with open(json_path, 'r') as f:
                    raw_data = json.load(f)

                pos_data, neg_data = parse_predicate_json(raw_data)
                nodes_pos = np.array(pos_data.get("nodes", [])) if pos_data else np.array([])
                nodes_neg = np.array(neg_data.get("nodes", [])) if neg_data else np.array([])

                if len(nodes_pos) == 0:
                    continue

                min_vals = np.min(nodes_pos, axis=0)
                max_vals = np.max(nodes_pos, axis=0)
                median_vals = lower_median(nodes_pos, axis=0)

                file_base = os.path.splitext(os.path.basename(json_path))[0]
                is_abs = file_base.endswith("_abs")

                # Determine active action mask for _abs files
                action_mask = None
                if is_abs:
                    # e.g., 'Find_the_Door_effect_abs' -> skill 'Find_the_Door'
                    # e.g., 'Find_the_Door_abs' -> skill 'Find_the_Door'
                    clean_name = file_base[:-4]  # remove '_abs'
                    for suffix in ["_effect", "_eff", "_init", "_pre"]:
                        if clean_name.endswith(suffix):
                            clean_name = clean_name[:-len(suffix)]
                            break
                    action_mask = action_masks.get(clean_name, None)

                display_title = self._format_display_name(json_path)
                if is_abs:
                    display_title += " (Abstracted Mask)"

                img = self.renderer.create_base_canvas(
                    display_title,
                    "Action Mask Invariants (Covered Dimensions)" if is_abs 
                    else "Positive GNG Invariants (Protected from Negative Domain)"
                )

                # 1. Lidars (Dimensions 0 to 7)
                lidar_active = []
                for i in range(8):
                    if action_mask is not None and i not in action_mask:
                        lidar_active.append(False)
                        continue
                    pos_c = nodes_pos[:, i]
                    neg_c = nodes_neg[:, i] if len(nodes_neg) > 0 else None
                    is_inv = self._is_dimension_invariant(pos_c, neg_c)
                    lidar_active.append(is_inv)

                lidar_data = list(zip(min_vals[:8], max_vals[:8], median_vals[:8]))
                self.renderer.draw_lidar(img, lidar_data, mode="categorical", active_flags=lidar_active)

                # 2. Retinal Ganglion Cells (Dimensions: on=8+c*5, off=9+c*5)
                ganglion_medians = []
                active_ganglion = []
                for cell_idx in range(48):
                    idx_on = 8 + (cell_idx * 5)
                    idx_off = idx_on + 1

                    if action_mask is not None and (idx_on not in action_mask and idx_off not in action_mask):
                        active_ganglion.append(False)
                    else:
                        pos_on, pos_off = nodes_pos[:, idx_on], nodes_pos[:, idx_off]
                        neg_on = nodes_neg[:, idx_on] if len(nodes_neg) > 0 else None
                        neg_off = nodes_neg[:, idx_off] if len(nodes_neg) > 0 else None

                        inv_on = self._is_dimension_invariant(pos_on, neg_on)
                        inv_off = self._is_dimension_invariant(pos_off, neg_off)
                        active_ganglion.append(bool(inv_on and inv_off))

                    ganglion_medians.append(median_vals[idx_on])
                    ganglion_medians.append(median_vals[idx_off])

                # 3. Cone Cells (Dimensions: R=10+c*5, G=11+c*5, B=12+c*5)
                cone_medians = []
                active_cones = []
                for cell_idx in range(48):
                    offset = 8 + (cell_idx * 5)
                    r_idx, g_idx, b_idx = offset + 2, offset + 3, offset + 4
                    r_p, g_p, b_p = nodes_pos[:, r_idx], nodes_pos[:, g_idx], nodes_pos[:, b_idx]
                    median_bgr, hue_var_pos = self.renderer.compute_hue_median_stats(r_p, g_p, b_p)
                    cone_medians.append(median_bgr)

                    if action_mask is not None and not any(dim in action_mask for dim in (r_idx, g_idx, b_idx)):
                        active_cones.append(False)
                    else:
                        is_pos_stable = hue_var_pos < self.CIRCULAR_VAR_THRESHOLD
                        separable = True
                        if len(nodes_neg) > 0:
                            for ch in range(3):
                                if not self._is_dimension_invariant(nodes_pos[:, offset + 2 + ch], nodes_neg[:, offset + 2 + ch]):
                                    separable = False
                                    break
                        active_cones.append(bool(is_pos_stable and separable))

                self.renderer.draw_grids(img, (ganglion_medians, cone_medians), is_live=False,
                                         active_ganglion=active_ganglion, active_cones=active_cones)

                overlap_weight = self._resolve_trans_weight(latest_session, json_path)
                self.renderer.draw_sample_badge(img, n_pos=len(nodes_pos), position=(35, 595), overlap_ratio=overlap_weight)

                out_dir = self.get_pictures_dir()
                if not out_dir:
                    return
                out_path = os.path.join(out_dir, f"{file_base}_invariants.png")
                cv2.imwrite(out_path, img)
                print(f"[PictureGenerator] Saved positive GNG invariants to {out_path}")
            except Exception as e:
                print(f"[PictureGenerator] Failed to process invariants for {json_path}: {e}")

class RepresentativePointPictureGenerator(PictureGenerator):
    """
    Iconic Representation: Visualises ONLY positive GNG prototypes.
    Takes the point closest to the center of the positive manifold.
    """
    def _format_display_name(self, filename):
        name = os.path.splitext(os.path.basename(filename))[0]
        match = re.fullmatch(r"sym_(.+?)_enables_(.+?)_pass", name)
        if match:
            from_name, to_name = match.groups()
            return f"{from_name.replace('_', ' ')} -> {to_name.replace('_', ' ')}"
        if name.endswith("_init"):
            return name[:-5].replace("_", " ") + " - Precondition"
        if name.endswith("_eff"):
            return name[:-4].replace("_", " ") + " - Effect"
        if name.endswith("_pass"):
            return name[:-5].replace("_", " ") + " - Transition"
        if name.endswith("_abs"):
            return name[:-5].replace("_", " ") + " - Abstract Symbol"
        return name.replace("_", " ").strip()

    def run(self):
        latest_session = get_session(self.logs_root, self.session_dir)
        if not latest_session:
            return
        symbols_dir = os.path.join(latest_session, "PDDL", "GNG")
        if not os.path.exists(symbols_dir):
            return
        out_dir = self.get_pictures_dir()
        if not out_dir:
            return

        stats_path = os.path.join(latest_session, "dimension_statistics.txt")
        if not os.path.exists(stats_path):
            stats_path = "dimension_statistics.txt"
        action_masks = load_action_masks(stats_path)

        for json_path in glob.glob(os.path.join(symbols_dir, "*.json")):
            try:
                with open(json_path, "r") as f:
                    raw_data = json.load(f)

                pos_data, _ = parse_predicate_json(raw_data)
                nodes_pos = np.array(pos_data.get("nodes", [])) if pos_data else np.array([])

                if len(nodes_pos) == 0:
                    continue

                centroid = np.mean(nodes_pos, axis=0)
                dists = np.linalg.norm(nodes_pos - centroid, axis=1)
                rep_node = nodes_pos[int(np.argmin(dists))]

                file_base = os.path.splitext(os.path.basename(json_path))[0]
                is_abs = file_base.endswith("_abs")

                # Resolve action mask for _abs files
                action_mask = None
                if is_abs:
                    clean_name = file_base[:-4]
                    for suffix in ["_effect", "_eff", "_init", "_pre"]:
                        if clean_name.endswith(suffix):
                            clean_name = clean_name[:-len(suffix)]
                            break
                    action_mask = action_masks.get(clean_name, None)

                # Active flags based on action mask
                lidar_active = None
                active_ganglion = None
                active_cones = None

                if action_mask is not None:
                    lidar_active = [i in action_mask for i in range(8)]
                    active_ganglion = [
                        (8 + c * 5 in action_mask) or (9 + c * 5 in action_mask)
                        for c in range(48)
                    ]
                    active_cones = [
                        any(8 + c * 5 + ch in action_mask for ch in (2, 3, 4))
                        for c in range(48)
                    ]

                display_title = self._format_display_name(json_path)
                if is_abs:
                    display_title += " (Action Mask Active)"

                overlap_weight = self._resolve_trans_weight(latest_session, json_path)
                img = self.renderer.create_base_canvas(
                    display_title,
                    "Action Mask Covered Dimensions" if is_abs else "Positive GNG Icon: Center Prototype"
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
                self.renderer.draw_sample_badge(img, n_pos=len(nodes_pos), position=(35, 595), overlap_ratio=overlap_weight)

                out_path = os.path.join(out_dir, f"{file_base}_icon.png")
                cv2.imwrite(out_path, img)
                print(f"[PictureGenerator] Saved positive GNG icon to {out_path}")

            except Exception as e:
                print(f"[PictureGenerator] Error processing icon for {json_path}: {e}")

class UpdateFrequencyPictureGenerator(PictureGenerator):
    def __init__(self, logs_root="logs", session_dir=None, timestamp_file="timestamps.txt", bin_size_sec=60.0):
        super().__init__(logs_root, session_dir=session_dir)
        self.timestamp_file = timestamp_file
        self.bin_size_sec = bin_size_sec

    def _resolve_file_path(self):
        latest_session = get_session(self.logs_root, self.session_dir)
        if latest_session:
            sess_path = os.path.join(latest_session, self.timestamp_file)
            if os.path.exists(sess_path):
                return sess_path
        if os.path.exists(self.timestamp_file):
            return self.timestamp_file
        return None

    def run(self):
        file_path = self._resolve_file_path()
        if not file_path:
            return
        try:
            with open(file_path, "r") as f:
                timestamps = [float(line.strip()) for line in f if line.strip()]
            img = self.renderer.create_base_canvas("UPDATE FREQUENCY OVER TIME")
            self.renderer.draw_frequency_plot(img, timestamps, bin_size_sec=self.bin_size_sec)
            self.renderer.draw_sample_badge(img, n_pos=len(timestamps), position=(35, 595))

            out_dir = self.get_pictures_dir() or "."
            out_path = os.path.join(out_dir, "update_frequency.png")
            cv2.imwrite(out_path, img)
            print(f"[PictureGenerator] Saved frequency plot to {out_path}")
        except Exception as e:
            print(f"[PictureGenerator] Failed to generate frequency plot: {e}")

class TransGraphPictureGenerator(PictureGenerator):
    def run(self):
        latest_session = get_session(self.logs_root, self.session_dir)
        if not latest_session:
            return
        graphs_dir = os.path.join(latest_session, "graphs")
        for mode in ('HARD', 'SOFT', 'PDDL'):
            json_path = os.path.join(graphs_dir, f"trans_graph_{mode}.json")
            if not os.path.exists(json_path):
                return

            try:
                with open(json_path, 'r') as f:
                    graph_data = json.load(f)
                img = self.renderer.create_base_canvas("TRANSITION GRAPH", f"{mode} mode")
                self.renderer.draw_transition_graph(img, graph_data)

                out_dir = self.get_pictures_dir()
                if not out_dir:
                    return
                out_path = os.path.join(out_dir, f"trans_graph_{mode}.png")
                cv2.imwrite(out_path, img)
                print(f"[PictureGenerator] Saved Transition Graph image to {out_path}")
            except Exception as e:
                print(f"[PictureGenerator] Failed to generate TransGraph: {e}")

class UnifiedGNGPictureGenerator(PictureGenerator):
    """Draws topological planes for positive GNG manifolds only."""
    def __init__(self, logs_root="logs", pred_type='init', session_dir=None):
        super().__init__(logs_root, session_dir=session_dir)
        self.gng_colors = {}
        self.pred_type = pred_type
        self.color_palette = [
            (60, 60, 220),   (60, 220, 60),   (220, 100, 60),   (60, 200, 220),
            (200, 60, 200),  (220, 200, 60),  (100, 120, 255),  (255, 120, 100),
            (100, 255, 120), (40, 140, 240),  (180, 50, 130),   (60, 140, 20),
            (160, 130, 240), (150, 210, 240), (50, 90, 180),   (120, 180, 50)
        ]

    def _format_display_name(self, filename):
        name = os.path.splitext(os.path.basename(filename))[0]
        match = re.fullmatch(r"sym_(.+?)_enables_(.+?)_pass", name)
        if match:
            from_name, to_name = match.groups()
            return f"{from_name.replace('_', ' ')} -> {to_name.replace('_', ' ')}"
        if name.endswith("_init"):
            return name[:-5].replace("_", " ")
        if name.endswith("_eff"):
            return name[:-4].replace("_", " ")
        if name.endswith("_pass"):
            return name[:-5].replace("_", " ")
        return name.replace("_", " ").strip()

    def _get_color(self, predicate_name):
        if predicate_name not in self.gng_colors:
            color_idx = len(self.gng_colors) % len(self.color_palette)
            self.gng_colors[predicate_name] = self.color_palette[color_idx]
        return self.gng_colors[predicate_name]

    def run(self):
        latest_session = get_session(self.logs_root, self.session_dir)
        if not latest_session:
            return
        symbols_dir = os.path.join(latest_session, "PDDL", "GNG")
        if not os.path.exists(symbols_dir):
            return

        json_files = glob.glob(os.path.join(symbols_dir, f"*_{self.pred_type}.json"))
        G_pos = nx.Graph()
        pos_colors = {}

        for json_path in json_files:
            try:
                with open(json_path, 'r') as f:
                    raw_data = json.load(f)

                pos_data, _ = parse_predicate_json(raw_data)
                pred_name = self._format_display_name(json_path)
                color = self._get_color(pred_name)

                if pos_data:
                    nodes = pos_data.get("nodes", [])
                    radii = pos_data.get("local_radii", [])
                    for i in range(len(nodes)):
                        node_id = f"{pred_name}_{i}"
                        r_val = radii[i] if i < len(radii) else 0.0
                        G_pos.add_node(node_id, radius=r_val)
                        pos_colors[node_id] = color
                    for edge_str in pos_data.get("edges", {}).keys():
                        u_str, v_str = edge_str.split(",")
                        G_pos.add_edge(f"{pred_name}_{u_str}", f"{pred_name}_{v_str}")
            except Exception as e:
                print(f"[PictureGenerator] Failed to process {json_path}: {e}")

        if len(G_pos.nodes) == 0:
            return

        pred_title = {'eff': 'effects', 'init': 'preconditions', 'pass': 'transitions'}.get(self.pred_type, 'state space')
        img = self.renderer.create_base_canvas("POSITIVE GNG TOPOLOGICAL PLANE", f"Topological manifolds for {pred_title}")
        self.renderer.draw_positive_GNG_plane(img, G_pos, pos_colors, self.gng_colors)

        out_dir = self.get_pictures_dir()
        if not out_dir:
            return
        out_path = os.path.join(out_dir, f"positive_gng_plane_{self.pred_type}.png")
        cv2.imwrite(out_path, img)
        print(f"[PictureGenerator] Saved positive GNG plane to {out_path}")

class SkillTablePictureGenerator(PictureGenerator):
    """Generates an image containing the live/status table across skills in sequence."""
    def __init__(self, logs_root="logs", session_dir=None, threshold=0.75):
        super().__init__(logs_root, session_dir=session_dir)
        self.threshold = threshold

    def run(self):
        latest_session = get_session(self.logs_root, self.session_dir)
        if not latest_session:
            return

        table_data = collect_skill_table_data(latest_session)
        if not table_data:
            return

        img = self.renderer.create_base_canvas(
            "SKILL TRANSITIONS & MANIFOLD METRICS",
            f"Overlap Threshold: {self.threshold * 100:.1f}% | Ordered Transition Flow",
            width=1200,
            height=620
        )
        self.renderer.draw_skill_table(img, table_data, threshold=self.threshold)

        out_dir = self.get_pictures_dir()
        if not out_dir:
            return
        out_path = os.path.join(out_dir, "skills_transition_table.png")
        cv2.imwrite(out_path, img)
        print(f"[PictureGenerator] Saved Skills Transition Table to {out_path}")

class TestSummaryPictureGenerator(PictureGenerator):
    def __init__(self, logs_root="logs", session_dir=None, stats_file="test_statistics.json"):
        super().__init__(logs_root, session_dir=session_dir)
        self.stats_file = stats_file

    def run(self):
        latest_session = get_session(self.logs_root, self.session_dir)
        if not latest_session:
            return
        json_path = os.path.join(latest_session, self.stats_file)
        if not os.path.exists(json_path):
            return

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            overall = data.get("overall_statistics", {})
            img = self.renderer.create_base_canvas("TEST SUMMARY STATISTICS", "Success rates, duration benchmarks, and successful spawn zones")

            sr = overall.get("success_rate", 0.0) * 100.0
            total_tests = overall.get("total_tests", 0)
            avg_dur = overall.get("mean_execution_time_s", 0.0)

            card_y = 110
            cv2.rectangle(img, (45, card_y), (225, card_y + 85), self.renderer.PANEL_BORDER, 1)
            cv2.putText(img, "TOTAL TESTS", (55, card_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.renderer.neutral_gray, 1, cv2.LINE_AA)
            cv2.putText(img, str(total_tests), (55, card_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.renderer.TEXT_MAIN, 2, cv2.LINE_AA)

            cv2.rectangle(img, (250, card_y), (475, card_y + 85), self.renderer.PANEL_BORDER, 1)
            cv2.putText(img, "SUCCESS RATE", (260, card_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.renderer.neutral_gray, 1, cv2.LINE_AA)
            cv2.putText(img, f"{sr:.1f}%", (260, card_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (60, 180, 75), 2, cv2.LINE_AA)

            cv2.rectangle(img, (500, card_y), (725, card_y + 85), self.renderer.PANEL_BORDER, 1)
            cv2.putText(img, "AVG DURATION", (510, card_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.renderer.neutral_gray, 1, cv2.LINE_AA)
            cv2.putText(img, f"{avg_dur:.2f}s", (510, card_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 1.1, self.renderer.TEXT_MAIN, 2, cv2.LINE_AA)

            start_plane = overall.get("acceptable_start_position_plane") or {}
            min_bounds = start_plane.get("min_bounds")
            max_bounds = start_plane.get("max_bounds")
            positions = start_plane.get("successful_positions", [])

            pane_x, pane_y, pane_w, pane_h = 45, 220, 910, 360
            cv2.rectangle(img, (pane_x, pane_y), (pane_x + pane_w, pane_y + pane_h), self.renderer.PANEL_BORDER, 1)
            cv2.putText(img, "ACCEPTABLE STARTING PLANE BOUNDS (Successful Spawns)", (pane_x + 15, pane_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

            if min_bounds and max_bounds and len(min_bounds) >= 2:
                plane_str = f"Area Bounds — X: [{min_bounds[0]:.3f} to {max_bounds[0]:.3f}] | Y: [{min_bounds[1]:.3f} to {max_bounds[1]:.3f}]"
                cv2.putText(img, plane_str, (pane_x + 15, pane_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.renderer.neutral_gray, 1, cv2.LINE_AA)

                draw_x, draw_y, draw_w, draw_h = pane_x + 300, pane_y + 80, 310, 240
                cv2.rectangle(img, (draw_x, draw_y), (draw_x + draw_w, draw_y + draw_h), (35, 35, 35), -1)
                cv2.rectangle(img, (draw_x, draw_y), (draw_x + draw_w, draw_y + draw_h), self.renderer.PANEL_BORDER, 1)

                xs = [p[0] for p in positions]
                ys = [p[1] for p in positions]
                min_x, max_x = min(xs) - 0.5, max(xs) + 0.5
                max_x = max_x if (max_x - min_x) > 0 else min_x + 1.0
                min_y, max_y = min(ys) - 0.5, max(ys) + 0.5
                max_y = max_y if (max_y - min_y) > 0 else min_y + 1.0

                for p in positions:
                    px = int(draw_x + ((p[0] - min_x) / (max_x - min_x)) * draw_w)
                    py = int(draw_y + draw_h - ((p[1] - min_y) / (max_y - min_y)) * draw_h)
                    cv2.circle(img, (px, py), 4, (60, 180, 75), -1)

                cv2.putText(img, f"Min X: {min_bounds[0]:.2f}", (draw_x - 75, draw_y + draw_h), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
                cv2.putText(img, f"Max X: {max_bounds[0]:.2f}", (draw_x + draw_w + 10, draw_y + draw_h), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
                cv2.putText(img, f"Max Y: {max_bounds[1]:.2f}", (draw_x, draw_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

            out_dir = self.get_pictures_dir()
            if out_dir:
                cv2.imwrite(os.path.join(out_dir, "test_summary.png"), img)
        except Exception as e:
            print(f"[PictureGenerator] Failed to generate test summary picture: {e}")

class TestSequenceFailurePictureGenerator(PictureGenerator):
    def __init__(self, logs_root="logs", session_dir=None, stats_file="test_statistics.json"):
        super().__init__(logs_root, session_dir=session_dir)
        self.stats_file = stats_file
        self.success_color = (60, 180, 75)
        self.fail_color = (60, 60, 220)

    def run(self):
        latest_session = get_session(self.logs_root, self.session_dir)
        if not latest_session:
            return
        json_path = os.path.join(latest_session, self.stats_file)
        if not os.path.exists(json_path):
            return

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            img = self.renderer.create_base_canvas("MOST POPULAR PLANS & FAILURE TRANSITIONS", "Frequent path success rates vs exact break points")

            panel_l_x, panel_y, panel_w, panel_h = 45, 110, 440, 480
            cv2.rectangle(img, (panel_l_x, panel_y), (panel_l_x + panel_w, panel_y + panel_h), self.renderer.PANEL_BORDER, 1)
            cv2.putText(img, "TOP FREQUENCY PLANS", (panel_l_x + 15, panel_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

            plan_freqs = data.get("plan_frequencies", {})
            sorted_plans = sorted(plan_freqs.items(), key=lambda k: k[1]["total"], reverse=True)[:5]

            for idx, (plan_str, counts) in enumerate(sorted_plans):
                curr_y = panel_y + 55 + (idx * 80)
                tot = counts["total"]
                suc = counts["succeeded"]
                failed = counts["failed"]
                disp_str = plan_str[:60] + "..." if len(plan_str) > 60 else plan_str
                cv2.putText(img, f"#{idx+1}: {disp_str}", (panel_l_x + 15, curr_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

                bar_x = panel_l_x + 15
                bar_w = 400
                cv2.rectangle(img, (bar_x, curr_y + 28), (bar_x + bar_w, curr_y + 40), (45, 45, 45), -1)
                if tot > 0:
                    split = int(bar_w * (suc / tot))
                    cv2.rectangle(img, (bar_x, curr_y + 28), (bar_x + split, curr_y + 40), self.success_color, -1)
                    cv2.rectangle(img, (bar_x + split, curr_y + 28), (bar_x + bar_w, curr_y + 40), self.fail_color, -1)

                cv2.putText(img, f"Attempts: {tot}  |  Succeeded: {suc}  |  Failed: {failed}",
                            (panel_l_x + 15, curr_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.renderer.neutral_gray, 1, cv2.LINE_AA)

            panel_r_x = 515
            cv2.rectangle(img, (panel_r_x, panel_y), (panel_r_x + panel_w, panel_y + panel_h), self.renderer.PANEL_BORDER, 1)
            cv2.putText(img, "TRANSITION FAILURES (e.g. A -> B)", (panel_r_x + 15, panel_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

            transitions = data.get("transition_outcomes", {})
            failed_transitions = sorted(transitions.items(), key=lambda x: x[1]["failed"], reverse=True)[:10]

            for idx, (trans_key, counts) in enumerate(failed_transitions):
                curr_y = panel_y + 52 + (idx * 41)
                att = counts["attempts"]
                fail = counts["failed"]
                fail_pct = (fail / att * 100) if att > 0 else 0.0
                color_use = self.fail_color if fail > 0 else self.renderer.TEXT_MAIN
                cv2.putText(img, f"{trans_key}"[:52], (panel_r_x + 15, curr_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
                cv2.putText(img, f"Failed: {fail}/{att} ({fail_pct:.0f}%)", (panel_r_x + 295, curr_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color_use, 1, cv2.LINE_AA)
                cv2.line(img, (panel_r_x + 15, curr_y + 26), (panel_r_x + panel_w - 15, curr_y + 26), (40, 40, 40), 1)

            out_dir = self.get_pictures_dir()
            if out_dir:
                cv2.imwrite(os.path.join(out_dir, "test_sequence_failures.png"), img)
        except Exception as e:
            print(f"[PictureGenerator] Failed to generate test sequence failures picture: {e}")

class PipelinePictureGenerator(PictureGenerator):
    def __init__(self, logs_root="logs", session_dir=None):
        super().__init__(logs_root=logs_root, session_dir=session_dir)
        self.generators = [
            GNGPictureGenerator(logs_root, session_dir=session_dir),
            TransGraphPictureGenerator(logs_root, session_dir=session_dir),
            UnifiedGNGPictureGenerator(logs_root, 'init', session_dir=session_dir),
            UnifiedGNGPictureGenerator(logs_root, 'eff', session_dir=session_dir),
            UnifiedGNGPictureGenerator(logs_root, 'pass', session_dir=session_dir),
            UpdateFrequencyPictureGenerator(logs_root, session_dir=session_dir),
            RepresentativePointPictureGenerator(logs_root, session_dir=session_dir),
            SkillTablePictureGenerator(logs_root, session_dir=session_dir),
            TestSummaryPictureGenerator(logs_root, session_dir=session_dir),
            TestSequenceFailurePictureGenerator(logs_root, session_dir=session_dir)
        ]

    def run(self):
        for gen in self.generators:
            gen.run()


# =============================================================================
# LIVE MONITORING
# =============================================================================

class LiveSensimotorMonitor(Renderer):
    def __init__(self, bus: Communicator, timeout: float = 0.1):
        super().__init__()
        self.bus = bus
        self.timeout = timeout
        self.window_name = "Live Sensimotor Monitor"
        self._window_created = False

    def _ensure_window(self):
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self._window_created = True

    def _safe_call(self, topic: str):
        try:
            try:
                return self.bus.call_service(topic, timeout=self.timeout)
            except TypeError:
                return self.bus.call_service(topic)
        except Exception:
            return None

    def update(self):
        self._ensure_window()
        cv2.waitKey(1)

        lidar_readings = []
        for direction in self.LIDAR_ANGLES:
            val = self._safe_call(f"/lidar/{direction}/ask/value")
            if val is None:
                return
            lidar_readings.append(val)

        raw_coded = self._safe_call("/camera/ask/coded")
        if raw_coded is None:
            return
        cam_array = np.asarray(raw_coded, dtype=float).flatten()
        if len(cam_array) < (6 * 8 * 5):
            return

        speeds = self._safe_call("/wheels/ask/speed")
        max_vel = self._safe_call("/get/wheels/max_velocity")
        if speeds is None or max_vel is None:
            return

        img = self.create_base_canvas("LIVE SENSIMOTOR MONITOR")
        self.draw_lidar(img, lidar_readings, mode="live")
        self.draw_wheels(img, speeds[0], speeds[1], float(max_vel))
        self.draw_grids(img, cam_array, is_live=True)
        cv2.imshow(self.window_name, img)

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False

class LiveGNGMonitor(Renderer):
    """Live monitor rendering positive manifolds exclusively."""
    def __init__(self, logs_root="logs"):
        super().__init__()
        self.logs_root = logs_root
        self.window_name = "Live Positive GNG Plane"
        self._window_created = False
        self._last_mtimes = {}
        self.gng_colors = {}
        self.color_palette = [
            (60, 60, 220),   (60, 220, 60),   (220, 100, 60),   (60, 200, 220),
            (200, 60, 200),  (220, 200, 60),  (100, 120, 255),  (255, 120, 100),
            (100, 255, 120), (40, 140, 240),  (180, 50, 130),   (60, 140, 20)
        ]

    def _format_display_name(self, filename):
        name = os.path.splitext(os.path.basename(filename))[0]
        if name.endswith("_init"):
            return name[:-5].replace("_", " ")
        if name.endswith("_eff"):
            return name[:-4].replace("_", " ")
        if name.endswith("_pass"):
            return name[:-5].replace("_", " ")
        return name.replace("_", " ").strip()

    def _ensure_window(self):
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            placeholder = self.create_base_canvas("LIVE POSITIVE GNG PLANE", "Waiting for predicate data...")
            cv2.imshow(self.window_name, placeholder)
            self._window_created = True

    def _get_color(self, label):
        if label not in self.gng_colors:
            color_idx = len(self.gng_colors) % len(self.color_palette)
            self.gng_colors[label] = self.color_palette[color_idx]
        return self.gng_colors[label]

    def _load_json_with_retry(self, path, retries=3, delay=0.05):
        for _ in range(retries):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except (PermissionError, OSError, json.JSONDecodeError):
                time.sleep(delay)
        return None

    def update(self):
        self._ensure_window()
        cv2.waitKey(1)

        latest_session = get_session(self.logs_root, None)
        if not latest_session:
            return
        symbols_dir = os.path.join(latest_session, "PDDL", "GNG")
        if not os.path.exists(symbols_dir):
            return

        json_files = glob.glob(os.path.join(symbols_dir, "*_init.json"))
        if not json_files:
            return

        needs_update = False
        for f in json_files:
            try:
                mtime = os.path.getmtime(f)
                if self._last_mtimes.get(f, 0.0) < mtime:
                    needs_update = True
                    self._last_mtimes[f] = mtime
            except OSError:
                continue

        if not needs_update:
            return

        G_pos = nx.Graph()
        pos_colors = {}

        for json_path in json_files:
            raw_data = self._load_json_with_retry(json_path)
            if raw_data is None:
                continue

            pos_data, _ = parse_predicate_json(raw_data)
            display_name = self._format_display_name(json_path)
            color = self._get_color(display_name)

            if pos_data:
                nodes = pos_data.get("nodes", [])
                radii = pos_data.get("local_radii", [])
                for i in range(len(nodes)):
                    node_id = f"{display_name}_{i}"
                    G_pos.add_node(node_id, radius=radii[i] if i < len(radii) else 0.0)
                    pos_colors[node_id] = color
                for edge_str in pos_data.get("edges", {}).keys():
                    u_str, v_str = edge_str.split(",")
                    G_pos.add_edge(f"{display_name}_{u_str}", f"{display_name}_{v_str}")

        if len(G_pos.nodes) == 0:
            return

        img = self.create_base_canvas("LIVE POSITIVE GNG PLANE", "Real-time Precondition Positive Manifolds")
        self.draw_positive_GNG_plane(img, G_pos, pos_colors, self.gng_colors)
        cv2.imshow(self.window_name, img)

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False

class LiveUpdateFrequencyMonitor(Renderer):
    def __init__(self, logs_root="logs", timestamp_filename="timestamps.txt", bin_size_sec=10.0):
        super().__init__()
        self.logs_root = logs_root
        self.timestamp_filename = timestamp_filename
        self.bin_size_sec = bin_size_sec
        self._last_mtime = 0.0
        self.window_name = "Live Update Frequency Monitor"
        self._window_created = False

    def _ensure_window(self):
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            placeholder = self.create_base_canvas("LIVE UPDATE FREQUENCY", "Waiting for timestamp data...")
            cv2.imshow(self.window_name, placeholder)
            self._window_created = True

    def update(self):
        self._ensure_window()
        cv2.waitKey(1)
        latest_session = get_session(self.logs_root, None)
        if not latest_session:
            return
        times_path = os.path.join(latest_session, "graphs", self.timestamp_filename)
        if not os.path.exists(times_path) or os.path.getsize(times_path) == 0:
            return

        try:
            mtime = os.path.getmtime(times_path)
            if mtime > self._last_mtime:
                with open(times_path, "r") as f:
                    timestamps = [float(line.strip()) for line in f if line.strip()]
                img = self.create_base_canvas("LIVE UPDATE FREQUENCY", f"Tracking: {times_path}")
                self.draw_frequency_plot(img, timestamps, bin_size_sec=self.bin_size_sec)
                self.draw_sample_badge(img, n_pos=len(timestamps), position=(35, 595))
                cv2.imshow(self.window_name, img)
                self._last_mtime = mtime
        except Exception as e:
            print(f"[LiveUpdateFrequencyMonitor] Error loading timestamps: {e}")

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False

class LiveGraphMonitor(Renderer):
    def __init__(self, logs_root="logs", trans_filename="trans_graph_PDDL.json"):
        super().__init__()
        self.logs_root = logs_root
        self.trans_filename = trans_filename
        self._last_trans_mtime = 0.0
        self.trans_window = "Live Transition Graph"
        self._window_created = False

    def _ensure_window(self):
        if not self._window_created:
            cv2.namedWindow(self.trans_window, cv2.WINDOW_AUTOSIZE)
            placeholder = self.create_base_canvas("TRANSITION GRAPH MONITOR", "Waiting for data...")
            cv2.imshow(self.trans_window, placeholder)
            self._window_created = True

    def update(self):
        self._ensure_window()
        cv2.waitKey(1)
        latest_session = get_session(self.logs_root, None)
        if not latest_session:
            return
        trans_path = os.path.join(latest_session, "graphs", self.trans_filename)

        try:
            if os.path.exists(trans_path) and os.path.getsize(trans_path) > 0:
                mtime = os.path.getmtime(trans_path)
                if mtime > self._last_trans_mtime:
                    with open(trans_path, "r") as f:
                        graph_data = json.load(f)
                    img = self.create_base_canvas("LIVE TRANSITION GRAPH", "From PDDL")
                    self.draw_transition_graph(img, graph_data)
                    cv2.imshow(self.trans_window, img)
                    self._last_trans_mtime = mtime
        except Exception as e:
            print(f"Error loading live graph: {e}")

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.trans_window)
            self._window_created = False

class LiveRepresentativePointMonitor(Renderer):
    """Real-time display showing the currently active skill's positive GNG center point."""
    def __init__(self, bus, logs_root="logs"):
        super().__init__()
        self.bus = bus
        self.logs_root = logs_root
        self.window_name = "Live Representative Point Monitor"
        self._window_created = False

    def _ensure_window(self):
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            placeholder = self.create_base_canvas("LIVE REPRESENTATIVE POINT", "Waiting for action query...")
            cv2.imshow(self.window_name, placeholder)
            self._window_created = True

    def _resolve_action_name(self):
        try:
            res = self.bus.call_service("agent/ask/action")
            if isinstance(res, dict):
                return res.get("action") or res.get("name")
            return res
        except Exception:
            return None

    def update(self):
        self._ensure_window()
        cv2.waitKey(1)

        action_name = self._resolve_action_name()
        if not action_name:
            return

        latest_session = get_session(self.logs_root, None)
        if not latest_session:
            return

        action_file = os.path.join(latest_session, "PDDL", "GNG", f"{action_name}_init.json")
        if not os.path.isfile(action_file):
            return

        try:
            with open(action_file, "r") as f:
                raw_data = json.load(f)

            pos_data, _ = parse_predicate_json(raw_data)
            nodes_pos = np.array(pos_data.get("nodes", [])) if pos_data else np.array([])

            if len(nodes_pos) == 0:
                return

            display_title = str(action_name).replace("_", " ")
            img = self.create_base_canvas(display_title, "Current Action Iconic State (Center of Positive GNG)")

            rep_node = nodes_pos[int(np.argmin(np.linalg.norm(nodes_pos - np.mean(nodes_pos, axis=0), axis=1)))]
            self.draw_lidar(img, rep_node[:8], center=(230, 280), mode="iconic", custom_color=self.primary_color)
            self.draw_grids(img, rep_node[8:248], is_live=True, origin=(520, 120))
            self.draw_sample_badge(img, n_pos=len(nodes_pos), position=(35, 595))
            cv2.imshow(self.window_name, img)

        except Exception as e:
            print(f"[LiveRepresentativePointMonitor] Error: {e}")

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False

class LiveSkillTableMonitor(Renderer):
    """
    Real-time OpenCV window rendering a dynamic table:
      Column: each skill in transition graph sequence
      Row 1: next action (e.g. start -> find the door)
      Row 2: overlap between effect and init for transition (green if >= threshold)
      Row 3: number of nodes in pos_gng
      Row 4: number of nodes in neg_gng
    """
    def __init__(self, logs_root="logs", threshold: float = 0.75):
        super().__init__()
        self.logs_root = logs_root
        self.threshold = threshold
        self.window_name = "Live Skill Transitions & Manifolds"
        self._window_created = False
        self._last_mtimes = {}

    def _ensure_window(self):
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            placeholder = self.create_base_canvas(
                "LIVE SKILL TRANSITION MONITOR",
                "Waiting for transition graph and manifold updates...",
                width=1200,
                height=620
            )
            cv2.imshow(self.window_name, placeholder)
            self._window_created = True

    def update(self):
        self._ensure_window()
        cv2.waitKey(1)

        latest_session = get_session(self.logs_root, None)
        if not latest_session:
            return

        graphs_file = os.path.join(latest_session, "graphs", "trans_graph.json")
        gng_files = glob.glob(os.path.join(latest_session, "PDDL", "GNG", "*.json"))

        check_files = [graphs_file] + gng_files if os.path.exists(graphs_file) else gng_files
        needs_refresh = False
        for f in check_files:
            try:
                mt = os.path.getmtime(f)
                if self._last_mtimes.get(f, 0.0) < mt:
                    needs_refresh = True
                    self._last_mtimes[f] = mt
            except OSError:
                continue

        if not needs_refresh:
            return

        table_data = collect_skill_table_data(latest_session)
        if not table_data:
            return

        img = self.create_base_canvas(
            "LIVE SKILL TRANSITIONS & MANIFOLD METRICS",
            f"Monitoring active pipeline | Green Overlap >= {self.threshold * 100:.1f}%",
            width=1200,
            height=620
        )
        self.draw_skill_table(img, table_data, threshold=self.threshold)
        cv2.imshow(self.window_name, img)

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False


if __name__ == "__main__":
    sd = None #"2026-10-16_22-36"
    gen = PipelinePictureGenerator(session_dir=sd)
    #gen = TransGraphPictureGenerator(session_dir=sd)
    gen.run()