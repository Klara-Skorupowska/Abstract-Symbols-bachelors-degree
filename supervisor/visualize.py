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

# =============================================================================
# SHARED RENDERERS & UTILS
# =============================================================================

class Renderer:
    """Shared drawing logic for sensimotor and graph visualizations."""
    
    def __init__(self):
        # Fresh, professional blue palette (BGR format for OpenCV)
        self.primary_color = (195, 105, 30)       # Rich Steel / Slate Blue
        self.secondary_color = (235, 185, 90)     # Sky Blue / Cyan Accent
        self.tertiary_color = (248, 235, 215)     # Pale Ice Blue tint
        
        self.BG_COLOR = (248, 249, 250)
        self.BORDER_COLOR = (40, 44, 52)
        self.TEXT_MAIN = (30, 30, 30)
        self.ROBOT_BODY = (40, 44, 52)
        self.PANEL_BORDER = (210, 215, 220)
        
        self.neutral_gray = (130, 135, 140)
        self.LIDAR_ANGLES = [17, 50, 90, 150, 210, 270, 310, 343]

    def compute_hue_median_stats(self, r_vals, g_vals, b_vals):
        """Computes lower-median hue and circular variance for cone samples."""
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
        
        # Lower median for circular difference
        median_diff = lower_median(diff_angles)
        median_angle = (mean_angle + median_diff + 2 * np.pi) % (2 * np.pi)
        median_angle_deg = np.rad2deg(median_angle)

        hsv_pixel = np.uint8([[[int(median_angle_deg / 2.0), 255, 255]]])
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        median_bgr = (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))

        return median_bgr, circ_var

    def draw_sample_badge(self, img, n_points, position=(35, 595), overlap_ratio=None):
        text = f"N = {n_points} samples"
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

    def create_base_canvas(self, title, subtitle=None):
        img = np.ones((620, 1000, 3), dtype=np.uint8)
        img[:] = self.BG_COLOR
        cv2.rectangle(img, (0, 0), (999, 619), self.BORDER_COLOR, 3)

        cv2.putText(img, title, (35, 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, self.TEXT_MAIN, 2, cv2.LINE_AA)
        if subtitle:
            cv2.putText(img, subtitle, (35, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 70, 20), 1, cv2.LINE_AA)
        
        y_line = 85 if subtitle else 65
        cv2.line(img, (35, y_line), (965, y_line), self.PANEL_BORDER, 1, cv2.LINE_AA)
        return img

    def draw_lidar(self, img, lidar_data, center=(230, 280), max_radius=150, mode="live", active_flags=None):
        cv2.putText(img, "LIDAR", (45, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)

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
                cv2.line(img, center, (x_val, y_val), self.primary_color, 3, cv2.LINE_AA)
                cv2.circle(img, (x_val, y_val), 4, self.primary_color, -1, cv2.LINE_AA)

            elif mode == "iconic":
                val = float(lidar_data[i])
                norm_val = val / 0.1 if val <= 0.1 else val
                r_val = (max(min(norm_val, 1.0), 0.0)) * max_radius
                x_val = int(center[0] + r_val * np.cos(rad))
                y_val = int(center[1] + r_val * np.sin(rad))
                cv2.circle(img, (x_val, y_val), 5, self.primary_color, -1, cv2.LINE_AA)

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
                
                cv2.line(img, (x_min, y_min), (x_max, y_max), self.secondary_color, 4, cv2.LINE_AA)
                cv2.circle(img, (x_med, y_med), 5, self.primary_color, -1, cv2.LINE_AA)

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

    def draw_grids(self, img, camera_data, is_live=True, active_ganglion=None, active_cones=None):
        rows, cols = 120 // 20, 160 // 20
        cell_size = 210 // rows
        
        rf_start_x, rf_start_y = 520, 120
        c_start_x, c_start_y = 520, 380

        cv2.putText(img, "Retinal Ganglion Cells", (rf_start_x, rf_start_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.putText(img, "Cone Cells", (c_start_x, c_start_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)

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

        center = (500, 350)
        node_radius = 50
        
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        for edge_info in edges:
            source = edge_info.get("source")
            target = edge_info.get("target")
            weight = edge_info.get("weight", 0)
            
            if weight is None or not isinstance(weight, (int, float)) or math.isnan(weight) or math.isinf(weight):
                weight = 0.0

            if source in nodes and target in nodes:
                G.add_edge(source, target, weight=float(weight))
                
        optimal_dist = node_radius / math.sqrt(max(len(nodes), 1))
        raw_pos = nx.spring_layout(G, k=optimal_dist, center=center, scale=210, seed=42)
        positions = {node: (int(coords[0]), int(coords[1])) for node, coords in raw_pos.items()}

        thickness = 2
        max_weight = 1
        min_weight = 0

        for edge_info in edges:
            source = edge_info.get("source")
            target = edge_info.get("target")
            weight = edge_info.get("weight", 0)

            if weight is None or not isinstance(weight, (int, float)) or math.isnan(weight) or math.isinf(weight):
                weight = 0.0

            if source not in positions or target not in positions:
                continue

            pt1 = positions[source]
            pt2 = positions[target]
                    
            if max_weight == min_weight:
                t = 1.0
            else:
                t = (weight - min_weight) / (max_weight - min_weight)
            faint_color = tuple(int(0.85 * bg + 0.15 * pri) for bg, pri in zip(self.BG_COLOR, self.primary_color))
            edge_color = tuple(int((1.0 - t) * faint_color[c] + t * self.primary_color[c]) for c in range(3))

            dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
            dist = math.hypot(dx, dy)

            if dist == 0:
                continue

            start_x = int(pt1[0] + (node_radius * dx / dist))
            start_y = int(pt1[1] + (node_radius * dy / dist))
            end_x = int(pt2[0] - (node_radius * dx / dist))
            end_y = int(pt2[1] - (node_radius * dy / dist))

            P1 = np.array([start_x, start_y], dtype=float)
            P2 = np.array([end_x, end_y], dtype=float)
            
            gap_dx = P2[0] - P1[0]
            gap_dy = P2[1] - P1[1]
            gap_dist = math.hypot(gap_dx, gap_dy)

            if gap_dist > 0:
                M = (P1 + P2) / 2.0
                nx_dir = -gap_dy / gap_dist
                ny_dir = gap_dx / gap_dist
                
                curve_offset = gap_dist * 0.2
                C = M + np.array([nx_dir, ny_dir]) * curve_offset
                
                t_samples = np.linspace(0, 1, 20).reshape(-1, 1)
                curve_pts = ((1 - t_samples)**2 * P1 + 2 * (1 - t_samples) * t_samples * C + t_samples**2 * P2).astype(np.int32)
                
                cv2.polylines(img, [curve_pts], isClosed=False, color=edge_color, thickness=thickness, lineType=cv2.LINE_AA)
                
                tx = P2[0] - C[0]
                ty = P2[1] - C[1]
                t_len = math.hypot(tx, ty)
                
                if t_len > 0:
                    tx /= t_len
                    ty /= t_len
                    arrow_start = (int(P2[0] - tx * 20), int(P2[1] - ty * 20))
                    arrow_end = (int(P2[0]), int(P2[1]))
                    cv2.arrowedLine(img, arrow_start, arrow_end, edge_color, thickness, tipLength=0.4, line_type=cv2.LINE_AA)

                # --- Edge weight label near the start on the convex side ---
                t_label = 0.2  # Near the beginning of the edge
                pt_label = (1 - t_label)**2 * P1 + 2 * (1 - t_label) * t_label * C + t_label**2 * P2

                # Shift outward along the normal (same direction as curve control point C)
                text_offset_dist = 14.0
                label_pos = pt_label + np.array([nx_dir, ny_dir]) * text_offset_dist

                label_text = f"{weight:.2f}" if isinstance(weight, float) else str(weight)
                font_lbl = cv2.FONT_HERSHEY_SIMPLEX
                scale_lbl = 0.38
                thick_lbl = 1
                (lbl_w, lbl_h), _ = cv2.getTextSize(label_text, font_lbl, scale_lbl, thick_lbl)

                lbl_x = int(label_pos[0] - lbl_w / 2)
                lbl_y = int(label_pos[1] + lbl_h / 2)

                cv2.putText(img, label_text, (lbl_x, lbl_y), font_lbl, scale_lbl, self.TEXT_MAIN, thick_lbl, cv2.LINE_AA)

        for node, (x, y) in positions.items():
            cv2.circle(img, (x, y), node_radius, self.tertiary_color, -1)
            cv2.circle(img, (x, y), node_radius, self.primary_color, 2)
                
            display_name = node.replace('_', ' ')
            wrapped_lines = textwrap.wrap(display_name, width=10)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
            thickness_txt = 1
            
            line_metrics = []
            for line in wrapped_lines:
                (tw, th), baseline = cv2.getTextSize(line, font, font_scale, thickness_txt)
                line_metrics.append((line, tw, th, baseline))
                
            total_height = sum(th + baseline for _, _, th, baseline in line_metrics) + max(0, len(wrapped_lines) - 1) * 3
            current_y = y - total_height // 2
            
            for line, tw, th, baseline in line_metrics:
                txt_x = int(x - tw / 2)
                txt_y = int(current_y + th)
                cv2.putText(img, line, (txt_x, txt_y), font, font_scale, self.TEXT_MAIN, thickness_txt, cv2.LINE_AA)
                current_y += th + baseline + 3

    def draw_unified_GNG(self, img, G, node_colors, edge_colors, gng_colors):
        if len(G.nodes) == 0:
            return

        pos = nx.spring_layout(G, seed=42, k=0.25)

        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        raw_w = max_x - min_x if max_x > min_x else 1.0
        raw_h = max_y - min_y if max_y > min_y else 1.0

        max_r_px = max([max(int(G.nodes[n].get("radius", 0.0) * 35), 5) for n in G.nodes()], default=5)

        pad_left = max_r_px + 30
        pad_right = max_r_px + 230
        pad_y_top = max_r_px + 100
        pad_y_bot = max_r_px + 30
        
        avail_w = 1000 - (pad_left + pad_right)
        avail_h = 620 - (pad_y_top + pad_y_bot)
        
        scale = min(avail_w / raw_w, avail_h / raw_h)
        
        center_x = pad_left + avail_w / 2 - ((min_x + max_x) / 2) * scale
        center_y = pad_y_top + avail_h / 2 - ((min_y + max_y) / 2) * scale

        for node in G.nodes():
            pt = (int(center_x + pos[node][0] * scale), int(center_y + pos[node][1] * scale))
            base_color = node_colors.get(node, self.neutral_gray)
            light_color = tuple(int(0.75 * bg + 0.25 * bc) for bg, bc in zip(self.BG_COLOR, base_color))
            
            r_val = G.nodes[node].get("radius", 0.0)
            if r_val > 0:
                r_px = max(int(r_val * 35), 5)
                cv2.circle(img, pt, r_px, light_color, -1, cv2.LINE_AA)

        for u, v in G.edges():
            pt1 = (int(center_x + pos[u][0] * scale), int(center_y + pos[u][1] * scale))
            pt2 = (int(center_x + pos[v][0] * scale), int(center_y + pos[v][1] * scale))
            color = edge_colors.get((u, v), self.neutral_gray)
            cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA)

        for node in G.nodes():
            pt = (int(center_x + pos[node][0] * scale), int(center_y + pos[node][1] * scale))
            color = node_colors.get(node, self.neutral_gray)
            cv2.circle(img, pt, 5, color, -1, cv2.LINE_AA)
            cv2.circle(img, pt, 5, self.TEXT_MAIN, 1, cv2.LINE_AA)

        legend_start_y = 110
        cv2.putText(img, "Predicates:", (700, legend_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_MAIN, 1, cv2.LINE_AA)
        for i, (pred_name, color) in enumerate(gng_colors.items()):
            y_pos = legend_start_y + 20 + (i * 18)
            if y_pos > 600:
                break
            cv2.circle(img, (710, y_pos - 4), 5, color, -1, cv2.LINE_AA)
            cv2.putText(img, pred_name[:24], (725, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.TEXT_MAIN, 1, cv2.LINE_AA)

    def draw_frequency_plot(self, img, timestamps, bin_size_sec=60.0):
        if not timestamps or len(timestamps) < 2:
            cv2.putText(img, "Insufficient timestamp data", (350, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.neutral_gray, 2, cv2.LINE_AA)
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
        cv2.putText(img, stat_text, (plot_x, plot_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_MAIN, 1, cv2.LINE_AA)

        max_count = max(int(np.max(counts)), 1)
        
        grid_steps = 4
        for i in range(grid_steps + 1):
            y_val = plot_y + plot_h - int(i * (plot_h / grid_steps))
            val_label = f"{int(i * (max_count / grid_steps))}"
            cv2.line(img, (plot_x, y_val), (plot_x + plot_w, y_val), self.tertiary_color, 1, cv2.LINE_AA)
            cv2.putText(img, val_label, (plot_x - 35, y_val + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.neutral_gray, 1, cv2.LINE_AA)

        bar_w = max(int(plot_w / n_bins) - 3, 2)
        curve_pts = []

        for i, count in enumerate(counts):
            bar_h = int((count / max_count) * (plot_h - 20))
            bx = plot_x + int(i * (plot_w / n_bins)) + 2
            by = plot_y + plot_h - bar_h

            cv2.rectangle(img, (bx, by), (bx + bar_w, plot_y + plot_h), self.secondary_color, -1)
            cv2.rectangle(img, (bx, by), (bx + bar_w, plot_y + plot_h), self.primary_color, 1)
            
            pt_center = (bx + bar_w // 2, by)
            curve_pts.append(pt_center)

        if len(curve_pts) > 1:
            for i in range(len(curve_pts) - 1):
                cv2.line(img, curve_pts[i], curve_pts[i + 1], self.BORDER_COLOR, 2, cv2.LINE_AA)
            for pt in curve_pts:
                cv2.circle(img, pt, 3, self.primary_color, -1, cv2.LINE_AA)

        x_label = f"Elapsed Time (Bins of {int(bin_size_sec)}s) ->"
        cv2.putText(img, x_label, (plot_x + plot_w // 2 - 100, plot_y + plot_h + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_MAIN, 1, cv2.LINE_AA)

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

    def run(self): raise NotImplementedError

    def _resolve_trans_weight(self, latest_session, filename):
        name = os.path.splitext(os.path.basename(filename))[0]
        match = re.fullmatch(r"sym_(.+?)_enables_(.+?)_pass", name)
        if not match:
            return None
        from_name, to_name = match.groups()
        graph_path = os.path.join(latest_session, "graphs", "trans_graph.json")
        if not os.path.exists(graph_path):
            return None
        try:
            with open(graph_path, 'r') as f:
                graph_data = json.load(f)
            for edge in graph_data.get("edges", []):
                if edge.get("source") == from_name and edge.get("target") == to_name:
                    return edge.get("weight")
        except Exception:
            pass
        return None


class GNGPictureGenerator(PictureGenerator):
    """Categorical Representation: Filters out variant dimensions; renders lower-medians and ranges."""
    
    LINEAR_VAR_THRESHOLD = 0.05      # Variance threshold for lidar and ganglion cells
    CIRCULAR_VAR_THRESHOLD = 0.15    # Circular variance threshold for cone cell hues

    def _format_display_name(self, filename):
        name = os.path.splitext(os.path.basename(filename))[0]

        match = re.fullmatch(r"sym_(.+?)_enables_(.+?)_pass", name)
        if match:
            from_name, to_name = match.groups()
            from_name = from_name.replace("_", " ").strip()
            to_name = to_name.replace("_", " ").strip()
            return f"{from_name} -> {to_name}"

        if name.endswith("_init"):
            name = name[:-5]
            name += ' - Precondition'
        elif name.endswith("_eff"):
            name = name[:-4]
            name += ' - Effect'
        elif name.endswith("_pass"):
            name = name[:-5]
            name += ' - Transition'
        return name.replace("_", " ").strip()

    def run(self):
        latest_session = get_session(self.logs_root, self.session_dir)
        if not latest_session: return
        symbols_dir = os.path.join(latest_session, "PDDL", "GNG")
        if not os.path.exists(symbols_dir): return
        
        for json_path in glob.glob(os.path.join(symbols_dir, "*.json")):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                nodes = np.array(data.get("nodes", []))
                if len(nodes) == 0: continue
                
                min_vals = np.min(nodes, axis=0)
                max_vals = np.max(nodes, axis=0)
                median_vals = lower_median(nodes, axis=0)
                var_vals = np.var(nodes, axis=0)
                
                display_title = self._format_display_name(json_path)
                img = self.renderer.create_base_canvas(display_title, "Categorical Representation")
                
                # Lidars: filter out dimensions with high variance
                lidar_vars = var_vals[:8]
                lidar_active = [bool(v < self.LINEAR_VAR_THRESHOLD) for v in lidar_vars]
                lidar_data = list(zip(min_vals[:8], max_vals[:8], median_vals[:8]))
                self.renderer.draw_lidar(img, lidar_data, mode="categorical", active_flags=lidar_active)
                
                # Retinal Ganglion Cells: 48 cells * 2 activations (on / off)
                ganglion_medians = []
                active_ganglion = []
                for cell_idx in range(48):
                    idx = 8 + (cell_idx * 5)
                    on_var, off_var = var_vals[idx], var_vals[idx + 1]
                    cell_active = (on_var < self.LINEAR_VAR_THRESHOLD) and (off_var < self.LINEAR_VAR_THRESHOLD)
                    active_ganglion.append(bool(cell_active))
                    ganglion_medians.append(median_vals[idx])
                    ganglion_medians.append(median_vals[idx + 1])

                # Cone Cells: 48 cells * 3 color channels
                cone_medians = []
                active_cones = []
                for cell_idx in range(48):
                    offset = 8 + (cell_idx * 5)
                    r_samples = nodes[:, offset + 2]
                    g_samples = nodes[:, offset + 3]
                    b_samples = nodes[:, offset + 4]
                    median_bgr, hue_var = self.renderer.compute_hue_median_stats(r_samples, g_samples, b_samples)
                    
                    cone_medians.append(median_bgr)
                    active_cones.append(bool(hue_var < self.CIRCULAR_VAR_THRESHOLD))

                self.renderer.draw_grids(img, (ganglion_medians, cone_medians), is_live=False, 
                                         active_ganglion=active_ganglion, active_cones=active_cones)

                overlap_weight = self._resolve_trans_weight(latest_session, json_path)
                self.renderer.draw_sample_badge(img, n_points=len(nodes), position=(35, 595), overlap_ratio=overlap_weight)

                out_dir = self.get_pictures_dir()
                if not out_dir:
                    return
                file_base = os.path.splitext(os.path.basename(json_path))[0]
                out_path = os.path.join(out_dir, f"{file_base}_invariants.png")
                cv2.imwrite(out_path, img)
                print(f"[PictureGenerator] Saved GNG categorical representation to {out_path}")
            except Exception as e:
                print(f"[PictureGenerator] Failed to process {json_path}: {e}")

class RepresentativePointPictureGenerator(PictureGenerator):
    """Iconic Representation: original data from the point closest to Center of Mass. Lidars rendered as points."""
    
    def _format_display_name(self, filename):
        name = os.path.splitext(os.path.basename(filename))[0]

        match = re.fullmatch(r"sym_(.+?)_enables_(.+?)_pass", name)
        if match:
            from_name, to_name = match.groups()
            from_name = from_name.replace("_", " ").strip()
            to_name = to_name.replace("_", " ").strip()
            return f"{from_name} -> {to_name}"

        if name.endswith("_init"):
            name = name[:-5]
            name += ' - Precondition'
        elif name.endswith("_eff"):
            name = name[:-4]
            name += ' - Effect'
        elif name.endswith("_pass"):
            name = name[:-5]
            name += ' - Transition'
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

        for json_path in glob.glob(os.path.join(symbols_dir, "*.json")):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                nodes = np.array(data.get("nodes", []))
                if len(nodes) == 0:
                    continue

                # Iconic point: sample closest to global center of mass
                centroid = np.mean(nodes, axis=0)
                dists = np.linalg.norm(nodes - centroid, axis=1)

                rep_idx = int(np.argmin(dists))
                rep_node = nodes[rep_idx]

                file_base = os.path.splitext(os.path.basename(json_path))[0]
                display_title = self._format_display_name(json_path)
                
                img = self.renderer.create_base_canvas(display_title, "Iconic Representation")

                lidar_data = rep_node[:8]
                self.renderer.draw_lidar(img, lidar_data, mode="iconic")

                cam_data = rep_node[8:248]
                self.renderer.draw_grids(img, cam_data, is_live=True)

                overlap_weight = self._resolve_trans_weight(latest_session, json_path)
                self.renderer.draw_sample_badge(img, n_points=len(nodes), position=(35, 595), overlap_ratio=overlap_weight)

                out_path = os.path.join(out_dir, f"{file_base}_icon.png")
                cv2.imwrite(out_path, img)
                print(f"[Picture Generator] Saved iconic representation to {out_path}")
            except Exception as e:
                print(f"[Picture Generator] Error processing {json_path}: {e}")

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
            print(f"[PictureGenerator] File not found: {self.timestamp_file}")
            return

        try:
            with open(file_path, "r") as f:
                timestamps = [float(line.strip()) for line in f if line.strip()]
            
            img = self.renderer.create_base_canvas("UPDATE FREQUENCY OVER TIME")
            self.renderer.draw_frequency_plot(img, timestamps, bin_size_sec=self.bin_size_sec)
            self.renderer.draw_sample_badge(img, n_points=len(timestamps), position=(35, 595))

            out_dir = self.get_pictures_dir() or "."
            out_path = os.path.join(out_dir, "update_frequency.png")
            cv2.imwrite(out_path, img)
            print(f"[PictureGenerator] Saved frequency plot to {out_path}")
        except Exception as e:
            print(f"[PictureGenerator] Failed to generate frequency plot: {e}")

class TransGraphPictureGenerator(PictureGenerator):
    def run(self):
        latest_session = get_session(self.logs_root, self.session_dir)
        if not latest_session: return
        graphs_dir = os.path.join(latest_session, "graphs")

        json_path = os.path.join(graphs_dir, "trans_graph.json")
        if not os.path.exists(json_path): return
        
        try:
            with open(json_path, 'r') as f:
                graph_data = json.load(f)
                
            img = self.renderer.create_base_canvas("TRANSITION GRAPH")
            self.renderer.draw_transition_graph(img, graph_data)

            out_dir = self.get_pictures_dir()
            if not out_dir:
                return
            out_path = os.path.join(out_dir, "trans_graph_visualised.png")
            cv2.imwrite(out_path, img)
            print(f"[PictureGenerator] Saved Transition Graph image to {out_path}")
        except Exception as e:
            print(f"[PictureGenerator] Failed to generate TransGraph: {e}")

class UnifiedGNGPictureGenerator(PictureGenerator):
    def __init__(self, logs_root="logs", pred_type='init', session_dir=None):
        super().__init__(logs_root, session_dir=session_dir)
        self.gng_colors = {}
        self.pred_type = pred_type
        self.color_palette = [
            (60, 60, 220),   (60, 220, 60),   (220, 100, 60),   (60, 200, 220),
            (200, 60, 200),  (220, 200, 60),  (100, 120, 255),  (255, 120, 100),
            (100, 255, 120), (40, 140, 240),  (180, 50, 130),   (60, 140, 20),
            (160, 130, 240), (150, 210, 240), (50, 90, 180),   (120, 180, 50),
            (190, 70, 130),  (30, 160, 190),  (160, 40, 70),    (40, 190, 140),
            (210, 140, 40),  (130, 80, 210),  (80, 150, 100),   (140, 140, 140)
        ]

    def _format_display_name(self, filename):
        name = os.path.splitext(os.path.basename(filename))[0]

        match = re.fullmatch(r"sym_(.+?)_enables_(.+?)_pass", name)
        if match:
            from_name, to_name = match.groups()
            from_name = from_name.replace("_", " ").strip()
            to_name = to_name.replace("_", " ").strip()
            return f"{from_name} -> {to_name}"

        if name.endswith("_init"):
            name = name[:-5]
        elif name.endswith("_eff"):
            name = name[:-4]
        elif name.endswith("_pass"):
            name = name[:-5]

        return name.replace("_", " ").strip()

    def _get_color(self, predicate_name):
        if predicate_name not in self.gng_colors:
            color_idx = len(self.gng_colors) % len(self.color_palette)
            self.gng_colors[predicate_name] = self.color_palette[color_idx]
        return self.gng_colors[predicate_name]

    def run(self):
        latest_session = get_session(self.logs_root, self.session_dir)
        if not latest_session: return
        symbols_dir = os.path.join(latest_session, "PDDL", "GNG")
        if not os.path.exists(symbols_dir): return

        json_files = glob.glob(os.path.join(symbols_dir, f"*_{self.pred_type}.json"))
        
        G = nx.Graph()
        node_colors = {}
        edge_colors = {}
        
        for json_path in json_files:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                pred_name = self._format_display_name(json_path)
                color = self._get_color(pred_name)
                
                nodes = data.get("nodes", [])
                radii = data.get("local_radii", [])
                
                for i in range(len(nodes)):
                    node_id = f"{pred_name}_{i}"
                    r_val = radii[i] if i < len(radii) else 0.0
                    G.add_node(node_id, radius=r_val)
                    node_colors[node_id] = color

                edges_dict = data.get("edges", {})
                for edge_str in edges_dict.keys():
                    u_str, v_str = edge_str.split(",")
                    u_id = f"{pred_name}_{u_str}"
                    v_id = f"{pred_name}_{v_str}"
                    G.add_edge(u_id, v_id)
                    edge_colors[(u_id, v_id)] = color
                    edge_colors[(v_id, u_id)] = color
            except Exception as e:
                print(f"[PictureGenerator] Failed to process {json_path} for unified graph: {e}")

        if len(G.nodes) == 0: return
        pred_type = 'Unknown Type'
        if self.pred_type == 'eff':
            pred_type = 'effects'
        elif self.pred_type == 'init':
            pred_type = 'preconditions'
        elif self.pred_type == 'pass':
            pred_type = "transitions"
        img = self.renderer.create_base_canvas("UNIFIED GNG PLANE", f"Topological state space mapping for {pred_type}")
        self.renderer.draw_unified_GNG(img, G, node_colors, edge_colors, self.gng_colors)

        out_dir = self.get_pictures_dir()
        if not out_dir: return
        out_path = os.path.join(out_dir, f"unified_gng_plane_{self.pred_type}.png")
        cv2.imwrite(out_path, img)
        print(f"[PictureGenerator] Saved Unified GNG plane to {out_path}")

class TestStatisticsPictureGenerator(PictureGenerator):
    """Renders test statistics: success rate, plan lengths, execution times, and skill executions per step."""
    
    def __init__(self, logs_root="logs", session_dir=None, stats_file="test_statistics.json"):
        super().__init__(logs_root, session_dir=session_dir)
        self.stats_file = stats_file
        self.success_color = (60, 180, 75)      # BGR Green
        self.fail_color = (60, 60, 220)         # BGR Red

    def run(self):
        latest_session = get_session(self.logs_root, self.session_dir)
        if not latest_session:
            return

        json_path = os.path.join(latest_session, self.stats_file)
        if not os.path.exists(json_path):
            print(f"[PictureGenerator] Test stats not found at {json_path}")
            return

        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            img = self.renderer.create_base_canvas("TEST SESSION PERFORMANCE", "Execution step breakdown & metrics")
            
            # 1. Summary Metrics Banner (Top Left)
            sr = data.get("success_rate", 0.0) * 100.0
            mean_pl = data.get("mean_plan_length", 0.0)
            mode_pl = data.get("mode_plan_length", [])
            mean_se = data.get("mean_steps_executed", 0.0)
            mode_se = data.get("mode_steps_executed", [])
            mean_dur = data.get("mean_execution_time_s", 0.0)
            n_tests = data.get("total_tests", 0)

            stats_lines = [
                f"Tests: {n_tests} | Success Rate: {sr:.1f}%",
                f"Plan Length -> Mean: {mean_pl:.2f} | Mode: {mode_pl}",
                f"Steps Executed -> Mean: {mean_se:.2f} | Mode: {mode_se}",
                f"Avg Execution Time: {mean_dur:.2f}s"
            ]

            box_x, box_y = 45, 100
            for i, line in enumerate(stats_lines):
                cv2.putText(img, line, (box_x, box_y + (i * 22)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

            # 2. Step vs Skill Stacked Bar Chart
            step_graph_data = data.get("step_graph_data", {})
            if not step_graph_data:
                return

            # Collect unique steps and skills
            all_steps = set()
            for sk, steps in step_graph_data.items():
                all_steps.update(int(s) for s in steps.keys())

            if not all_steps:
                return

            max_step = max(all_steps)
            steps_range = list(range(max_step + 1))
            skills = sorted(list(step_graph_data.keys()))

            # Chart area setup
            plot_x, plot_y = 45, 210
            plot_w, plot_h = 910, 360
            cv2.rectangle(img, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), self.renderer.PANEL_BORDER, 1)

            # Find maximum total attempts on any single step
            max_y_val = 1
            for s in steps_range:
                step_total = sum(step_graph_data.get(sk, {}).get(str(s), {}).get("total", 0) for sk in skills)
                max_y_val = max(max_y_val, step_total)

            # Draw Y Gridlines
            grid_steps = 4
            for i in range(grid_steps + 1):
                y_val = plot_y + plot_h - int(i * (plot_h / grid_steps))
                val_label = f"{int(i * (max_y_val / grid_steps))}"
                cv2.line(img, (plot_x, y_val), (plot_x + plot_w, y_val), self.renderer.tertiary_color, 1, cv2.LINE_AA)
                cv2.putText(img, val_label, (plot_x - 30, y_val + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.renderer.neutral_gray, 1, cv2.LINE_AA)

            # Draw bars: grouped by step, divided by skill
            step_width = plot_w / len(steps_range)
            bar_slot_width = step_width * 0.8
            bar_width = max(int(bar_slot_width / max(len(skills), 1)), 3)

            for s_idx, s in enumerate(steps_range):
                step_base_x = plot_x + int(s_idx * step_width) + int(step_width * 0.1)
                
                # Step X-axis label
                cv2.putText(img, f"Step {s}", (step_base_x + int(bar_slot_width / 2) - 18, plot_y + plot_h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

                for k_idx, sk in enumerate(skills):
                    sk_step_info = step_graph_data.get(sk, {}).get(str(s), {"succeed": 0, "fail": 0, "total": 0})
                    n_succ = sk_step_info.get("succeed", 0)
                    n_fail = sk_step_info.get("fail", 0)

                    if n_succ == 0 and n_fail == 0:
                        continue

                    bx = step_base_x + (k_idx * bar_width)
                    h_succ = int((n_succ / max_y_val) * (plot_h - 20))
                    h_fail = int((n_fail / max_y_val) * (plot_h - 20))

                    # Draw successful (green) segment
                    if h_succ > 0:
                        by_succ = plot_y + plot_h - h_succ
                        cv2.rectangle(img, (bx, by_succ), (bx + bar_width - 1, plot_y + plot_h), self.success_color, -1)

                    # Draw failed (red) segment stacked above
                    if h_fail > 0:
                        by_fail = plot_y + plot_h - h_succ - h_fail
                        cv2.rectangle(img, (bx, by_fail), (bx + bar_width - 1, plot_y + plot_h - h_succ), self.fail_color, -1)

            # 3. Legend (Top Right)
            cv2.rectangle(img, (740, 95), (752, 107), self.success_color, -1)
            cv2.putText(img, "Succeeded", (760, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)
            cv2.rectangle(img, (740, 115), (752, 127), self.fail_color, -1)
            cv2.putText(img, "Failed", (760, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.renderer.TEXT_MAIN, 1, cv2.LINE_AA)

            out_dir = self.get_pictures_dir()
            if not out_dir:
                return
            out_path = os.path.join(out_dir, "test_statistics.png")
            cv2.imwrite(out_path, img)
            print(f"[PictureGenerator] Saved test statistics graph to {out_path}")

        except Exception as e:
            print(f"[PictureGenerator] Failed to generate test statistics picture: {e}")
            
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
            TestStatisticsPictureGenerator(logs_root, session_dir=session_dir)
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
            try: return self.bus.call_service(topic, timeout=self.timeout)
            except TypeError: return self.bus.call_service(topic)
        except Exception: return None

    def update(self):
        self._ensure_window()
        cv2.waitKey(1)

        lidar_readings = []
        for direction in self.LIDAR_ANGLES:
            val = self._safe_call(f"/lidar/{direction}/ask/value")
            if val is None: return
            lidar_readings.append(val)

        raw_coded = self._safe_call("/camera/ask/coded")
        if raw_coded is None: return
        cam_array = np.asarray(raw_coded, dtype=float).flatten()
        if len(cam_array) < (6 * 8 * 5): return

        speeds = self._safe_call("/wheels/ask/speed")
        max_vel = self._safe_call("/get/wheels/max_velocity")
        if speeds is None or max_vel is None: return

        img = self.create_base_canvas("LIVE SENSIMOTOR MONITOR")
        self.draw_lidar(img, lidar_readings, mode="live")
        self.draw_wheels(img, speeds[0], speeds[1], float(max_vel))
        self.draw_grids(img, cam_array, is_live=True)

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
        if not latest_session: return

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
                self.draw_sample_badge(img, n_points=len(timestamps), position=(35, 595))

                cv2.imshow(self.window_name, img)
                self._last_mtime = mtime
        except Exception as e:
            print(f"[LiveUpdateFrequencyMonitor] Error loading timestamps: {e}")

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False

class LiveGraphMonitor(Renderer):
    def __init__(self, logs_root="logs", trans_filename="trans_graph.json"):
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
        if not latest_session: return

        trans_path = os.path.join(latest_session, "graphs", self.trans_filename)

        try:
            if os.path.exists(trans_path) and os.path.getsize(trans_path) > 0:
                mtime = os.path.getmtime(trans_path)
                if mtime > self._last_trans_mtime:
                    with open(trans_path, "r") as f:
                        graph_data = json.load(f)
                        
                    img = self.create_base_canvas("LIVE TRANSITION GRAPH")
                    self.draw_transition_graph(img, graph_data)
                    cv2.imshow(self.trans_window, img)
                    
                    self._last_trans_mtime = mtime
        except Exception as e:
            print(f"Error loading live graph: {e}")

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.trans_window)
            self._window_created = False

class LiveGNGMonitor(Renderer):
    def __init__(self, logs_root="logs"):
        super().__init__()
        self.logs_root = logs_root

        self.window_name = "Live GNG Plane"
        self._window_created = False

        self._last_mtimes = {}
        self.gng_colors = {}
        
        self.color_palette = [
            (60, 60, 220),   (60, 220, 60),   (220, 100, 60),   (60, 200, 220),
            (200, 60, 200),  (220, 200, 60),  (100, 120, 255),  (255, 120, 100),
            (100, 255, 120), (40, 140, 240),  (180, 50, 130),   (60, 140, 20),
            (160, 130, 240), (150, 210, 240), (50, 90, 180),   (120, 180, 50),
            (190, 70, 130),  (30, 160, 190),  (160, 40, 70),    (40, 190, 140),
            (210, 140, 40),  (130, 80, 210),  (80, 150, 100),   (140, 140, 140)
        ]

    def _format_display_name(self, filename):
        name = os.path.splitext(os.path.basename(filename))[0]

        match = re.fullmatch(r"sym_(.+?)_enables_(.+?)_pass", name)
        if match:
            from_name, to_name = match.groups()
            from_name = from_name.replace("_", " ").strip()
            to_name = to_name.replace("_", " ").strip()
            return f"{from_name} -> {to_name}"

        if name.endswith("_init"):
            name = name[:-5]
            name += ' - Precondition'
        elif name.endswith("_eff"):
            name = name[:-4]
            name += ' - Effect'
        elif name.endswith("_pass"):
            name = name[:-5]
            name += ' - Transition'
        return name.replace("_", " ").strip()

    def _ensure_window(self):
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            placeholder = self.create_base_canvas("LIVE GNG PLANE", "Waiting for predicate data...")
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

        G = nx.Graph()
        node_colors = {}
        edge_colors = {}

        for json_path in json_files:
            data = self._load_json_with_retry(json_path)
            if data is None:
                continue

            display_name = self._format_display_name(json_path)
            color = self._get_color(display_name)

            nodes = data.get("nodes", [])
            radii = data.get("local_radii", [])

            for i in range(len(nodes)):
                node_id = f"{display_name}_{i}"
                r_val = radii[i] if i < len(radii) else 0.0
                G.add_node(node_id, radius=r_val)
                node_colors[node_id] = color

            edges_dict = data.get("edges", {})
            for edge_str in edges_dict.keys():
                u_str, v_str = edge_str.split(",")
                u_id = f"{display_name}_{u_str}"
                v_id = f"{display_name}_{v_str}"
                G.add_edge(u_id, v_id)
                edge_colors[(u_id, v_id)] = color
                edge_colors[(v_id, u_id)] = color

        if len(G.nodes) == 0:
            return

        img = self.create_base_canvas("LIVE GNG PLANE", "preconditions")
        self.draw_unified_GNG(img, G, node_colors, edge_colors, self.gng_colors)
        cv2.imshow(self.window_name, img)

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False

class LiveRepresentativePointMonitor(Renderer):
    def __init__(self, bus, logs_root="logs"):
        super().__init__()
        self.bus = bus
        self.logs_root = logs_root
        self.window_name = "Live Representative Point Monitor"
        self._window_created = False


    def _ensure_window(self):
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            placeholder = self.create_base_canvas(
                "LIVE REPRESENTATIVE POINT", "Waiting for symbol data..."
            )
            cv2.imshow(self.window_name, placeholder)
            self._window_created = True

    def _resolve_action_name(self):
        try:
            res = self.bus.call_service("agent/ask/action")
            if isinstance(res, dict):
                return res.get("action") or res.get("name")
            return res
        except Exception as e:
            print(f"[LiveRepresentativePointMonitor] Failed to query action service: {e}")
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

        action_file = os.path.join(
            latest_session, "PDDL", "GNG", f"{action_name}_init.json"
        )
        if not os.path.isfile(action_file):
            return

        try:
            with open(action_file, "r") as f:
                data = json.load(f)

            nodes = np.array(data.get("nodes", []))
            if len(nodes) == 0:
                return

            # Representative sample point closest to center of mass
            centroid = np.mean(nodes, axis=0)
            dists = np.linalg.norm(nodes - centroid, axis=1)

            rep_idx = int(np.argmin(dists))
            rep_node = nodes[rep_idx]

            display_title = str(action_name).replace("_", " ")
            img = self.create_base_canvas(display_title, "Iconic Representation")

            self.draw_lidar(img, rep_node[:8], mode="iconic")
            self.draw_grids(img, rep_node[8:248], is_live=True)
            self.draw_sample_badge(img, n_points=len(nodes), position=(35, 595))

            cv2.imshow(self.window_name, img)

        except Exception as e:
            print(f"[LiveRepresentativePointMonitor] Error rendering point: {e}")

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False

if __name__ == "__main__":
    sd = "offline" # or None for newest one
    gen = PipelinePictureGenerator(session_dir=sd)
    gen.run()