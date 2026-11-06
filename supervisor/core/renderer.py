import cv2
import math
import textwrap
import numpy as np
import networkx as nx
from supervisor.utils.core_utils import lower_median, SKILL_ORDER

class Renderer:
    def __init__(self):
        self.primary_color = (195, 105, 30)
        self.secondary_color = (235, 185, 90)
        self.tertiary_color = (248, 235, 215)
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
            r, g, b = np.array(r_vals, dtype=np.float32), np.array(g_vals, dtype=np.float32), np.array(b_vals, dtype=np.float32)
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin
        v = cmax
        s = np.zeros_like(v)
        nonzero_v = v > 1e-06
        s[nonzero_v] = delta[nonzero_v] / v[nonzero_v]
        h = np.zeros_like(r)
        nonzero_delta = delta > 1e-06
        mask = nonzero_delta & (cmax == r)
        h[mask] = 60.0 * (((g[mask] - b[mask]) / delta[mask]) % 6)
        mask = nonzero_delta & (cmax == g)
        h[mask] = 60.0 * (((b[mask] - r[mask]) / delta[mask]) + 2)
        mask = nonzero_delta & (cmax == b)
        h[mask] = 60.0 * (((r[mask] - g[mask]) / delta[mask]) + 4)
        angles = np.deg2rad(h)
        sin_mean, cos_mean = np.mean(np.sin(angles)), np.mean(np.cos(angles))
        R = np.hypot(sin_mean, cos_mean)
        circ_var = float(1.0 - R)
        mean_angle = np.arctan2(sin_mean, cos_mean)
        diff_angles = np.arctan2(np.sin(angles - mean_angle), np.cos(angles - mean_angle))
        median_diff = lower_median(diff_angles)
        median_angle = (mean_angle + median_diff + 2 * np.pi) % (2 * np.pi)
        median_angle_deg = np.rad2deg(median_angle)
        median_s, median_v = float(lower_median(s)), float(lower_median(v))
        h_cv = int(np.clip(median_angle_deg / 2.0, 0, 179))
        s_cv, v_cv = int(np.clip(median_s * 255.0, 0, 255)), int(np.clip(median_v * 255.0, 0, 255))
        hsv_pixel = np.uint8([[[h_cv, s_cv, v_cv]]])
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])), circ_var

    def draw_sample_badge(self, img, count: int, label: str='Nodes', position=(90, 595)):
        text = f'{label}: {count}'
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
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

    def draw_lidar(self, img, lidar_data, center=(230, 280), max_radius=150, mode='live', active_flags=None, custom_color=None):
        cv2.putText(img, 'LIDAR', (45, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)
        color_main = custom_color if custom_color is not None else self.primary_color
        color_sec = self.secondary_color if custom_color is None else tuple(int(0.6 * c + 0.4 * 255) for c in color_main)
        for ang in self.LIDAR_ANGLES:
            rad = np.deg2rad(ang - 90)
            cv2.line(img, center, (int(center[0] + max_radius * np.cos(rad)), int(center[1] + max_radius * np.sin(rad))), self.tertiary_color, 2, cv2.LINE_AA)
        for i, ang in enumerate(self.LIDAR_ANGLES):
            if active_flags is not None and not active_flags[i]: continue
            rad = np.deg2rad(ang - 90)
            if mode == 'live':
                r_val = (max(min(float(lidar_data[i]), 0.1), 0.0) / 0.1) * max_radius
                x_val, y_val = int(center[0] + r_val * np.cos(rad)), int(center[1] + r_val * np.sin(rad))
                cv2.line(img, center, (x_val, y_val), color_main, 3, cv2.LINE_AA)
                cv2.circle(img, (x_val, y_val), 4, color_main, -1, cv2.LINE_AA)
            elif mode == 'iconic':
                val = float(lidar_data[i])
                r_val = max(min(val / 0.1 if val <= 0.1 else val, 1.0), 0.0) * max_radius
                x_val, y_val = int(center[0] + r_val * np.cos(rad)), int(center[1] + r_val * np.sin(rad))
                cv2.circle(img, (x_val, y_val), 5, color_main, -1, cv2.LINE_AA)
            elif mode == 'categorical':
                val_min, val_max, val_median = lidar_data[i]
                r_min = max(min(val_min / 0.1 if val_min <= 0.1 else val_min, 1.0), 0.0) * max_radius
                r_max = max(min(val_max / 0.1 if val_max <= 0.1 else val_max, 1.0), 0.0) * max_radius
                r_med = max(min(val_median / 0.1 if val_median <= 0.1 else val_median, 1.0), 0.0) * max_radius
                x_min, y_min = int(center[0] + r_min * np.cos(rad)), int(center[1] + r_min * np.sin(rad))
                x_max, y_max = int(center[0] + r_max * np.cos(rad)), int(center[1] + r_max * np.sin(rad))
                x_med, y_med = int(center[0] + r_med * np.cos(rad)), int(center[1] + r_med * np.sin(rad))
                cv2.line(img, (x_min, y_min), (x_max, y_max), color_sec, 4, cv2.LINE_AA)
                cv2.circle(img, (x_med, y_med), 5, color_main, -1, cv2.LINE_AA)
        cv2.circle(img, center, 34, self.ROBOT_BODY, -1, cv2.LINE_AA)
        cv2.circle(img, center, 36, (100, 105, 115), 2, cv2.LINE_AA)
        cv2.line(img, (center[0], center[1] - 34), (center[0], center[1] - 18), (255, 255, 255), 2, cv2.LINE_AA)

    def draw_wheels(self, img, left, right, max_vel, center=(230, 520)):
        cv2.putText(img, 'Wheels', (100, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.putText(img, 'L', (165, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.putText(img, 'R', (265, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.line(img, (140, center[1]), (290, center[1]), self.PANEL_BORDER, 2, cv2.LINE_AA)
        max_h, max_vel = 40, max(max_vel, 0.001)
        l_h = int(np.clip((left / max_vel) * max_h, -max_h, max_h))
        r_h = int(np.clip((right / max_vel) * max_h, -max_h, max_h))
        cv2.rectangle(img, (150, center[1] - max_h), (190, center[1] + max_h), self.tertiary_color, -1)
        cv2.rectangle(img, (250, center[1] - max_h), (290, center[1] + max_h), self.tertiary_color, -1)
        cv2.rectangle(img, (150, center[1] - l_h), (190, center[1]), self.primary_color, -1)
        cv2.rectangle(img, (250, center[1] - r_h), (290, center[1]), self.primary_color, -1)

    def draw_grids(self, img, camera_data, is_live=True, active_ganglion=None, active_cones=None, origin=(520, 120)):
        rows, cols = 6, 8
        cell_size = 210 // rows
        rf_start_x, rf_start_y = origin
        c_start_x, c_start_y = origin[0], origin[1] + 260
        cv2.putText(img, 'Retinal Ganglion Cells Receptive Fields', (rf_start_x, rf_start_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.putText(img, 'Cone Cells', (c_start_x, c_start_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.TEXT_MAIN, 2, cv2.LINE_AA)
        for r in range(rows):
            for c in range(cols):
                cell_idx = r * cols + c
                rf_x, rf_y = rf_start_x + c * cell_size, rf_start_y + r * cell_size
                cp_x, cp_y = c_start_x + c * cell_size, c_start_y + r * cell_size
                ganglion_active = True if active_ganglion is None else active_ganglion[cell_idx]
                cone_active = True if active_cones is None else active_cones[cell_idx]
                if is_live:
                    idx = cell_idx * 5
                    on_val, off_val, red, green, blue = camera_data[idx:idx + 5]
                    if ganglion_active:
                        rf_color = self.primary_color if (on_val > 0.5 and off_val <= 0.5) else self.secondary_color if (off_val > 0.5 and on_val <= 0.5) else self.tertiary_color
                        cv2.rectangle(img, (rf_x, rf_y), (rf_x + cell_size - 2, rf_y + cell_size - 2), rf_color, -1)
                    else:
                        cv2.rectangle(img, (rf_x, rf_y), (rf_x + cell_size - 2, rf_y + cell_size - 2), self.PANEL_BORDER, 1)
                    if cone_active:
                        if max(red, green, blue) <= 1.0: red, green, blue = red * 255.0, green * 255.0, blue * 255.0
                        cp_color = (int(np.clip(blue, 0, 255)), int(np.clip(green, 0, 255)), int(np.clip(red, 0, 255)))
                        cv2.rectangle(img, (cp_x, cp_y), (cp_x + cell_size - 2, cp_y + cell_size - 2), cp_color, -1)
                    else:
                        cv2.rectangle(img, (cp_x, cp_y), (cp_x + cell_size - 2, cp_y + cell_size - 2), self.PANEL_BORDER, 1)
                else:
                    ganglion_medians, cone_medians = camera_data
                    if ganglion_active:
                        idx = cell_idx * 2
                        on_med, off_med = ganglion_medians[idx], ganglion_medians[idx + 1]
                        rf_color = self.primary_color if (on_med > 0.5 and off_med <= 0.5) else self.secondary_color if (off_med > 0.5 and on_med <= 0.5) else self.tertiary_color
                        cv2.rectangle(img, (rf_x, rf_y), (rf_x + cell_size - 2, rf_y + cell_size - 2), rf_color, -1)
                    else:
                        cv2.rectangle(img, (rf_x, rf_y), (rf_x + cell_size - 2, rf_y + cell_size - 2), self.PANEL_BORDER, 1)
                    if cone_active:
                        cv2.rectangle(img, (cp_x, cp_y), (cp_x + cell_size - 2, cp_y + cell_size - 2), cone_medians[cell_idx], -1)
                    else:
                        cv2.rectangle(img, (cp_x, cp_y), (cp_x + cell_size - 2, cp_y + cell_size - 2), self.PANEL_BORDER, 1)

    def draw_transition_graph(self, img, graph_data):
        nodes, edges = graph_data.get('nodes', []), graph_data.get('edges', [])
        if not nodes:
            cv2.putText(img, 'No graph data', (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, self.neutral_gray, 2)
            return
        ordered_nodes = [n for n in [s.lower() for s in SKILL_ORDER] if n in nodes]
        for n in nodes: 
            if n not in ordered_nodes: ordered_nodes.append(n)
        node_indices = {node: i for i, node in enumerate(ordered_nodes)}
        n_nodes, (h, w) = len(ordered_nodes), img.shape[:2]
        node_radius, y_center, margin_x = 42, h // 2, 80
        if n_nodes > 1:
            step_x = (w - 2 * margin_x) / (n_nodes - 1)
            positions = {node: (int(margin_x + i * step_x), y_center) for i, node in enumerate(ordered_nodes)}
        else:
            positions = {ordered_nodes[0]: (w // 2, y_center)}
        thickness, max_weight, min_weight = 2, 1.0, 0.0
        for edge_info in edges:
            source, target, weight = edge_info.get('source'), edge_info.get('target'), edge_info.get('weight', 1)
            if weight is None or not isinstance(weight, (int, float)) or math.isnan(weight) or math.isinf(weight): weight = 0.0
            if source not in positions or target not in positions: continue
            pt1, pt2 = positions[source], positions[target]
            idx_src, idx_tgt = node_indices[source], node_indices[target]
            if weight >= 1.0:
                edge_color = self.status_green
            else:
                t = max(0.0, min(1.0, (weight - min_weight) / (max_weight - min_weight))) if max_weight != min_weight else 1.0
                faint_color = tuple(int(0.85 * bg + 0.15 * pri) for bg, pri in zip(self.BG_COLOR, self.primary_color))
                edge_color = tuple(int((1.0 - t) * faint_color[c] + t * self.primary_color[c]) for c in range(3))
            if source == target:
                P1, P2, C = np.array([pt1[0] - 12.0, pt1[1] - node_radius]), np.array([pt1[0] + 12.0, pt1[1] - node_radius]), np.array([pt1[0], pt1[1] - node_radius - 65.0])
                t_samples = np.linspace(0, 1, 25).reshape(-1, 1)
                curve_pts = ((1 - t_samples) ** 2 * P1 + 2 * (1 - t_samples) * t_samples * C + t_samples ** 2 * P2).astype(np.int32)
                cv2.polylines(img, [curve_pts], False, edge_color, thickness, cv2.LINE_AA)
                continue
            is_forward = idx_src < idx_tgt
            y_direction = -1.0 if is_forward else 1.0
            P1, P2 = np.array([pt1[0], pt1[1] + y_direction * node_radius]), np.array([pt2[0], pt2[1] + y_direction * node_radius])
            node_step_diff = abs(idx_tgt - idx_src)
            curve_height = 45.0 + (node_step_diff - 1) * 26.0
            M = (P1 + P2) / 2.0
            C = np.array([M[0], M[1] + y_direction * curve_height])
            t_samples = np.linspace(0, 1, 35).reshape(-1, 1)
            curve_pts = ((1 - t_samples) ** 2 * P1 + 2 * (1 - t_samples) * t_samples * C + t_samples ** 2 * P2).astype(np.int32)
            cv2.polylines(img, [curve_pts], False, edge_color, thickness, cv2.LINE_AA)
            tangent = P2 - C
            t_len = math.hypot(tangent[0], tangent[1])
            if t_len > 0:
                tx, ty = tangent[0] / t_len, tangent[1] / t_len
                cv2.arrowedLine(img, (int(P2[0] - tx * 16), int(P2[1] - ty * 16)), (int(P2[0]), int(P2[1])), edge_color, thickness, tipLength=0.45, line_type=cv2.LINE_AA)
            apex = (P1 + 2 * C + P2) / 4.0
            label_text = f'{weight:.2f}' if isinstance(weight, float) else str(weight)
            (lbl_w, lbl_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(img, label_text, (int(apex[0] - lbl_w / 2), int(apex[1] + y_direction * 14.0 + (lbl_h / 2 if y_direction > 0 else -lbl_h / 2))), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.TEXT_MAIN, 1, cv2.LINE_AA)
        for node, (x, y) in positions.items():
            cv2.circle(img, (x, y), node_radius, self.tertiary_color, -1)
            cv2.circle(img, (x, y), node_radius, self.primary_color, 2)
            wrapped_lines = textwrap.wrap(node.replace('_', ' ').title(), width=9)
            line_metrics = [(*cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1), line) for line in wrapped_lines]
            total_height = sum(th + b for (tw, th), b, line in line_metrics) + max(0, len(wrapped_lines) - 1) * 3
            current_y = y - total_height // 2
            for (tw, th), baseline, line in line_metrics:
                cv2.putText(img, line, (int(x - tw / 2), int(current_y + th)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.TEXT_MAIN, 1, cv2.LINE_AA)
                current_y += th + baseline + 3

    def draw_frequency_plot(self, img, timestamps, bin_size_sec=60.0):
        if not timestamps or len(timestamps) < 2:
            cv2.putText(img, 'Insufficient timestamp data', (350, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.neutral_gray, 2, cv2.LINE_AA)
            return
        ts = np.sort(np.array(timestamps, dtype=float))
        t_start, t_end = ts[0], ts[-1]
        duration = max(t_end - t_start, 1.0)
        n_bins = max(int(np.ceil(duration / bin_size_sec)), 1)
        bin_edges = np.linspace(t_start, t_start + n_bins * bin_size_sec, n_bins + 1)
        counts, _ = np.histogram(ts, bins=bin_edges)
        plot_x, plot_y, plot_w, plot_h = 90, 140, 820, 380
        cv2.rectangle(img, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), self.PANEL_BORDER, 1)
        avg_freq = len(ts) / (duration / 60.0)
        cv2.putText(img, f'Total: {len(ts)} updates | Duration: {duration:.1f}s | Avg Rate: {avg_freq:.2f} updates/min', (plot_x, plot_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_MAIN, 1, cv2.LINE_AA)
        max_count = max(int(np.max(counts)), 1)
        grid_steps = 4
        for i in range(grid_steps + 1):
            y_val = plot_y + plot_h - int(i * plot_h / grid_steps)
            cv2.line(img, (plot_x, y_val), (plot_x + plot_w, y_val), self.tertiary_color, 1, cv2.LINE_AA)
            cv2.putText(img, f'{int(i * max_count / grid_steps)}', (plot_x - 35, y_val + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.neutral_gray, 1, cv2.LINE_AA)
        bar_w = max(int(plot_w / n_bins) - 3, 2)
        curve_pts = []
        for i, count in enumerate(counts):
            bar_h = int((count / max_count) * (plot_h - 20))
            bx, by = plot_x + int(i * plot_w / n_bins) + 2, plot_y + plot_h - bar_h
            cv2.rectangle(img, (bx, by), (bx + bar_w, plot_y + plot_h), self.secondary_color, -1)
            cv2.rectangle(img, (bx, by), (bx + bar_w, plot_y + plot_h), self.primary_color, 1)
            curve_pts.append((bx + bar_w // 2, by))
        if len(curve_pts) > 1:
            for i in range(len(curve_pts) - 1): cv2.line(img, curve_pts[i], curve_pts[i + 1], self.BORDER_COLOR, 2, cv2.LINE_AA)
            for pt in curve_pts: cv2.circle(img, pt, 3, self.primary_color, -1, cv2.LINE_AA)
        cv2.putText(img, f'Elapsed Time (Bins of {int(bin_size_sec)}s) ->', (plot_x + plot_w // 2 - 100, plot_y + plot_h + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT_MAIN, 1, cv2.LINE_AA)
