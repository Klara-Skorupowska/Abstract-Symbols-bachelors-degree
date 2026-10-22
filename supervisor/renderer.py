import numpy as np
import cv2
import textwrap
import math
import networkx as nx

SKILL_ORDER = [
    "Start", "Find_The_Door", "Go_To_The_Door", "Open_The_Door",
    "Find_The_Goal", "Go_To_The_Goal", "Finish"
]
SKILL_ORDER = [skill.lower() for skill in SKILL_ORDER]

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

        # Saturation & Value/Brightness calculations
        v = cmax
        s = np.zeros_like(v)
        nonzero_v = v > 1e-6
        s[nonzero_v] = delta[nonzero_v] / v[nonzero_v]

        # Hue calculation
        h = np.zeros_like(r)
        nonzero_delta = delta > 1e-6

        mask = nonzero_delta & (cmax == r)
        h[mask] = (60.0 * (((g[mask] - b[mask]) / delta[mask]) % 6))

        mask = nonzero_delta & (cmax == g)
        h[mask] = (60.0 * (((b[mask] - r[mask]) / delta[mask]) + 2))

        mask = nonzero_delta & (cmax == b)
        h[mask] = (60.0 * (((r[mask] - g[mask]) / delta[mask]) + 4))

        # Circular statistics for Hue
        angles = np.deg2rad(h)
        sin_mean = np.mean(np.sin(angles))
        cos_mean = np.mean(np.cos(angles))
        R = np.hypot(sin_mean, cos_mean)
        circ_var = float(1.0 - R)

        mean_angle = np.arctan2(sin_mean, cos_mean)
        diff_angles = np.arctan2(np.sin(angles - mean_angle), np.cos(angles - mean_angle))

        median_diff = lower_median(diff_angles)
        median_angle = (mean_angle + median_diff + 2 * np.pi) % (2 * np.pi)
        median_angle_deg = np.rad2deg(median_angle)

        # Compute representative median Saturation and Value
        median_s = float(lower_median(s))
        median_v = float(lower_median(v))

        # Map to OpenCV HSV 8-bit scales: H in [0, 179], S in [0, 255], V in [0, 255]
        h_cv = int(np.clip(median_angle_deg / 2.0, 0, 179))
        s_cv = int(np.clip(median_s * 255.0, 0, 255))
        v_cv = int(np.clip(median_v * 255.0, 0, 255))

        hsv_pixel = np.uint8([[[h_cv, s_cv, v_cv]]])
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

        # Ensure lowercase skill order matches lowercase node names
        order_lower = [s.lower() for s in SKILL_ORDER]
        ordered_nodes = [n for n in order_lower if n in nodes]
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

            # Green for 1.0 connections, default palette for others
            if weight >= 1.0:
                edge_color = self.status_green
            else:
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

            display_name = node.replace("_", " ").title()
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
