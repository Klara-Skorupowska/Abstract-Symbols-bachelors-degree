import cv2
import numpy as np
from scipy.stats import gaussian_kde
from supervisor.utils.core_utils import lower_median

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
            rad = np.deg2rad(-ang - 90)
            cv2.line(img, center, (int(center[0] + max_radius * np.cos(rad)), int(center[1] + max_radius * np.sin(rad))), self.tertiary_color, 2, cv2.LINE_AA)
        for i, ang in enumerate(self.LIDAR_ANGLES):
            if active_flags is not None and not active_flags[i]: continue
            rad = np.deg2rad(-ang - 90)
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

    def draw_kde_plot(self, img, epoch_datasets: list[tuple[str, np.ndarray]], bw_method=None):
        """
        Renders a sequential, continuous KDE plot across chained epochs where each
        epoch follows the previous one, showing both local update density and
        cumulative data accumulation.
        """
        valid_series = [(label, np.sort(np.asarray(ts, dtype=float))) for label, ts in epoch_datasets if len(ts) >= 2]
        if not valid_series:
            cv2.putText(img, 'Insufficient timestamp data for KDE', (320, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.neutral_gray, 2, cv2.LINE_AA)
            return

        plot_x, plot_y, plot_w, plot_h = 90, 140, 820, 380

        # Concatenate epochs sequentially along continuous active time
        concat_ts = []
        boundaries = []  # (start_t, end_t, label, count)
        cursor_t = 0.0

        for label, ts in valid_series:
            rel = ts - ts[0]
            dur = max(float(rel[-1]), 1e-3)
            shifted = rel + cursor_t
            concat_ts.append(shifted)
            boundaries.append((cursor_t, cursor_t + dur, label, len(ts)))
            cursor_t += dur

        all_ts = np.concatenate(concat_ts)
        total_duration = max(cursor_t, 1.0)
        total_samples = len(all_ts)

        # Use a fixed temporal bandwidth in seconds (default: 120s instead of Scott's ~16,000s)
        bw_seconds = float(bw_method) if bw_method is not None else 120.0
        std_ts = float(np.std(all_ts, ddof=1))
        if std_ts <= 0:
            cv2.putText(img, 'Zero variance in timestamps for KDE', (320, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.neutral_gray, 2, cv2.LINE_AA)
            return

        # Boundary reflection within 3*bw_seconds so edges don't artificially drop to half-height
        pad = 3.0 * bw_seconds
        left_reflect = -all_ts[all_ts < pad]
        right_reflect = 2.0 * total_duration - all_ts[all_ts > (total_duration - pad)]
        padded_ts = np.concatenate([left_reflect, all_ts, right_reflect])
        padded_std = float(np.std(padded_ts, ddof=1))

        try:
            kde = gaussian_kde(padded_ts, bw_method=bw_seconds / padded_std)
            x_grid = np.linspace(0.0, total_duration, plot_w)  # 1 evaluation per horizontal pixel
            rates = kde(x_grid) * len(padded_ts)
        except Exception:
            cv2.putText(img, 'Failed to compute KDE curve', (340, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.neutral_gray, 2, cv2.LINE_AA)
            return

        max_rate = max(float(np.max(rates)), 1e-6)

        palette = [
            self.primary_color,
            self.status_green,
            (180, 90, 140),
            (60, 140, 180),
            self.status_red,
            self.BORDER_COLOR,
        ]

        # 1. Alternating epoch background bands & boundary dividers
        for idx, (t0, t1, label, count) in enumerate(boundaries):
            x0 = plot_x + int((t0 / total_duration) * plot_w)
            x1 = plot_x + int((t1 / total_duration) * plot_w)
            if idx % 2 == 1:
                cv2.rectangle(img, (x0, plot_y), (x1, plot_y + plot_h), (240, 242, 245), -1)
            if idx > 0:
                for y_dash in range(plot_y, plot_y + plot_h, 10):
                    cv2.line(img, (x0, y_dash), (x0, min(y_dash + 5, plot_y + plot_h)), self.neutral_gray, 1, cv2.LINE_AA)

            # Epoch header label inside top of band
            band_w = x1 - x0
            if band_w > 45:
                short_lbl = f"E{idx + 1} (+{count})" if band_w < 110 else f"E{idx + 1}: {label[:10]} (+{count})"
                cv2.putText(img, short_lbl, (x0 + 6, plot_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, palette[idx % len(palette)], 1, cv2.LINE_AA)

        cv2.rectangle(img, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), self.PANEL_BORDER, 1)

        # 2. Horizontal grid lines & Left Y-axis (Update Rate: updates/sec)
        grid_steps = 4
        for i in range(grid_steps + 1):
            y_val = plot_y + plot_h - int(i * plot_h / grid_steps)
            cv2.line(img, (plot_x, y_val), (plot_x + plot_w, y_val), self.tertiary_color, 1, cv2.LINE_AA)
            tick_val = (i * max_rate) / grid_steps
            cv2.putText(img, f'{tick_val:.2f}', (plot_x - 45, y_val + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.neutral_gray, 1, cv2.LINE_AA)

        # X-axis ticks (Cumulative Active Time)
        for i in range(5):
            x_val = plot_x + int(i * plot_w / 4)
            t_tick = (i * total_duration) / 4.0
            cv2.line(img, (x_val, plot_y + plot_h), (x_val, plot_y + plot_h + 5), self.neutral_gray, 1, cv2.LINE_AA)
            cv2.putText(img, f'{t_tick:.0f}s', (x_val - 12, plot_y + plot_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.neutral_gray, 1, cv2.LINE_AA)

        # 3. Draw KDE curve colored per sequential epoch with translucent fill
        overlay = img.copy()
        for idx, (t0, t1, _, _) in enumerate(boundaries):
            color = palette[idx % len(palette)]
            mask = (x_grid >= t0) & (x_grid <= t1)
            if idx > 0 and np.any(mask):
                # Include boundary point so segments connect seamlessly
                first_idx = np.argmax(mask)
                if first_idx > 0:
                    mask[first_idx - 1] = True

            seg_x = x_grid[mask]
            seg_r = rates[mask]
            if len(seg_x) < 2:
                continue

            pts = [
                (
                    plot_x + int((gx / total_duration) * plot_w),
                    plot_y + plot_h - int((r / max_rate) * (plot_h - 28))
                )
                for gx, r in zip(seg_x, seg_r)
            ]
            pts_np = np.array(pts, dtype=np.int32)
            poly_pts = np.vstack([[(pts[0][0], plot_y + plot_h)], pts_np, [(pts[-1][0], plot_y + plot_h)]])
            cv2.fillPoly(overlay, [poly_pts], color)
            cv2.polylines(img, [pts_np], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.20, img, 0.80, 0, img)

        # 4. Linear trend line across KDE update rate
        if len(x_grid) >= 2:
            slope, intercept = np.polyfit(x_grid, rates, 1)
            y_start = float(np.clip(intercept, 0.0, max_rate))
            y_end = float(np.clip(slope * total_duration + intercept, 0.0, max_rate))

            pt1 = (plot_x, plot_y + plot_h - int((y_start / max_rate) * (plot_h - 28)))
            pt2 = (plot_x + plot_w, plot_y + plot_h - int((y_end / max_rate) * (plot_h - 28)))

            cv2.line(img, pt1, pt2, self.BORDER_COLOR, 1, cv2.LINE_AA)

            slope_per_hr = slope * 3600.0
            trend_label = f"Trend: {slope_per_hr:+.4f} (updates/s)/hr"
            cv2.putText(img, trend_label, (plot_x + 150, plot_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.BORDER_COLOR, 1, cv2.LINE_AA)

        # Axis titles & Legend note
        cv2.putText(img, 'Rate (updates/s)', (plot_x - 55, plot_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.primary_color, 1, cv2.LINE_AA)
        cv2.putText(img, 'Cumulative Active Time Across Sequential Epochs (s) ->', (plot_x + plot_w // 2 - 165, plot_y + plot_h + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.46, self.TEXT_MAIN, 1, cv2.LINE_AA)

    def draw_log_mode_bins(
        self,
        img,
        bins_data: list[dict],
        bin_size: int = 100,
        epoch_boundaries: list[tuple[int, str, int]] | None = None
    ):
        """
        Draws a binned stacked bar chart showing EXPLORE vs PLAN counts,
        an overlaid line/marker series and linear trend line for COLLECTED tag counts per bin,
        and vertical dashed dividers marking transitions between chained epochs.

        bins_data: list of dicts with keys:
            {'total': int, 'explore': int, 'plan': int, 'collected': int}
        epoch_boundaries: list of (start_log_index, epoch_label, new_logs_count)
        """
        if not bins_data:
            cv2.putText(img, 'Insufficient log data in updatefreq.txt', (310, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.neutral_gray, 2, cv2.LINE_AA)
            return

        plot_x, plot_y, plot_w, plot_h = 90, 145, 820, 370
        n_bins = len(bins_data)
        max_y = max(max(b['total'] for b in bins_data), bin_size, 1)

        total_logs = sum(b['total'] for b in bins_data)
        tot_explore = sum(b['explore'] for b in bins_data)
        tot_plan = sum(b['plan'] for b in bins_data)
        tot_collected = sum(b['collected'] for b in bins_data)

        # Header stats
        stats_text = (
            f"Total: {total_logs} | Explore: {tot_explore} ({tot_explore / total_logs * 100:.1f}%) | "
            f"Plan: {tot_plan} ({tot_plan / total_logs * 100:.1f}%) | "
            f"Collected: {tot_collected} ({tot_collected / total_logs * 100:.1f}%)"
        )
        cv2.putText(img, stats_text, (plot_x, plot_y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, self.TEXT_MAIN, 1, cv2.LINE_AA)

        # Legend (top-right above plot)
        legend_items = [
            ("EXPLORE", self.tertiary_color, "box"),
            ("PLAN", self.secondary_color, "box"),
            ("COLLECTED", self.primary_color, "line"),
        ]
        lx = plot_x + plot_w - 310
        ly = plot_y - 8
        for lbl, col, kind in legend_items:
            if kind == "box":
                cv2.rectangle(img, (lx, ly - 10), (lx + 12, ly + 2), col, -1)
                cv2.rectangle(img, (lx, ly - 10), (lx + 12, ly + 2), self.BORDER_COLOR, 1)
            else:
                cv2.line(img, (lx, ly - 4), (lx + 14, ly - 4), col, 2, cv2.LINE_AA)
                cv2.circle(img, (lx + 7, ly - 4), 4, col, -1, cv2.LINE_AA)
            cv2.putText(img, lbl, (lx + 18, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.TEXT_MAIN, 1, cv2.LINE_AA)
            lx += 105

        slot_w = plot_w / n_bins

        # 1. Alternating epoch background bands (drawn behind grid and bars)
        if epoch_boundaries and len(epoch_boundaries) > 1:
            for idx, (start_idx, _, _) in enumerate(epoch_boundaries):
                end_idx = epoch_boundaries[idx + 1][0] if idx + 1 < len(epoch_boundaries) else total_logs
                x0 = plot_x + int((start_idx / bin_size) * slot_w)
                x1 = min(plot_x + int((end_idx / bin_size) * slot_w), plot_x + plot_w)
                if idx % 2 == 1 and x1 > x0:
                    cv2.rectangle(img, (x0, plot_y), (x1, plot_y + plot_h), (238, 240, 244), -1)

        # Plot border & horizontal grid lines
        cv2.rectangle(img, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), self.PANEL_BORDER, 1)
        grid_steps = 4
        for i in range(grid_steps + 1):
            y_val = plot_y + plot_h - int(i * plot_h / grid_steps)
            cv2.line(img, (plot_x, y_val), (plot_x + plot_w, y_val), self.tertiary_color, 1, cv2.LINE_AA)
            tick_val = int(round((i * max_y) / grid_steps))
            cv2.putText(img, f'{tick_val}', (plot_x - 38, y_val + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.neutral_gray, 1, cv2.LINE_AA)

        # 2. Draw stacked bars (EXPLORE bottom, PLAN top) and collect points for COLLECTED curve
        bar_w = max(int(slot_w * 0.72), 4)
        collected_pts = []

        for i, b in enumerate(bins_data):
            cx = plot_x + int((i + 0.5) * slot_w)
            bx0 = cx - bar_w // 2
            bx1 = bx0 + bar_w
            base_y = plot_y + plot_h

            # EXPLORE bar (tertiary_color)
            exp_h = int((b['explore'] / max_y) * plot_h)
            exp_y0 = base_y - exp_h
            if exp_h > 0:
                cv2.rectangle(img, (bx0, exp_y0), (bx1, base_y), self.tertiary_color, -1)
                cv2.rectangle(img, (bx0, exp_y0), (bx1, base_y), self.BORDER_COLOR, 1)

            # PLAN bar stacked on top of EXPLORE (secondary_color)
            plan_h = int((b['plan'] / max_y) * plot_h)
            plan_y0 = exp_y0 - plan_h
            if plan_h > 0:
                cv2.rectangle(img, (bx0, plan_y0), (bx1, exp_y0), self.secondary_color, -1)
                cv2.rectangle(img, (bx0, plan_y0), (bx1, exp_y0), self.BORDER_COLOR, 1)

            # COLLECTED point (primary_color)
            col_h = int((b['collected'] / max_y) * plot_h)
            col_y = base_y - col_h
            collected_pts.append((cx, col_y, b['collected']))

            # X-axis tick labels (bin range)
            if n_bins <= 12 or i % max(1, n_bins // 10) == 0:
                start_log = i * bin_size + 1
                end_log = i * bin_size + b['total']
                tick_lbl = f"{start_log}-{end_log}"
                (tw, _), _ = cv2.getTextSize(tick_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                cv2.putText(img, tick_lbl, (cx - tw // 2, plot_y + plot_h + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.neutral_gray, 1, cv2.LINE_AA)

        # 3. Draw COLLECTED curve, markers, and linear trend line
        if len(collected_pts) > 1:
            for i in range(len(collected_pts) - 1):
                cv2.line(img, collected_pts[i][:2], collected_pts[i + 1][:2],
                         self.primary_color, 2, cv2.LINE_AA)

            # Linear trend line for COLLECTED across bins (normalized per bin_size logs)
            x_bins = np.arange(n_bins, dtype=float)
            y_rates = np.array(
                [(b['collected'] / max(b['total'], 1)) * bin_size for b in bins_data],
                dtype=float
            )
            slope, intercept = np.polyfit(x_bins, y_rates, 1)
            y_start = float(np.clip(intercept, 0.0, max_y))
            y_end = float(np.clip(slope * (n_bins - 1) + intercept, 0.0, max_y))

            pt1 = (collected_pts[0][0], plot_y + plot_h - int((y_start / max_y) * plot_h))
            pt2 = (collected_pts[-1][0], plot_y + plot_h - int((y_end / max_y) * plot_h))
            cv2.line(img, pt1, pt2, self.BORDER_COLOR, 1, cv2.LINE_AA)

            trend_label = f"Collected Trend: {slope:+.2f} / {bin_size} logs"
            cv2.putText(img, trend_label, (plot_x, plot_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.BORDER_COLOR, 1, cv2.LINE_AA)

        for cx, cy, val in collected_pts:
            cv2.circle(img, (cx, cy), 4, self.primary_color, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 5, (255, 255, 255), 1, cv2.LINE_AA)
            if n_bins <= 18:
                cv2.putText(img, str(val), (cx - 8, max(cy - 8, plot_y + 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.primary_color, 1, cv2.LINE_AA)

        # 4. Epoch boundary dashed dividers & labels (without count)
        if epoch_boundaries:
            for idx, (start_idx, label, _) in enumerate(epoch_boundaries):
                end_idx = epoch_boundaries[idx + 1][0] if idx + 1 < len(epoch_boundaries) else total_logs
                x0 = plot_x + int((start_idx / bin_size) * slot_w)
                x1 = min(plot_x + int((end_idx / bin_size) * slot_w), plot_x + plot_w)

                if idx > 0:
                    for y_dash in range(plot_y, plot_y + plot_h, 10):
                        cv2.line(img, (x0, y_dash), (x0, min(y_dash + 5, plot_y + plot_h)),
                                 self.BORDER_COLOR, 1, cv2.LINE_AA)
                    tri = np.array([[x0 - 4, plot_y], [x0 + 4, plot_y], [x0, plot_y + 6]], dtype=np.int32)
                    cv2.fillPoly(img, [tri], self.BORDER_COLOR)

                band_w = x1 - x0
                if band_w > 28:
                    ep_lbl = f"E{idx + 1}" if band_w < 95 else f"E{idx + 1}: {label[:12]}"
                    cv2.putText(img, ep_lbl, (x0 + 5, plot_y + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.BORDER_COLOR, 1, cv2.LINE_AA)

        cv2.putText(img, f'Consecutive Log Bins (Bin Size: {bin_size} logs) ->',
                    (plot_x + plot_w // 2 - 145, plot_y + plot_h + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, self.TEXT_MAIN, 1, cv2.LINE_AA)