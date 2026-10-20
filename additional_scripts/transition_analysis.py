import json
from collections import Counter
import cv2
import numpy as np
import os
from agent.brain_network import BrainNetwork
from supervisor.renderer import Renderer


class VennPic(Renderer):
    """Venn diagram renderer using area-proportional circular geometry."""
    def draw_venn_diagram(self, img, stats, center=(500, 345), max_box_r=230):
        cx, cy = center

        val_true = stats.get("TRUE", 0)
        val_false = stats.get("FALSE", 0)
        val_conflict = stats.get("CONFLICT", 0)
        val_unknown = stats.get("UNKNOWN", 0)

        total_true = val_true + val_conflict
        total_false = val_false + val_conflict
        total_nodes = total_true + total_false - val_conflict + val_unknown

        if total_nodes <= 0:
            return

        font_header = cv2.FONT_HERSHEY_DUPLEX
        font_num = cv2.FONT_HERSHEY_SIMPLEX

        # ------------------------------------------------------------------
        # Case 1: 100% TRUE (All nodes and TRUE are identical circles)
        # ------------------------------------------------------------------
        if val_true == total_nodes:
            r = int(min(max_box_r * 0.75, 170))
            overlay = img.copy()

            # Fill with TRUE color
            fill_true = (175, 230, 180)
            cv2.circle(overlay, (cx, cy), r, fill_true, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

            # Identical outer border
            cv2.circle(img, (cx, cy), r, self.status_green, 2, cv2.LINE_AA)

            # Centered stats
            cv2.putText(
                img,
                f"ALL NODES = TRUE ({total_nodes})",
                (cx - 120, cy - 12),
                font_header,
                0.60,
                self.status_green,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                str(val_true),
                (cx - 20, cy + 32),
                font_num,
                1.0,
                self.TEXT_MAIN,
                2,
                cv2.LINE_AA,
            )
            return

        # ------------------------------------------------------------------
        # Case 2: Multi-set area calculation
        # ------------------------------------------------------------------
        # Total area mapped to circular space: Area ~ Total => R_total ~ sqrt(Total)
        scale = max_box_r / np.sqrt(max(total_nodes, 1))

        r_true = int(np.sqrt(total_true) * scale) if total_true > 0 else 0
        r_false = int(np.sqrt(total_false) * scale) if total_false > 0 else 0

        # Calculate circle displacement d
        if total_true > 0 and total_false > 0:
            min_pop = min(total_true, total_false)
            overlap_ratio = np.clip(val_conflict / min_pop, 0.0, 1.0) if min_pop > 0 else 0.0
            
            # d spans from touching perimeters (no conflict) to deeply overlapping
            d_max = (r_true + r_false) * 0.95
            d_min = abs(r_true - r_false) + 15
            d = d_max - overlap_ratio * (d_max - d_min)
            
            c_true = (int(cx - d / 2), cy + 12)
            c_false = (int(cx + d / 2), cy + 12)
        elif total_true > 0:
            c_true = (cx, cy + 12)
            c_false = None
            d = 0
        else:
            c_true = None
            c_false = (cx, cy + 12)
            d = 0

        # Calculate bounding box of the active circles
        min_x, max_x = cx, cx
        min_y, max_y = cy, cy

        if c_true is not None:
            min_x = min(min_x, c_true[0] - r_true)
            max_x = max(max_x, c_true[0] + r_true)
            min_y = min(min_y, c_true[1] - r_true)
            max_y = max(max_y, c_true[1] + r_true)

        if c_false is not None:
            min_x = min(min_x, c_false[0] - r_false)
            max_x = max(max_x, c_false[0] + r_false)
            min_y = min(min_y, c_false[1] - r_false)
            max_y = max(max_y, c_false[1] + r_false)

        # ------------------------------------------------------------------
        # Compute Universe Oval dimensions (a: semi-major, b: semi-minor)
        # ------------------------------------------------------------------
        pad_x = 45 + int(np.sqrt(val_unknown) * 7)
        pad_y = 40 + int(np.sqrt(val_unknown) * 6)

        semi_a = int(max((max_x - min_x) / 2 + pad_x, 140))
        semi_b = int(max((max_y - min_y) / 2 + pad_y, 110))

        # 1. Draw Universe Oval
        overlay_univ = img.copy()
        cv2.ellipse(overlay_univ, (cx, cy + 12), (semi_a, semi_b), 0, 0, 360, self.tertiary_color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay_univ, 0.70, img, 0.30, 0, img)
        cv2.ellipse(img, (cx, cy + 12), (semi_a, semi_b), 0, 0, 360, self.PANEL_BORDER, 2, cv2.LINE_AA)

        # Label Universe
        cv2.putText(
            img,
            f"ALL NODES ({total_nodes})",
            (cx - semi_a + 25, cy + 12 - semi_b + 32),
            font_header,
            0.50,
            self.TEXT_MAIN,
            1,
            cv2.LINE_AA,
        )
        if val_unknown > 0:
            cv2.putText(
                img,
                f"UNKNOWN: {val_unknown}",
                (cx - semi_a + 25, cy + 12 - semi_b + 52),
                font_num,
                0.45,
                self.neutral_gray,
                1,
                cv2.LINE_AA,
            )

        # 2. Draw Venn Circles
        overlay_circ = img.copy()
        fill_true = (175, 230, 180)   # Soft green (BGR)
        fill_false = (180, 180, 245)  # Soft red (BGR)

        if c_true is not None and r_true > 0:
            cv2.circle(overlay_circ, c_true, r_true, fill_true, -1, cv2.LINE_AA)
        if c_false is not None and r_false > 0:
            cv2.circle(overlay_circ, c_false, r_false, fill_false, -1, cv2.LINE_AA)

        cv2.addWeighted(overlay_circ, 0.55, img, 0.45, 0, img)

        if c_true is not None and r_true > 0:
            cv2.circle(img, c_true, r_true, self.status_green, 2, cv2.LINE_AA)
        if c_false is not None and r_false > 0:
            cv2.circle(img, c_false, r_false, self.status_red, 2, cv2.LINE_AA)

        # Helper to center title and count
        def draw_badge(canvas, title, count, pos, color_title, color_num):
            px, py = pos
            (tw1, th1), _ = cv2.getTextSize(title, font_header, 0.45, 1)
            cv2.putText(
                canvas, title, (px - tw1 // 2, py - 4), font_header, 0.45, color_title, 1, cv2.LINE_AA
            )
            (tw2, th2), _ = cv2.getTextSize(str(count), font_num, 0.70, 2)
            cv2.putText(
                canvas, str(count), (px - tw2 // 2, py + th2 + 6), font_num, 0.70, color_num, 2, cv2.LINE_AA
            )

        # 3. Text Placement
        if c_true is not None and r_true > 0:
            offset_t = int(r_true * 0.35) if c_false is not None else 0
            draw_badge(
                img,
                "TRUE",
                val_true,
                (c_true[0] - offset_t, c_true[1]),
                self.status_green,
                self.TEXT_MAIN,
            )

        if c_false is not None and r_false > 0:
            offset_f = int(r_false * 0.35) if c_true is not None else 0
            draw_badge(
                img,
                "FALSE",
                val_false,
                (c_false[0] + offset_f, c_false[1]),
                self.status_red,
                self.TEXT_MAIN,
            )

        if c_true is not None and c_false is not None and val_conflict > 0:
            draw_badge(
                img,
                "CONFLICT",
                val_conflict,
                (cx, cy + 12),
                (150, 40, 160),
                (40, 20, 60),
            )

    def render_transition_venn_card(self, preceeding, following, stats, width=1000, height=620):
        title = f"Skill Transition: {preceeding} -> {following}"
        subtitle = "Precondition Evaluation of Effect Nodes (Dynamic Area Overlap)"

        img = self.create_base_canvas(title=title, subtitle=subtitle, width=width, height=height)

        # Draw scaled oval universe + circles (no sample badge drawn at bottom)
        self.draw_venn_diagram(
            img=img,
            stats=stats,
            center=(width // 2, 345),
            max_box_r=190,
        )

        return img



SESSION = "2026-10-18_16-54"
SKILL_ORDER = [
    "Start", "Find_The_Door", "Go_To_The_Door", "Open_The_Door",
    "Find_The_Goal", "Go_To_The_Goal", "Finish"
]

brain = BrainNetwork(log_dir=f"logs/transition_analysis/{SESSION}")
brain.load_brain(f"logs/{SESSION}")
brain.prepare_nodes(SKILL_ORDER)
venn_output_dir = f"logs/transition_analysis/{SESSION}/diagrams"
os.makedirs(venn_output_dir, exist_ok=True)

stats = {}
renderer = VennPic()
for i, preceeding in enumerate(SKILL_ORDER[:-1]):
    following = SKILL_ORDER[i+1]
    if preceeding == following: continue
            
    src_skill = brain.skills.get(preceeding)
    tgt_skill = brain.skills.get(following)
    if not src_skill or not tgt_skill: continue

    eff_pred = src_skill["effect"]
    if not eff_pred.nodes or len(eff_pred.nodes) < 1: continue
            
    init_pred = tgt_skill["precondition"]
    if not init_pred.nodes or len(init_pred.nodes) < 1: continue

    points_stati = [
        init_pred.evaluate(node) for node in eff_pred.nodes
    ]

    counts = Counter(points_stati)
    stat = {
        "TRUE": counts.get("TRUE", 0),
        "FALSE": counts.get("FALSE", 0),
        "CONFLICT": counts.get("CONFLICT", 0),
        "UNKNOWN": counts.get("UNKNOWN", 0)
    }

    stats[preceeding] = stat

    img = renderer.render_transition_venn_card(
        preceeding=preceeding,
        following=following,
        stats=stat,
    )
    cv2.imwrite(f"{venn_output_dir}/venn_{preceeding}_to_{following}.png", img)

output_path_txt = f"logs/transition_analysis/{SESSION}/transition_stats.txt"
with open(output_path_txt, "w", encoding="utf-8") as f:
    for skill, counts_dict in stats.items():
        f.write(f"--- Transition from: {skill} ---\n")
        for key, val in counts_dict.items():
            f.write(f"  {key}: {val}\n")
        f.write("\n")

output_path_json = f"logs/transition_analysis/{SESSION}/transition_stats.json"
with open(output_path_json, "w", encoding="utf-8") as f:
    f.write(json.dumps(stats, indent=2))



