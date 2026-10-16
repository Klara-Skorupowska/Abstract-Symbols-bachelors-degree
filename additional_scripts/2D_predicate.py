import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from agent.predicate import Predicate, GNG

def save_animation_frame(fig, frame_nr: int, out_dir: str = "."):
    """Saves the current animation frame following '<frame nr>_2Dpredicate.png' format."""
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{frame_nr}_2Dpredicate.png"
    filepath = os.path.join(out_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved frame: {filepath}")

def draw_gng_manifold(gng: GNG, ax, sphere_color, edge_color, node_color, label_prefix=""):
    """Visualizes the receptive spheres, Delaunay graph edges, and nodes for a GNG manifold."""
    if not gng.nodes:
        return

    # 1. Local receptive spheres
    for i, node in enumerate(gng.nodes):
        radius = gng._get_local_radius(i)
        circle = plt.Circle(
            (node[0], node[1]),
            radius,
            facecolor=sphere_color,
            alpha=0.18,
            edgecolor=sphere_color,
            linewidth=1.2,
            zorder=2
        )
        ax.add_patch(circle)

    # 2. Topology edges
    for (u, v) in gng.edges.keys():
        if u < len(gng.nodes) and v < len(gng.nodes):
            p1, p2 = gng.nodes[u], gng.nodes[v]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=edge_color, alpha=0.6, linewidth=1.5, zorder=3)

    # 3. Prototype centroids / nodes
    nodes_arr = np.array(gng.nodes)
    ax.scatter(nodes_arr[:, 0], nodes_arr[:, 1], c=node_color, marker='o', s=30, zorder=4,
               edgecolors='black', linewidths=0.5, label=f"{label_prefix} Nodes ({len(gng.nodes)})")


def plot_3valued_predicate_2d(predicate: Predicate, title="3-Valued Predicate Evaluation",
                              test_points=None, true_annulus=None, ax=None, grid_res=90):
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        show_plot = True

    ax.clear()

    # 1. Background Epistemic Decision Surface: Unknown (0), True (1), False (2)
    x = np.linspace(0, 1, grid_res)
    y = np.linspace(0, 1, grid_res)
    xx, yy = np.meshgrid(x, y)
    grid_coords = np.c_[xx.ravel(), yy.ravel()]

    if predicate.gng_pos.nodes or predicate.gng_neg.nodes:
        state_map = {"UNKNOWN": 0, "TRUE": 1, "FALSE": 2, "CONFLICT": 2,}
        zz = np.array([state_map[predicate.evaluate(pt)] for pt in grid_coords]).reshape(xx.shape)

        # Discrete palette: Unknown -> muted light gray, True -> soft green, False -> soft red
        cmap = ListedColormap(['#f2f4f8', '#c8e6c9', '#ffcdd2'])
        ax.imshow(
            zz,
            extent=[0, 1, 0, 1],
            origin='lower',
            cmap=cmap,
            vmin=0,
            vmax=2,
            alpha=0.45,
            zorder=0
        )

    # 2. Target Ground Truth Annulus (2.0 < r < 4.0)
    if true_annulus is not None:
        center, r_inner, r_outer = true_annulus
        outer_c = plt.Circle(center, r_outer, facecolor='none', edgecolor='darkgreen',
                             linestyle='--', linewidth=2, label="Ground Truth (True Region)", zorder=1)
        inner_c = plt.Circle(center, r_inner, facecolor='none', edgecolor='darkgreen',
                             linestyle='--', linewidth=2, zorder=1)
        ax.add_patch(outer_c)
        ax.add_patch(inner_c)

    # 3. Batch training points
    if test_points:
        pos_pts = [p for p, val in test_points if val]
        neg_pts = [p for p, val in test_points if not val]
        if pos_pts:
            pos_arr = np.array(pos_pts)
            ax.scatter(pos_arr[:, 0], pos_arr[:, 1], c='forestgreen', alpha=0.35, s=12, zorder=1)
        if neg_pts:
            neg_arr = np.array(neg_pts)
            ax.scatter(neg_arr[:, 0], neg_arr[:, 1], c='crimson', alpha=0.25, s=12, zorder=1)

    # 4. Draw Positive and Negative Manifolds
    draw_gng_manifold(predicate.gng_pos, ax, sphere_color='mediumseagreen',
                      edge_color='forestgreen', node_color='darkgreen', label_prefix="Positive (True)")
    draw_gng_manifold(predicate.gng_neg, ax, sphere_color='salmon',
                      edge_color='firebrick', node_color='darkred', label_prefix="Negative (False)")

    # 5. Styling and Legend
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, pad=10)
    ax.grid(True, linestyle=':', alpha=0.4)

    region_patches = [
        Patch(facecolor='#c8e6c9', edgecolor='darkgreen', label='Predicate: TRUE'),
        Patch(facecolor='#ffcdd2', edgecolor='darkred', label='Predicate: FALSE'),
        Patch(facecolor='#f2f4f8', edgecolor='gray', label='Predicate: UNKNOWN')
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + region_patches, loc='upper right', framealpha=0.9, fontsize=8)

    if show_plot:
        plt.tight_layout()
        plt.show()


# ==========================================
# ANIMATION CONFIGURATION
# ==========================================

pred = Predicate(
    max_points=20,
    dim=2,
    base_radius=0.1,
    max_radius=0.7,
    learning_rate_b=0.2,
    learning_rate_n=0.01,
    max_edge_age=15,
    lambda_step=5,
    alpha=0.7,
    d=0.99
)

num_frames_to_save = 3

# Ground truth region:
center = np.array([0.5, 0.5])
r_inner, r_outer = 0.15, 0.35
target_shape = (center, r_inner, r_outer)

fig, ax = plt.subplots(figsize=(9, 9))
updates_per_frame = 40
total_frames = 120

def animate(frame):
    current_batch_points = []

    for _ in range(updates_per_frame):
        pt = np.random.uniform(0, 1, 2)
        status = pred.evaluate(pt)
        dist = np.linalg.norm(pt - center)
        valence = r_inner < dist < r_outer
        is_unknown = (status == "UNKNOWN")
        is_conflicted = (status == "CONFLICT")
        mismatch = (status == "TRUE" and not valence) or (status == "FALSE" and valence)
        if valence or mismatch or is_unknown or is_conflicted:
            pred.update(pt, valence=valence)
        current_batch_points.append((pt, valence))

    total_updates = (frame + 1) * updates_per_frame
    plot_3valued_predicate_2d(
        predicate=pred,
        title=f"3-Valued Predicate Learning — Step {total_updates}",
        test_points=current_batch_points,
        true_annulus=target_shape,
        ax=ax,
        grid_res=100
    )

    if frame < num_frames_to_save:
        save_animation_frame(fig, frame_nr=frame, out_dir="./pictures/predicate")

print("Starting 4-valued predicate animation...")
ani = animation.FuncAnimation(fig, animate, frames=total_frames, interval=40, repeat=False)
plt.tight_layout()
plt.show()

# first, middle, last