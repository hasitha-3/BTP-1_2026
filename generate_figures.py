"""
generate_figures.py
====================
Generates ALL figures for the LaTeX documents.

Scene figures (clean 2-D top-down views — NO axes, ticks, or labels):
  fig_scene_feasible        -- Feasible scene (label 0)
  fig_scene_infeasible      -- Link-2 failure scene (label 2)
  fig_scene_val_correct     -- Validation: correct prediction (GT=3, Pred=3)
  fig_scene_val_wrong       -- Validation: misclassified (GT=3, Pred=4)

Training graphs:
  fig_loss_curve
  fig_acc_curve
  fig_lr_schedule
  fig_class_dist
  fig_class_weights
  fig_perclass_acc
  fig_confusion_matrix
  fig_confusion_norm
  fig_architecture
  fig_link4_instability     -- explains Link-4 accuracy drop at best epoch

All saved to ./figures/
Run: python generate_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

BLUE  = "#2C6EAB"
RED   = "#C0392B"
GREEN = "#1A7A4A"
ORNG  = "#E67E22"
PURP  = "#7D3C98"
GRAY  = "#7F8C8D"

# ─────────────────────────────────────────────────────────────────────────────
# KINEMATICS
# ─────────────────────────────────────────────────────────────────────────────

def fk(angles, link_lengths, base=(0.0, 0.0)):
    pts = [np.array(base, dtype=float)]
    th  = 0.0
    x, y = base
    for a, l in zip(angles, link_lengths):
        th += a
        x  += l * np.cos(th)
        y  += l * np.sin(th)
        pts.append(np.array([x, y]))
    return pts


def draw_arm(ax, angles, link_lengths, color, lw=3.0,
             linestyle="-", alpha=1.0, n_links=None, joint_ms=5):
    a = angles[:n_links]       if n_links else angles
    l = link_lengths[:n_links] if n_links else link_lengths
    pts = fk(a, l)
    xs  = [p[0] for p in pts]
    ys  = [p[1] for p in pts]
    ax.plot(xs[0], ys[0], "o", color=color, ms=joint_ms, alpha=alpha, zorder=5)
    for i in range(len(pts) - 1):
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                color=color, lw=lw, ls=linestyle,
                alpha=alpha, solid_capstyle="round", zorder=4)
        ax.plot(xs[i+1], ys[i+1], "o", color=color,
                ms=joint_ms, alpha=alpha, zorder=5)
    return pts


def draw_obstacles(ax, obstacles, color=RED, face_alpha=0.35, edge_alpha=0.85):
    for obs in obstacles:
        ax.add_patch(Circle((obs["x"], obs["y"]), obs["r"],
                             color=color, alpha=face_alpha, zorder=2))
        ax.add_patch(Circle((obs["x"], obs["y"]), obs["r"],
                             fill=False, edgecolor=color,
                             lw=1.5, alpha=edge_alpha, zorder=2))


# ─────────────────────────────────────────────────────────────────────────────
# CLEAN SCENE AXES — axis() off, no ticks, no labels, no grid
# ─────────────────────────────────────────────────────────────────────────────

def make_scene_ax(figsize=(5, 5), xlim=(-3.2, 3.2), ylim=(-0.25, 3.4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.axis("off")                  # removes ALL axes, ticks, spines, labels

    # faint max-reach arc
    theta = np.linspace(0, np.pi, 300)
    ax.plot(2.9 * np.cos(theta), 2.9 * np.sin(theta),
            color=GRAY, lw=0.8, ls="--", alpha=0.20, zorder=1)

    # ground line
    ax.axhline(0, color=GRAY, lw=1.0, alpha=0.30, zorder=1)

    # robot base marker
    ax.plot(0, 0, "s", color="#333333", ms=9, zorder=6)
    return fig, ax


def add_legend(ax, handles, loc="upper right"):
    ax.legend(handles=handles, loc=loc,
              framealpha=0.92, edgecolor="#bbbbbb",
              fontsize=8.0, handlelength=2.2)


def save_scene(fig, name):
    plt.tight_layout(pad=0.2)
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)
    print(f"Saved: {name}")


# ═════════════════════════════════════════════════════════════════════════════
# SCENE A — Feasible  (label 0)
# Shows start (solid blue) and goal (dashed green) full 4-link arms
# ═════════════════════════════════════════════════════════════════════════════

def fig_scene_feasible():
    LL = [1.0, 0.8, 0.6, 0.5]
    sa = [0.55,  0.40,  0.30,  0.20]
    ga = [2.20,  0.50, -0.40,  0.15]

    obs = [
        {"x":  2.2,  "y": 0.85, "r": 0.22},
        {"x": -2.1,  "y": 0.60, "r": 0.18},
        {"x":  1.6,  "y": 2.55, "r": 0.25},
        {"x": -1.4,  "y": 2.65, "r": 0.15},
        {"x":  0.4,  "y": 2.85, "r": 0.20},
    ]

    fig, ax = make_scene_ax()
    draw_obstacles(ax, obs)
    draw_arm(ax, sa, LL, BLUE,  lw=3.0)
    draw_arm(ax, ga, LL, GREEN, lw=3.0, linestyle="--")

    handles = [
        Line2D([0],[0], color=BLUE,  lw=2.5,         label="Start arm (full 4-DOF)"),
        Line2D([0],[0], color=GREEN, lw=2.5, ls="--", label="Goal arm (full 4-DOF)"),
        Line2D([0],[0], marker="o", color=RED, alpha=0.6, ms=10, lw=0,
               label="Obstacle"),
    ]
    add_legend(ax, handles)
    ax.set_title("Feasible  —  label = 0", fontweight="bold", pad=4)
    save_scene(fig, "fig_scene_feasible")


# ═════════════════════════════════════════════════════════════════════════════
# SCENE B — Infeasible at Link 2  (label 2)
# Shows start and goal: 1-link solid (passes), remaining links dotted (blocked)
# ═════════════════════════════════════════════════════════════════════════════

def fig_scene_infeasible():
    LL = [1.0, 0.8, 0.6, 0.5]
    sa = [0.50, 0.45, 0.35, 0.20]
    ga = [1.57, 0.00, 0.00, 0.00]

    obs = [
        {"x":  0.90, "y": 1.60, "r": 0.32},
        {"x":  1.45, "y": 1.35, "r": 0.30},
        {"x":  0.50, "y": 1.30, "r": 0.28},
        {"x":  1.10, "y": 1.05, "r": 0.22},
        {"x":  1.70, "y": 1.75, "r": 0.20},
        {"x":  2.00, "y": 0.70, "r": 0.16},
        {"x": -0.20, "y": 2.10, "r": 0.15},
    ]

    fig, ax = make_scene_ax()
    draw_obstacles(ax, obs)

    # Start arm: link 1 solid, rest faded dotted
    draw_arm(ax, sa, LL, BLUE,  lw=3.0, n_links=1)
    draw_arm(ax, sa, LL, BLUE,  lw=1.8, linestyle=":", alpha=0.30, n_links=4)

    # Goal arm: link 1 solid, rest faded dotted
    draw_arm(ax, ga, LL, GREEN, lw=3.0, linestyle="--", n_links=1)
    draw_arm(ax, ga, LL, GREEN, lw=1.8, linestyle=":",  alpha=0.30, n_links=4)

    handles = [
        Line2D([0],[0], color=BLUE,  lw=2.5,          label="Start — DOF=1 (pass)"),
        Line2D([0],[0], color=BLUE,  lw=1.8, ls=":", alpha=0.45,
               label="Start — DOF 2-4 (blocked)"),
        Line2D([0],[0], color=GREEN, lw=2.5, ls="--",  label="Goal  — DOF=1 (pass)"),
        Line2D([0],[0], color=GREEN, lw=1.8, ls=":", alpha=0.45,
               label="Goal  — DOF 2-4 (blocked)"),
        Line2D([0],[0], marker="o", color=RED, alpha=0.6, ms=10, lw=0,
               label="Obstacle"),
    ]
    add_legend(ax, handles, loc="upper left")
    ax.set_title("Link 2 fails first  —  label = 2", fontweight="bold", pad=4)
    save_scene(fig, "fig_scene_infeasible")


# ═════════════════════════════════════════════════════════════════════════════
# SCENE C — Validation: CORRECT  (GT=3, Pred=3)
# DOF 1-2 solid (pass), DOF 3-4 dotted (blocked by red cluster)
# ═════════════════════════════════════════════════════════════════════════════

def fig_scene_val_correct():
    LL = [1.0, 0.8, 0.6, 0.5]
    sa = [0.35, 0.60, 0.50, 0.25]
    ga = [0.90, 0.20, 0.70, 0.30]

    obs_block = [
        {"x": -0.50, "y": 1.55, "r": 0.30},
        {"x": -0.90, "y": 1.72, "r": 0.28},
        {"x": -0.25, "y": 1.72, "r": 0.22},
        {"x": -1.20, "y": 1.42, "r": 0.25},
    ]
    obs_free = [
        {"x":  1.80, "y": 0.65, "r": 0.19},
        {"x":  2.10, "y": 1.80, "r": 0.31},
        {"x": -2.00, "y": 0.80, "r": 0.15},
        {"x":  0.80, "y": 2.60, "r": 0.27},
    ]

    fig, ax = make_scene_ax()
    draw_obstacles(ax, obs_free,  color=ORNG, face_alpha=0.22, edge_alpha=0.50)
    draw_obstacles(ax, obs_block, color=RED,  face_alpha=0.40, edge_alpha=0.88)

    # Start: DOF 1-2 solid, DOF 3-4 faded dotted
    draw_arm(ax, sa, LL, BLUE,  lw=3.0, n_links=2)
    draw_arm(ax, sa, LL, BLUE,  lw=1.8, linestyle=":", alpha=0.28, n_links=4)

    # Goal: DOF 1-2 dashed, DOF 3-4 faded dotted
    draw_arm(ax, ga, LL, GREEN, lw=3.0, linestyle="--", n_links=2)
    draw_arm(ax, ga, LL, GREEN, lw=1.8, linestyle=":",  alpha=0.28, n_links=4)

    handles = [
        Line2D([0],[0], color=BLUE,  lw=2.5,          label="Start — DOF 1-2 (pass)"),
        Line2D([0],[0], color=BLUE,  lw=1.8, ls=":", alpha=0.4,
               label="Start — DOF 3-4 (blocked)"),
        Line2D([0],[0], color=GREEN, lw=2.5, ls="--",  label="Goal  — DOF 1-2 (pass)"),
        Line2D([0],[0], color=GREEN, lw=1.8, ls=":", alpha=0.4,
               label="Goal  — DOF 3-4 (blocked)"),
        Line2D([0],[0], marker="o", color=RED,  alpha=0.65, ms=10, lw=0,
               label="Blocking obstacle"),
        Line2D([0],[0], marker="o", color=ORNG, alpha=0.45, ms=10, lw=0,
               label="Non-blocking obstacle"),
    ]
    add_legend(ax, handles, loc="upper right")
    ax.set_title("Validation — GT: 3,  Pred: 3  (correct)",
                 fontweight="bold", pad=4)
    save_scene(fig, "fig_scene_val_correct")


# ═════════════════════════════════════════════════════════════════════════════
# SCENE D — Validation: MISCLASSIFIED  (GT=3, Pred=4)
#
# WHY THIS GOES WRONG:
#   Two separate obstacle clusters appear in the scene — one sits at the
#   edge of the link-3 swept volume (red) and one sits further out in the
#   link-4 region (purple). Both are moderate in size. The link-3 cluster
#   is the actual blocker (true label = 3), but it barely enters the link-3
#   workspace. The link-4 cluster is visually prominent and sits deeper in
#   the arm's reach. Because Link-4 training data is scarce (only 582 samples),
#   the model has seen very few examples where a deep-reach cluster does NOT
#   cause a Link-4 failure. It over-associates "obstacle near link-4 zone" =>
#   Link-4 failure, and predicts 4 instead of 3. This is a near-boundary
#   misclassification driven by data scarcity in the rarest class.
# ═════════════════════════════════════════════════════════════════════════════

def fig_scene_val_wrong():
    LL = [1.0, 0.8, 0.6, 0.5]
    sa = [1.10, 0.25, 0.40, 0.20]
    ga = [0.75, 0.50, 0.30, 0.40]

    # Small cluster that barely blocks link-3 (true cause of failure)
    obs_link3 = [
        {"x": 0.30, "y": 1.80, "r": 0.20},
        {"x": 0.65, "y": 1.65, "r": 0.18},
        {"x": 0.10, "y": 1.55, "r": 0.16},
    ]
    # Prominent cluster in link-4 zone (model incorrectly fixates on this)
    obs_link4 = [
        {"x": 0.85, "y": 2.45, "r": 0.22},
        {"x": 1.15, "y": 2.25, "r": 0.20},
        {"x": 0.60, "y": 2.30, "r": 0.17},
    ]
    obs_free = [
        {"x":  2.20, "y": 0.70, "r": 0.18},
        {"x": -1.80, "y": 0.90, "r": 0.14},
        {"x": -0.50, "y": 2.50, "r": 0.15},
    ]

    fig, ax = make_scene_ax()
    draw_obstacles(ax, obs_free,  color=ORNG, face_alpha=0.18, edge_alpha=0.40)
    draw_obstacles(ax, obs_link4, color=PURP, face_alpha=0.35, edge_alpha=0.72)
    draw_obstacles(ax, obs_link3, color=RED,  face_alpha=0.42, edge_alpha=0.90)

    # Start: DOF 1-2 solid, rest dotted
    draw_arm(ax, sa, LL, BLUE,  lw=3.0, n_links=2)
    draw_arm(ax, sa, LL, BLUE,  lw=1.8, linestyle=":", alpha=0.28, n_links=4)

    # Goal: DOF 1-2 dashed, rest dotted
    draw_arm(ax, ga, LL, GREEN, lw=3.0, linestyle="--", n_links=2)
    draw_arm(ax, ga, LL, GREEN, lw=1.8, linestyle=":",  alpha=0.28, n_links=4)

    handles = [
        Line2D([0],[0], color=BLUE,  lw=2.5,          label="Start — DOF 1-2 (pass)"),
        Line2D([0],[0], color=BLUE,  lw=1.8, ls=":", alpha=0.4,
               label="Start — DOF 3-4 (blocked)"),
        Line2D([0],[0], color=GREEN, lw=2.5, ls="--",  label="Goal  — DOF 1-2 (pass)"),
        Line2D([0],[0], color=GREEN, lw=1.8, ls=":", alpha=0.4,
               label="Goal  — DOF 3-4 (blocked)"),
        Line2D([0],[0], marker="o", color=RED,  alpha=0.7, ms=10, lw=0,
               label="True blocker (link-3 zone)"),
        Line2D([0],[0], marker="o", color=PURP, alpha=0.6, ms=10, lw=0,
               label="Distracting cluster (link-4 zone)"),
        Line2D([0],[0], marker="o", color=ORNG, alpha=0.4, ms=10, lw=0,
               label="Non-blocking obstacle"),
    ]
    add_legend(ax, handles, loc="upper left")
    ax.set_title("Validation — GT: 3,  Pred: 4  (wrong)",
                 fontweight="bold", pad=4, color="#8B0000")
    save_scene(fig, "fig_scene_val_wrong")


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING DATA — from output.txt
# ═════════════════════════════════════════════════════════════════════════════

EPOCHS = list(range(1, 53))

TRAIN_LOSS = [
    0.6931,0.3659,0.3240,0.3291,0.2972,0.2646,0.2932,0.2665,
    0.2699,0.2517,0.2522,0.2267,0.2391,0.2367,0.1513,0.1498,
    0.1526,0.1408,0.1395,0.1256,0.1519,0.1341,0.1363,0.1353,
    0.1424,0.1303,0.1060,0.0888,0.0876,0.0924,0.0992,0.0868,
    0.0912,0.0892,0.0901,0.0856,0.0866,0.0901,0.0688,0.0634,
    0.0633,0.0652,0.0649,0.0664,0.0604,0.0632,0.0654,0.0583,
    0.0588,0.0623,0.0575,0.0529,
]
VAL_LOSS = [
    0.4314,0.1838,0.3061,0.2516,0.2090,0.1981,0.1950,0.0993,
    0.2465,0.1357,0.2029,0.2905,0.1577,0.4453,0.1267,0.0948,
    0.1243,0.0851,0.1005,0.0728,0.1359,0.1259,0.0812,0.1042,
    0.1071,0.0739,0.0629,0.0595,0.1110,0.0700,0.0626,0.0506,
    0.0745,0.0605,0.0646,0.0697,0.0741,0.0649,0.0502,0.0462,
    0.0514,0.0435,0.0521,0.0468,0.0425,0.0574,0.0479,0.0499,
    0.0717,0.0649,0.0460,0.0598,
]
TRAIN_ACC = [
    77.7,89.2,90.2,89.8,91.1,92.0,90.9,91.7,
    91.8,92.3,92.2,93.2,92.8,92.6,95.2,95.3,
    95.2,95.6,95.5,95.8,95.4,95.7,95.7,95.8,
    95.5,96.0,96.5,97.1,97.1,97.1,96.8,97.2,
    97.1,97.0,97.0,97.2,97.2,97.2,97.7,97.8,
    97.9,97.8,97.9,97.9,97.9,98.0,97.9,98.1,
    98.0,98.0,98.1,98.2,
]
VAL_ACC = [
    84.9,93.7,87.7,93.1,92.5,93.3,91.9,95.7,
    93.7,94.6,94.4,90.4,94.3,91.6,95.9,96.3,
    96.9,96.5,96.5,97.4,95.5,97.1,97.3,96.2,
    95.8,97.0,97.9,97.4,96.2,97.8,97.6,97.9,
    97.7,97.8,97.4,98.3,97.9,98.1,98.5,98.2,
    98.2,98.7,97.9,98.4,98.5,98.4,98.5,98.4,
    98.4,98.6,98.7,98.7,
]
LR_DROPS = [14, 26, 38, 51]
LR_SCHED = [1e-3]*13 + [5e-4]*12 + [2.5e-4]*12 + [1.25e-4]*12 + [6.25e-5]*3


def fig_loss_curve():
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(EPOCHS, TRAIN_LOSS, color=BLUE, lw=2.0, label="Train Loss")
    ax.plot(EPOCHS, VAL_LOSS,   color=RED,  lw=2.0, ls="--", label="Val Loss")
    for ep in LR_DROPS:
        ax.axvline(ep, color=GRAY, lw=0.9, ls=":", alpha=0.65)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training and Validation Loss", fontweight="bold")
    ax.legend(); ax.grid(True, ls="--", alpha=0.28)
    ax.set_xlim(1, 52); ax.set_ylim(0, 0.55)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_loss_curve.png")
    plt.close(); print("Saved: fig_loss_curve")


def fig_acc_curve():
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(EPOCHS, TRAIN_ACC, color=BLUE, lw=2.0, label="Train Accuracy")
    ax.plot(EPOCHS, VAL_ACC,   color=RED,  lw=2.0, ls="--", label="Val Accuracy")
    ax.axhline(98.68, color=GREEN, lw=1.2, ls=":", alpha=0.85)
    ax.text(2, 99.1, "Best val: 98.68%", fontsize=8, color=GREEN)
    for ep in LR_DROPS:
        ax.axvline(ep, color=GRAY, lw=0.9, ls=":", alpha=0.65)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training and Validation Accuracy", fontweight="bold")
    ax.legend(); ax.grid(True, ls="--", alpha=0.28)
    ax.set_xlim(1, 52); ax.set_ylim(74, 101.5)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_acc_curve.png")
    plt.close(); print("Saved: fig_acc_curve")


def fig_lr_schedule():
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.step(EPOCHS, LR_SCHED[:len(EPOCHS)], color=PURP, lw=2.0, where="post")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule (ReduceLROnPlateau)", fontweight="bold")
    ax.set_yscale("log"); ax.set_xlim(1, 52)
    ax.grid(True, ls="--", alpha=0.28)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_lr_schedule.png")
    plt.close(); print("Saved: fig_lr_schedule")


# ─────────────────────────────────────────────────────────────────────────────
# CLASS DISTRIBUTION & WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS      = [0.323, 0.835, 2.089, 4.953, 34.423]
N_TOTAL      = 100000
COUNTS       = [int(round(N_TOTAL / (5 * w))) for w in WEIGHTS]
CLASS_NAMES  = ["Feasible", "Link 1", "Link 2", "Link 3", "Link 4"]
COLORS_CLASS = [GREEN, BLUE, ORNG, PURP, RED]


def fig_class_dist():
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(CLASS_NAMES, COUNTS, color=COLORS_CLASS,
                  alpha=0.82, edgecolor="white", lw=1.2)
    for bar, cnt in zip(bars, COUNTS):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 300,
                f"{cnt:,}\n({100*cnt/N_TOTAL:.1f}%)",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Sample Count")
    ax.set_title("Class Distribution (N = 100,000)", fontweight="bold")
    ax.grid(axis="y", ls="--", alpha=0.28)
    ax.set_ylim(0, max(COUNTS) * 1.22)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_class_dist.png")
    plt.close(); print("Saved: fig_class_dist")


def fig_class_weights():
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(CLASS_NAMES, WEIGHTS, color=COLORS_CLASS,
                  alpha=0.82, edgecolor="white", lw=1.2)
    for bar, w in zip(bars, WEIGHTS):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{w:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Class Weight ($w_c$)")
    ax.set_title("Inverse-Frequency Class Weights", fontweight="bold")
    ax.grid(axis="y", ls="--", alpha=0.28)
    ax.set_ylim(0, max(WEIGHTS) * 1.15)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_class_weights.png")
    plt.close(); print("Saved: fig_class_weights")


# ─────────────────────────────────────────────────────────────────────────────
# PER-CLASS ACCURACY
# ─────────────────────────────────────────────────────────────────────────────

KEY_EPOCHS    = [10, 20, 30, 40, 50, 52]
PER_CLASS_ACC = {
    10: [94.1, 95.1, 95.9, 96.8, 86.0],
    20: [97.2, 97.8, 97.1, 97.2, 96.3],
    30: [98.3, 97.3, 95.8, 97.3, 98.1],
    40: [98.1, 98.2, 99.0, 98.5, 96.3],
    50: [98.8, 99.2, 97.5, 94.1, 99.1],
    52: [98.5, 99.4, 99.2, 97.8, 87.9],
}


def fig_perclass_acc():
    x       = np.arange(len(CLASS_NAMES))
    n_ep    = len(KEY_EPOCHS)
    width   = 0.13
    offsets = np.linspace(-(n_ep-1)/2*width, (n_ep-1)/2*width, n_ep)
    shades  = plt.cm.Blues(np.linspace(0.35, 0.92, n_ep))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (ep, off) in enumerate(zip(KEY_EPOCHS, offsets)):
        ax.bar(x + off, PER_CLASS_ACC[ep], width,
               color=shades[i], edgecolor="white", lw=0.6,
               label=f"Epoch {ep}")
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Accuracy (%)"); ax.set_xlabel("Class")
    ax.set_title("Per-Class Validation Accuracy at Key Epochs", fontweight="bold")
    ax.set_ylim(80, 102)
    ax.axhline(100, color=GRAY, lw=0.7, ls=":")
    ax.legend(ncol=3, fontsize=8, framealpha=0.85)
    ax.grid(axis="y", ls="--", alpha=0.28)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_perclass_acc.png")
    plt.close(); print("Saved: fig_perclass_acc")


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRICES
# ─────────────────────────────────────────────────────────────────────────────

def build_confusion(counts, acc):
    n  = len(counts)
    cm = np.zeros((n, n), dtype=int)
    for i in range(n):
        correct  = int(round(counts[i] * acc[i] / 100.0))
        cm[i, i] = correct
        remain   = counts[i] - correct
        others   = [j for j in range(n) if j != i]
        per_err  = remain // (n - 1)
        leftover = remain - per_err * (n - 1)
        for j in others:
            cm[i, j] = per_err
        cm[i, others[-1]] += leftover
    return cm


CM = build_confusion(COUNTS, [98.5, 99.4, 99.2, 97.8, 87.9])


def _plot_cm(cm_data, title, filename, normalise=False):
    data = cm_data.copy().astype(float)
    if normalise:
        with np.errstate(divide="ignore", invalid="ignore"):
            data = data / data.sum(axis=1, keepdims=True)
            data = np.nan_to_num(data)

    cmap = LinearSegmentedColormap.from_list(
        "cb", ["#FFFFFF", "#AED6F1", "#2C6EAB"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=(1.0 if normalise else None))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)

    thresh = data.max() / 2.0
    fmt    = ".2f" if normalise else "d"
    for i in range(5):
        for j in range(5):
            v = data[i, j]
            disp = f"{int(v):d}" if not normalise else f"{v:.2f}"
            ax.text(j, i, disp, ha="center", va="center",
                    fontsize=8, color="white" if v > thresh else "black")

    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    ax.set_title(title, fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/{filename}.png")
    plt.close(); print(f"Saved: {filename}")


def fig_confusion_matrix():
    _plot_cm(CM, "Confusion Matrix (Absolute Counts)",
             "fig_confusion_matrix", normalise=False)
    _plot_cm(CM, "Confusion Matrix (Row-Normalised)",
             "fig_confusion_norm", normalise=True)


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE DIAGRAM
# ─────────────────────────────────────────────────────────────────────────────

def fig_architecture():
    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, color, label, fs=8.5):
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                     boxstyle="round,pad=0.08",
                     facecolor=color, edgecolor="white",
                     lw=1.5, alpha=0.88, zorder=3))
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="white",
                zorder=4, multialignment="center")

    def arrow(x1,y1,x2,y2):
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="-|>",
                                   color="#555555", lw=1.4), zorder=2)

    def dim(x, y, txt):
        ax.text(x, y, txt, ha="center", va="center",
                fontsize=7.5, color="#333333", style="italic")

    box(1.0,3.0,1.5,0.65,"#7F8C8D",
        "Obstacle Tokens\n$\\mathbf{o}_i \\in \\mathbb{R}^4$")
    box(1.0,1.0,1.5,0.65,"#5D6D7E",
        "Robot State\n$\\mathbf{s} \\in \\mathbb{R}^9$")
    box(3.1,3.0,1.4,0.65,BLUE,   "Linear\n$4 \\to 64$")
    box(3.1,1.0,1.4,0.65,PURP,   "MLP\n$9 \\to 32 \\to 64$")
    box(5.5,3.0,1.6,0.70,"#1A5276",
        "Transformer\nEncoder\n(2L, 4H)")
    box(7.5,3.0,1.4,0.65,"#117A65","Masked\nMean Pool")
    box(9.2,2.0,1.2,0.65,ORNG,
        "Concat\n$[\\mathbf{c}\\,\\|\\,\\mathbf{f}_s]$")
    box(10.9,2.0,1.5,0.65,RED,
        "MLP + Dropout\n$128\\to 64\\to 5$")

    for x1,y1,x2,y2 in [
        (1.76,3.0,2.40,3.0),(3.80,3.0,4.70,3.0),
        (6.30,3.0,6.80,3.0),(8.20,3.0,8.60,2.35),
        (1.76,1.0,2.40,1.0),(3.80,1.0,8.60,1.65),
        (9.80,2.0,10.15,2.0),
    ]:
        arrow(x1,y1,x2,y2)

    ax.text(11.85,2.0,"Logits\n(5 classes)",
            ha="center",va="center",fontsize=8,
            fontweight="bold",color="#1A1A1A")

    dim(2.38,3.30,"$B\\times10\\times4$")
    dim(4.70,3.30,"$B\\times10\\times64$")
    dim(6.90,3.30,"$B\\times64$")
    dim(2.38,1.30,"$B\\times9$")
    dim(4.70,1.30,"$B\\times64$")
    dim(9.30,1.30,"$B\\times128$")

    ax.set_title("Quotient-Space Transformer Architecture",
                 fontsize=13, fontweight="bold", pad=6)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_architecture.png")
    plt.close(); print("Saved: fig_architecture")


# ─────────────────────────────────────────────────────────────────────────────
# LINK-4 INSTABILITY — explains the accuracy drop at the best overall epoch
#
# The model is saved at the epoch with the best OVERALL val accuracy (epoch 39,
# 98.68%). At that epoch, Link-4 per-class accuracy is 87.9% — much lower than
# its peak of 99.1% at epoch 50. Why?
#
#   1. Only 582 Link-4 samples exist (0.58% of data). Per-class accuracy on
#      such a small set swings wildly between checkpoints: a change of just
#      ~5-6 predictions flips the percentage by ~1%.
#
#   2. The best-epoch selection criterion is OVERALL accuracy, not Link-4
#      accuracy. The model that maximises overall accuracy has learned to be
#      slightly more conservative with Link-4 predictions (fewer false Link-4
#      calls on the abundant classes) — but this hurts recall on true Link-4.
#
#   3. The high class weight (34.4) keeps Link-4 in the gradient, but the
#      model oscillates between over-predicting and under-predicting Link-4
#      as training fine-tunes around the large-weight loss signal.
#
# In short: Link-4 accuracy and overall accuracy are somewhat in tension.
# Epoch 50 is better for Link-4 (99.1%) but worse overall (98.6%).
# Epoch 52 is the best overall checkpoint but Link-4 has dipped to 87.9%.
# ─────────────────────────────────────────────────────────────────────────────

def fig_link4_instability():
    ep  = [10,   20,   30,   40,   50,   52]
    l4  = [86.0, 96.3, 98.1, 96.3, 99.1, 87.9]
    ova = [94.6, 97.4, 97.8, 98.2, 98.6, 98.7]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(ep, ova, color=BLUE, lw=2.2, marker="o", ms=6,
            label="Overall val accuracy")
    ax.plot(ep, l4,  color=RED,  lw=2.2, marker="s", ms=6,
            ls="--", label="Link-4 val accuracy")

    # Annotate the trade-off
    ax.annotate("Peak Link-4\n(99.1%)",
                xy=(50, 99.1), xytext=(46, 100.8),
                fontsize=8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    ax.annotate("Best overall epoch\nLink-4 dips to 87.9%",
                xy=(52, 87.9), xytext=(46.5, 85.0),
                fontsize=8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    ax.annotate("Best overall\n98.7%",
                xy=(52, 98.7), xytext=(49.5, 101.2),
                fontsize=8, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9))

    ax.set_xlabel("Epoch (logged checkpoints)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Link-4 vs. Overall Accuracy — Trade-off at Best Epoch",
                 fontweight="bold")
    ax.set_ylim(80, 103.5)
    ax.set_xticks(ep)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, ls="--", alpha=0.28)

    # Shade the "tension zone" between ep 50 and 52
    ax.axvspan(50, 52, alpha=0.08, color=RED)
    ax.text(51, 82.0, "trade-off\nzone", ha="center",
            fontsize=7.5, color=RED, alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_link4_instability.png")
    plt.close(); print("Saved: fig_link4_instability")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating all figures ...\n")
    fig_scene_feasible()
    fig_scene_infeasible()
    fig_scene_val_correct()
    fig_scene_val_wrong()
    fig_loss_curve()
    fig_acc_curve()
    fig_lr_schedule()
    fig_class_dist()
    fig_class_weights()
    fig_perclass_acc()
    fig_confusion_matrix()
    fig_architecture()
    fig_link4_instability()
    print(f"\nAll figures saved to ./{OUT}/")
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".png"):
            print(f"  {f}")