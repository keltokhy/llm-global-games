"""
Generate explanatory diagrams for the LLM Global Games paper.

Produces:
  1. Game structure diagram (Morris-Shin regime change game)
  2. Signal-to-text pipeline (θ → briefing → LLM → decision)
  3. Experimental design overview (all treatments)
  4. Authoritarian information control (how instruments attack the channel)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
from agent_based_simulation.runtime import apply_serif_paper_style
apply_serif_paper_style()
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Shared palette ───────────────────────────────────────────────────
C = {
    "bg": "#FAFAFA",
    "box": "#2C3E50",
    "box_fill": "#EBF5FB",
    "accent": "#E74C3C",
    "accent2": "#27AE60",
    "accent3": "#F39C12",
    "accent4": "#8E44AD",
    "grey": "#95A5A6",
    "light_grey": "#ECF0F1",
    "dark": "#2C3E50",
    "blue": "#2980B9",
    "signal": "#3498DB",
    "text_dark": "#1A1A2E",
    "arrow": "#34495E",
}

def _box(ax, xy, w, h, text, fc=None, ec=None, fontsize=10, bold=False,
         text_color=None, alpha=1.0, zorder=3, style="round,pad=0.02",
         ha="center", va="center", lw=1.5):
    """Draw a rounded rectangle with centered text."""
    fc = fc or C["box_fill"]
    ec = ec or C["box"]
    tc = text_color or C["text_dark"]
    box = FancyBboxPatch(
        xy, w, h, boxstyle=style,
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, text, ha=ha, va=va, fontsize=fontsize,
            fontweight=weight, color=tc, zorder=zorder + 1)
    return box


def _arrow(ax, xy_from, xy_to, color=None, lw=1.8, style="-|>",
           connectionstyle="arc3,rad=0", zorder=2, shrinkA=0, shrinkB=0):
    color = color or C["arrow"]
    a = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle=style, color=color, lw=lw,
        connectionstyle=connectionstyle, zorder=zorder,
        shrinkA=shrinkA, shrinkB=shrinkB,
        mutation_scale=15,
    )
    ax.add_patch(a)
    return a


# ======================================================================
# DIAGRAM 1: The Global Game of Regime Change
# ======================================================================
def diagram_game_structure():
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-1, 8)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title
    ax.text(5.25, 7.5, "The Global Game of Regime Change",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=C["dark"])
    ax.text(5.25, 7.0, "Morris & Shin (2003)",
            ha="center", va="center", fontsize=11, color=C["grey"],
            style="italic")

    # ── Nature box ──
    _box(ax, (3.5, 5.7), 3.5, 0.85, "Nature draws  $\\theta \\sim \\mathrm{Uniform}$",
         fc="#FADBD8", ec=C["accent"], fontsize=11, bold=True)

    # ── Signal generation ──
    _arrow(ax, (5.25, 5.7), (5.25, 5.15), color=C["accent"])
    _box(ax, (2.5, 4.2), 5.5, 0.85,
         "Private signals:  $x_i = \\theta + \\varepsilon_i$,    $\\varepsilon_i \\sim \\mathcal{N}(0, \\sigma^2)$",
         fc="#D5F5E3", ec=C["accent2"], fontsize=10.5, bold=False)

    # ── Citizens ──
    _arrow(ax, (5.25, 4.2), (5.25, 3.6), color=C["accent2"])

    # Draw N citizens as small boxes
    n_show = 5
    cw, ch = 1.3, 0.65
    gap = 0.45
    total_w = n_show * cw + (n_show - 1) * gap
    x_start = 5.25 - total_w / 2
    citizen_labels = ["$i=1$", "$i=2$", "$i=3$", "$\\cdots$", "$i=N$"]
    for k in range(n_show):
        cx = x_start + k * (cw + gap)
        fc_k = "#EBF5FB" if k != 3 else "white"
        ec_k = C["blue"] if k != 3 else "white"
        lw_k = 1.5 if k != 3 else 0
        _box(ax, (cx, 2.85), cw, ch, citizen_labels[k],
             fc=fc_k, ec=ec_k, fontsize=9, lw=lw_k)

    ax.text(5.25, 3.55, "Each citizen observes only their own $x_i$ and chooses:",
            ha="center", va="center", fontsize=9.5, color=C["grey"])

    # ── Decision ──
    # JOIN arrow
    ax.annotate("", xy=(2.5, 1.8), xytext=(4.0, 2.85),
                arrowprops=dict(arrowstyle="-|>", color=C["accent"],
                                lw=1.5, connectionstyle="arc3,rad=0.15"))
    _box(ax, (1.3, 1.15), 2.2, 0.6, "JOIN  ($a_i = 1$)",
         fc="#FADBD8", ec=C["accent"], fontsize=10, bold=True)

    # STAY arrow
    ax.annotate("", xy=(8.0, 1.8), xytext=(6.5, 2.85),
                arrowprops=dict(arrowstyle="-|>", color=C["accent2"],
                                lw=1.5, connectionstyle="arc3,rad=-0.15"))
    _box(ax, (7.0, 1.15), 2.2, 0.6, "STAY  ($a_i = 0$)",
         fc="#D5F5E3", ec=C["accent2"], fontsize=10, bold=True)

    # ── Regime outcome ──
    _arrow(ax, (5.25, 1.15), (5.25, 0.55), color=C["dark"])
    _box(ax, (2.0, -0.5), 6.5, 0.95,
         "Regime falls iff  $A = \\int a_i\\,di > \\theta$",
         fc="#F9E79F", ec=C["accent3"], fontsize=11.5, bold=True)

    # ── Payoff annotations ──
    ax.text(0.0, 0.35,
            "Payoffs:\n"
            "  JOIN + success: $+B$\n"
            "  JOIN + failure:  $-C$\n"
            "  STAY:  $\\;\\;0$",
            ha="left", va="top", fontsize=9, color=C["dark"],
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc=C["light_grey"],
                      ec=C["grey"], alpha=0.7))

    # ── Equilibrium annotation ──
    ax.text(9.0, 0.35,
            "Equilibrium:\n"
            "  JOIN iff $x_i < x^*$\n"
            "  $x^* = \\theta^* + \\sigma\\Phi^{-1}(\\theta^*)$\n"
            "  $\\theta^* = B/(B+C)$",
            ha="left", va="top", fontsize=9, color=C["dark"],
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#EBF5FB",
                      ec=C["blue"], alpha=0.7))

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "diagram_game_structure.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "diagram_game_structure.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  -> diagram_game_structure")


# ======================================================================
# DIAGRAM 2: Signal-to-Text-to-Decision Pipeline
# ======================================================================
def diagram_pipeline():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(-0.5, 13)
    ax.set_ylim(-1.5, 4.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(6.25, 4.1, "Signal $\\rightarrow$ Text $\\rightarrow$ Decision Pipeline",
            ha="center", va="center", fontsize=15, fontweight="bold", color=C["dark"])

    # Pipeline boxes
    stages = [
        (0.0,  1.8, 1.6, 1.0, "$\\theta$\nRegime\nstrength", "#FADBD8", C["accent"]),
        (2.2,  1.8, 2.0, 1.0, "$x_i = \\theta + \\varepsilon_i$\nPrivate\nsignal", "#D5F5E3", C["accent2"]),
        (4.8,  1.8, 1.6, 1.0, "$z_i$\nz-score\nnormalized", "#D6EAF8", C["blue"]),
        (7.0,  1.55, 2.3, 1.5, "Briefing\nGenerator\n(8 domains,\n3 sliders)", "#F9E79F", C["accent3"]),
        (9.9,  1.8, 1.6, 1.0, "LLM\nAgent\n$i$", "#E8DAEF", C["accent4"]),
        (12.0, 1.8, 0.9, 1.0, "JOIN\nor\nSTAY", "#FADBD8", C["accent"]),
    ]

    for x, y, w, h, txt, fc, ec in stages:
        _box(ax, (x, y), w, h, txt, fc=fc, ec=ec, fontsize=9, bold=True)

    # Arrows between boxes
    arrow_pairs = [
        ((1.6, 2.3), (2.2, 2.3)),
        ((4.2, 2.3), (4.8, 2.3)),
        ((6.4, 2.3), (7.0, 2.3)),
        ((9.3, 2.3), (9.9, 2.3)),
        ((11.5, 2.3), (12.0, 2.3)),
    ]
    for fr, to in arrow_pairs:
        _arrow(ax, fr, to, color=C["arrow"], lw=2.0)

    # Labels on arrows
    arrow_labels = [
        (1.9, 2.65, "$+\\varepsilon_i$", 8),
        (4.5, 2.65, "standardize", 7.5),
        (6.55, 2.65, "map to text", 7.5),
        (9.55, 2.65, "natural\nlanguage", 7),
        (11.7, 2.65, "binary\nchoice", 7),
    ]
    for x, y, txt, fs in arrow_labels:
        ax.text(x, y, txt, ha="center", va="bottom", fontsize=fs,
                color=C["grey"], style="italic")

    # Briefing detail annotation below
    detail_text = (
        "Briefing Generator internals:\n"
        "  z-score  --[logistic]-->  Direction slider (weak <-> strong)\n"
        "  z-score  --[Gaussian]-->  Clarity slider (ambiguous <-> clear)\n"
        "  z-score  --[logistic]-->  Coordination slider (quiet <-> open)\n\n"
        "  -> 8 evidence domains x 4 phrase ladders\n"
        "  -> Many small word choices (\"dithering\") recover continuity"
    )
    ax.text(6.25, -0.15, detail_text,
            ha="center", va="top", fontsize=8.5, color=C["dark"],
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc=C["light_grey"],
                      ec=C["grey"], alpha=0.6))

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "diagram_pipeline.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "diagram_pipeline.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  -> diagram_pipeline")


# ======================================================================
# DIAGRAM 3: Experimental Design Overview
# ======================================================================
def diagram_experimental_design():
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-1, 10)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(6.0, 9.5, "Experimental Design",
            ha="center", va="center", fontsize=16, fontweight="bold", color=C["dark"])

    # ── Part I header ──
    _box(ax, (0.0, 8.2), 5.2, 0.7, "Part I: Do LLMs Play the Game?",
         fc=C["blue"], ec=C["dark"], fontsize=12, bold=True, text_color="white")

    # Part I treatments
    part1 = [
        ("Pure Global Game", "Private signals only\nSimultaneous decisions",
         "#D5F5E3", C["accent2"]),
        ("Communication", "Pre-play messaging\n(Watts-Strogatz network)",
         "#D6EAF8", C["blue"]),
        ("Scramble", "Briefings randomly\nredistributed across periods",
         "#F9E79F", C["accent3"]),
        ("Flip", "Z-score negated\nbefore briefing",
         "#FADBD8", C["accent"]),
    ]
    for i, (title, desc, fc, ec) in enumerate(part1):
        y = 7.3 - i * 1.15
        _box(ax, (0.3, y), 2.0, 0.85, title,
             fc=fc, ec=ec, fontsize=9.5, bold=True)
        ax.text(2.55, y + 0.42, desc, ha="left", va="center",
                fontsize=8, color=C["dark"])

    # Bracket for falsification
    ax.annotate("", xy=(5.1, 5.72), xytext=(5.1, 4.57),
                arrowprops=dict(arrowstyle="-", color=C["grey"], lw=1.2))
    ax.text(5.3, 5.15, "Falsification\ntests", ha="left", va="center",
            fontsize=8, color=C["grey"], style="italic")

    # ── Part II header ──
    _box(ax, (6.5, 8.2), 5.2, 0.7, "Part II: Information Design",
         fc=C["accent4"], ec=C["dark"], fontsize=12, bold=True, text_color="white")

    # Part II treatments
    part2 = [
        ("Stability Design", "4$\\times$ clarity width\nFlatter direction slope",
         "#E8DAEF", C["accent4"]),
        ("Instability Design", "0.15$\\times$ clarity width\nSteeper direction slope",
         "#E8DAEF", C["accent4"]),
        ("Public Signal", "Shared news bulletin\nfrom $\\theta$",
         "#D6EAF8", C["blue"]),
        ("Censorship", "Upper / lower\nbinary signal pooling",
         "#F9E79F", C["accent3"]),
        ("Surveillance", "\"Communications\nmonitored\" warning",
         "#FADBD8", C["accent"]),
        ("Propaganda", "$k$ regime plants\nsend pro-regime msgs",
         "#FADBD8", C["accent"]),
    ]
    for i, (title, desc, fc, ec) in enumerate(part2):
        y = 7.3 - i * 0.92
        _box(ax, (6.8, y), 2.2, 0.7, title,
             fc=fc, ec=ec, fontsize=9, bold=True)
        ax.text(9.25, y + 0.35, desc, ha="left", va="center",
                fontsize=7.5, color=C["dark"])

    # ── Bottom: shared infrastructure ──
    _box(ax, (1.0, -0.5), 10.0, 0.85,
         "Shared: 9 models  |  25 agents/period  |  $\\sigma = 0.3$  |  "
         "temp $= 0.7$  |  narrative briefings, no payoff tables",
         fc=C["light_grey"], ec=C["grey"], fontsize=9.5, bold=False)

    # Key results annotations
    results_text = (
        "Key findings:\n"
        "$\\bullet$ Pure: mean $r = +0.73$\n"
        "$\\bullet$ Scramble: $r \\to +0.23$\n"
        "$\\bullet$ Flip: $r \\to -0.67$\n"
        "$\\bullet$ Comm: $+3.7$ pp"
    )
    ax.text(0.3, 2.9, results_text, ha="left", va="top",
            fontsize=8, color=C["dark"],
            bbox=dict(boxstyle="round,pad=0.3", fc="#D5F5E3",
                      ec=C["accent2"], alpha=0.6))

    results_text2 = (
        "Key findings:\n"
        "$\\bullet$ Stability: $+19.5$ pp\n"
        "$\\bullet$ Public signal: $-10.7$ pp\n"
        "$\\bullet$ Censorship: $+18.5$ pp\n"
        "$\\bullet$ Surveillance: $-17.5$ pp\n"
        "$\\bullet$ Surv $+$ censor: 30.9% $\\to$ 3.7%"
    )
    ax.text(6.8, 1.5, results_text2, ha="left", va="top",
            fontsize=8, color=C["dark"],
            bbox=dict(boxstyle="round,pad=0.3", fc="#E8DAEF",
                      ec=C["accent4"], alpha=0.6))

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "diagram_experimental_design.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "diagram_experimental_design.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  -> diagram_experimental_design")


# ======================================================================
# DIAGRAM 4: Authoritarian Information Control
# ======================================================================
def diagram_authoritarian_control():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-1, 9)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(6.0, 8.6, "The Information Channel as a Vulnerability",
            ha="center", va="center", fontsize=15, fontweight="bold", color=C["dark"])
    ax.text(6.0, 8.1,
            "The same channel that enables coordination is exploitable by the regime",
            ha="center", va="center", fontsize=10, color=C["grey"], style="italic")

    # ── Central channel: Private Info → Communication → Coordination ──
    channel_boxes = [
        (0.5,  5.5, 2.4, 1.0, "Private\nInformation\n($x_i = \\theta + \\varepsilon_i$)",
         "#D5F5E3", C["accent2"]),
        (4.3,  5.5, 2.4, 1.0, "Communication\nChannel\n(network messages)",
         "#D6EAF8", C["blue"]),
        (8.2,  5.5, 2.8, 1.0, "Coordination\nOutcome\n($A > \\theta$ ?)",
         "#F9E79F", C["accent3"]),
    ]
    for x, y, w, h, txt, fc, ec in channel_boxes:
        _box(ax, (x, y), w, h, txt, fc=fc, ec=ec, fontsize=9.5, bold=True)

    # Arrows along channel
    _arrow(ax, (2.9, 6.0), (4.3, 6.0), color=C["accent2"], lw=2.5)
    _arrow(ax, (6.7, 6.0), (8.2, 6.0), color=C["blue"], lw=2.5)

    # ── Regime instruments attacking from below ──
    # Censorship attacks private info
    _box(ax, (0.0, 2.5), 2.8, 1.1,
         "CENSORSHIP\nPools signals $\\to$ removes\nprivate information",
         fc="#FADBD8", ec=C["accent"], fontsize=8.5, bold=True)
    ax.annotate("", xy=(1.4, 5.5), xytext=(1.4, 3.6),
                arrowprops=dict(arrowstyle="-|>", color=C["accent"],
                                lw=2.2, connectionstyle="arc3,rad=0"))
    ax.text(1.7, 4.55, "BLOCKS", ha="left", va="center",
            fontsize=9, fontweight="bold", color=C["accent"], rotation=90)

    # Surveillance attacks communication
    _box(ax, (3.7, 2.5), 3.0, 1.1,
         "SURVEILLANCE\nPoisons channel via\npreference falsification",
         fc="#FADBD8", ec=C["accent"], fontsize=8.5, bold=True)
    ax.annotate("", xy=(5.2, 5.5), xytext=(5.2, 3.6),
                arrowprops=dict(arrowstyle="-|>", color=C["accent"],
                                lw=2.2, connectionstyle="arc3,rad=0"))
    ax.text(5.5, 4.55, "POISONS", ha="left", va="center",
            fontsize=9, fontweight="bold", color=C["accent"], rotation=90)

    # Propaganda attacks communication
    _box(ax, (7.8, 2.5), 3.5, 1.1,
         "PROPAGANDA\nDilutes channel with\npro-regime messages ($k$ plants)",
         fc="#FADBD8", ec=C["accent"], fontsize=8.5, bold=True)
    ax.annotate("", xy=(9.0, 5.5), xytext=(9.0, 3.6),
                arrowprops=dict(arrowstyle="-|>", color=C["accent"],
                                lw=2.2, connectionstyle="arc3,rad=0"))
    ax.text(9.3, 4.55, "DILUTES", ha="left", va="center",
            fontsize=9, fontweight="bold", color=C["accent"], rotation=90)

    # ── Interaction box at bottom ──
    _box(ax, (1.5, 0.2), 9.0, 1.5, "", fc="white", ec=C["dark"], lw=2.0)
    ax.text(6.0, 1.45, "Instrument Interactions", ha="center", va="center",
            fontsize=11, fontweight="bold", color=C["dark"])

    interactions = [
        ("Surv + Propaganda:", "Sub-additive (both attack same channel)", 1.05),
        ("Surv + Censorship:", "Super-additive (attack different channels)", 0.65),
        ("All three:", "30.9% $\\to$ 3.7%  (coordination collapse)", 0.25),
    ]
    for label, desc, y in interactions:
        ax.text(2.0, y, label, ha="left", va="center",
                fontsize=9, fontweight="bold", color=C["dark"])
        ax.text(5.0, y, desc, ha="left", va="center",
                fontsize=9, color=C["dark"])

    # ── Key insight callout ──
    ax.text(6.0, -0.7,
            '"The regime does not need to change what citizens believe; '
            'it needs only to make them uncertain about each other."',
            ha="center", va="center", fontsize=9.5, style="italic",
            color=C["accent"],
            bbox=dict(boxstyle="round,pad=0.3", fc="#FEF9E7",
                      ec=C["accent3"], alpha=0.8))

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "diagram_authoritarian_control.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "diagram_authoritarian_control.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  -> diagram_authoritarian_control")


# ======================================================================
# DIAGRAM 5: Communication Treatment Detail
# ======================================================================
def diagram_communication():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-0.5, 7.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(5.25, 7.2, "Communication Treatment: Two-Phase Protocol",
            ha="center", va="center", fontsize=14, fontweight="bold", color=C["dark"])

    # ── Phase 1 ──
    _box(ax, (0.0, 5.7), 4.8, 0.9, "Phase 1: Message Round",
         fc=C["blue"], ec=C["dark"], fontsize=12, bold=True, text_color="white")

    # Network diagram (simplified)
    # 5 nodes in a ring + some shortcuts
    center_x, center_y = 2.4, 4.0
    r = 1.2
    n_nodes = 5
    angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, n_nodes, endpoint=False)
    positions = [(center_x + r * np.cos(a), center_y + r * np.sin(a)) for a in angles]

    # Draw edges (ring + shortcut)
    for i in range(n_nodes):
        j = (i + 1) % n_nodes
        ax.plot([positions[i][0], positions[j][0]],
                [positions[i][1], positions[j][1]],
                color=C["grey"], lw=1.0, alpha=0.5, zorder=1)
    # Shortcut
    ax.plot([positions[0][0], positions[2][0]],
            [positions[0][1], positions[2][1]],
            color=C["grey"], lw=1.0, alpha=0.5, ls="--", zorder=1)

    # Draw nodes
    for i, (px, py) in enumerate(positions):
        circle = plt.Circle((px, py), 0.22, fc="#D6EAF8", ec=C["blue"],
                             lw=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(px, py, f"$i_{i+1}$", ha="center", va="center",
                fontsize=8, zorder=4)

    ax.text(2.4, 2.4,
            "Watts-Strogatz network\n$k=4$, $p=0.3$\n\n"
            "Each agent sends one\nmessage to neighbors",
            ha="center", va="top", fontsize=8.5, color=C["dark"],
            bbox=dict(boxstyle="round,pad=0.2", fc=C["light_grey"],
                      ec=C["grey"], alpha=0.5))

    # ── Phase 2 ──
    _box(ax, (5.8, 5.7), 4.8, 0.9, "Phase 2: Decision Round",
         fc=C["accent2"], ec=C["dark"], fontsize=12, bold=True, text_color="white")

    # Decision process
    _box(ax, (6.2, 4.2), 4.0, 0.65, "Agent observes:",
         fc="#D5F5E3", ec=C["accent2"], fontsize=10, bold=True)

    items = [
        "1. Own private briefing",
        "2. Messages from neighbors",
    ]
    for i, txt in enumerate(items):
        ax.text(6.5, 3.7 - i * 0.45, txt, ha="left", va="center",
                fontsize=9.5, color=C["dark"])

    _arrow(ax, (8.2, 2.7), (8.2, 2.2), color=C["accent2"], lw=2)

    _box(ax, (6.8, 1.4), 2.8, 0.7, "JOIN  or  STAY",
         fc="#F9E79F", ec=C["accent3"], fontsize=11, bold=True)

    # Arrow from phase 1 to phase 2
    ax.annotate("", xy=(5.8, 5.3), xytext=(4.8, 5.3),
                arrowprops=dict(arrowstyle="-|>", color=C["dark"],
                                lw=2.0))
    ax.text(5.3, 5.55, "then", ha="center", va="center",
            fontsize=9, color=C["grey"])

    # Key finding
    ax.text(5.25, 0.2,
            "Finding: Communication raises beliefs ($+2.4$ pp) but not actions ($-0.9$ pp, $p = 0.34$).\n"
            "The channel transmits strategic uncertainty about others' willingness to act.",
            ha="center", va="center", fontsize=9, color=C["dark"],
            bbox=dict(boxstyle="round,pad=0.35", fc="#FEF9E7",
                      ec=C["accent3"], alpha=0.8))

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "diagram_communication.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "diagram_communication.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  -> diagram_communication")


# ======================================================================
if __name__ == "__main__":
    print("Generating diagrams...")
    diagram_game_structure()
    diagram_pipeline()
    diagram_experimental_design()
    diagram_authoritarian_control()
    diagram_communication()
    print("Done.")
