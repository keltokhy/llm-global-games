"""
Generate explanatory diagrams for the LLM Global Games paper.

Produces:
  1. Signal-to-text pipeline (θ → briefing → LLM → decision)
  2. Authoritarian information control (how instruments attack the channel)
"""

import json
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

STATS = json.loads((Path(__file__).resolve().parent / "verified_stats.json").read_text())

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
# DIAGRAM 1: Signal-to-Text-to-Decision Pipeline
# ======================================================================
def diagram_pipeline():
    """Publication-quality pipeline diagram: θ → signal → z → briefing → LLM → decision."""
    _border  = "#4A4A4A"
    _fill    = "#F0F0F0"
    _fill2   = "#E4E4E4"
    _accent  = "#2C5F8A"
    _arrow_c = "#555555"
    _text    = "#1A1A1A"
    _muted   = "#666666"

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.set_xlim(-0.1, 7.1)
    ax.set_ylim(-0.2, 3.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    bh = 0.95   # box height

    # ── Row 1 (top): θ → x_i → z_i ──
    y1 = 1.95
    # θ
    _box(ax, (0.0, y1), 1.3, bh, "", fc=_fill, ec=_border, lw=1.5)
    ax.text(0.65, y1 + bh/2 + 0.15, r"$\theta$", ha="center", va="center",
            fontsize=14, fontweight="bold", color=_text)
    ax.text(0.65, y1 + bh/2 - 0.18, "Regime strength", ha="center", va="center",
            fontsize=8, color=_muted)
    # x_i
    _box(ax, (2.2, y1), 2.2, bh, "", fc=_fill, ec=_border, lw=1.5)
    ax.text(3.3, y1 + bh/2 + 0.15, r"$x_i = \theta + \varepsilon_i$",
            ha="center", va="center", fontsize=12, fontweight="bold", color=_text)
    ax.text(3.3, y1 + bh/2 - 0.18, "Private signal", ha="center", va="center",
            fontsize=8, color=_muted)
    # z_i
    _box(ax, (5.5, y1), 1.5, bh, "", fc=_fill, ec=_border, lw=1.5)
    ax.text(6.25, y1 + bh/2 + 0.15, r"$z_i$", ha="center", va="center",
            fontsize=14, fontweight="bold", color=_text)
    ax.text(6.25, y1 + bh/2 - 0.18, "Standardized", ha="center", va="center",
            fontsize=8, color=_muted)

    # Row 1 arrows
    _arrow(ax, (1.3, y1 + bh/2), (2.2, y1 + bh/2), color=_arrow_c, lw=2.0)
    ax.text(1.75, y1 + bh/2 + 0.15, r"$+\varepsilon_i$", ha="center", va="bottom",
            fontsize=9, color=_muted, style="italic")
    _arrow(ax, (4.4, y1 + bh/2), (5.5, y1 + bh/2), color=_arrow_c, lw=2.0)
    ax.text(4.95, y1 + bh/2 + 0.15, "normalize", ha="center", va="bottom",
            fontsize=8, color=_muted, style="italic")

    # ── Row 2 (bottom): Briefing Gen → LLM → Decision ──
    y2 = 0.1
    # Briefing Generator
    _box(ax, (0.0, y2), 2.8, bh + 0.15, "", fc=_fill2, ec=_accent, lw=1.5)
    ax.text(1.4, y2 + (bh + 0.15)/2 + 0.15, "Briefing Generator",
            ha="center", va="center", fontsize=11, fontweight="bold", color=_accent)
    ax.text(1.4, y2 + (bh + 0.15)/2 - 0.2,
            "3 sliders $\\times$ 8 domains $\\times$ 4 phrase ladders",
            ha="center", va="center", fontsize=7.5, color=_text)
    # LLM
    _box(ax, (3.7, y2 + 0.08), 1.5, bh, "", fc=_fill, ec=_border, lw=1.5)
    ax.text(4.45, y2 + 0.08 + bh/2 + 0.15, "LLM", ha="center", va="center",
            fontsize=13, fontweight="bold", color=_text)
    ax.text(4.45, y2 + 0.08 + bh/2 - 0.18, "Agent $i$", ha="center", va="center",
            fontsize=8, color=_muted)
    # Decision
    _box(ax, (5.9, y2 + 0.08), 1.1, bh, "", fc=_fill, ec=_border, lw=1.5)
    ax.text(6.45, y2 + 0.08 + bh/2, "JOIN\nor\nSTAY", ha="center", va="center",
            fontsize=11, fontweight="bold", color=_text, linespacing=0.85)

    # Row 2 arrows
    _arrow(ax, (2.8, y2 + (bh + 0.15)/2), (3.7, y2 + 0.08 + bh/2),
           color=_arrow_c, lw=2.0)
    ax.text(3.25, y2 + (bh + 0.15)/2 + 0.15, "narrative", ha="center",
            va="bottom", fontsize=8, color=_muted, style="italic")
    _arrow(ax, (5.2, y2 + 0.08 + bh/2), (5.9, y2 + 0.08 + bh/2),
           color=_arrow_c, lw=2.0)
    ax.text(5.55, y2 + 0.08 + bh/2 + 0.15, "binary", ha="center",
            va="bottom", fontsize=8, color=_muted, style="italic")

    # ── Elbow connector: z_i → Briefing Generator ──
    # Go straight down from z_i, then left along the gap, then into BG top
    z_bot = y1                              # bottom of z_i box
    bg_top = y2 + bh + 0.15                 # top of BG box
    mid_y = (z_bot + bg_top) / 2            # midpoint in the gap
    z_cx = 6.25                             # z_i center x
    bg_cx = 1.4                             # BG center x

    # Vertical segment down from z_i
    ax.plot([z_cx, z_cx], [z_bot, mid_y], color=_arrow_c, lw=2.0, solid_capstyle="butt")
    # Horizontal segment left
    ax.plot([z_cx, bg_cx], [mid_y, mid_y], color=_arrow_c, lw=2.0, solid_capstyle="butt")
    # Arrow down into BG
    _arrow(ax, (bg_cx, mid_y), (bg_cx, bg_top + 0.02), color=_arrow_c, lw=2.0)
    # Label on horizontal segment
    ax.text((z_cx + bg_cx) / 2, mid_y + 0.1, "map to text",
            ha="center", va="bottom", fontsize=9, color=_muted, style="italic",
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none"))

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "diagram_pipeline.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "diagram_pipeline.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  -> diagram_pipeline")


# ======================================================================
# DIAGRAM 2: Authoritarian Information Control
# ======================================================================
def diagram_authoritarian_control():
    """Publication-quality diagram: how regime instruments attack the coordination chain."""
    _bg      = "white"
    _border  = "#4A4A4A"
    _fill    = "#F0F0F0"
    _red     = "#B03030"
    _red_bg  = "#F5E0E0"
    _arrow_c = "#555555"
    _text    = "#1A1A1A"
    _muted   = "#666666"

    fig, ax = plt.subplots(figsize=(7, 7.5))
    ax.set_xlim(-0.2, 7.2)
    ax.set_ylim(-2.0, 8.0)
    ax.axis("off")
    fig.patch.set_facecolor(_bg)

    # ── Title ──
    ax.text(3.5, 7.7, "The Information Channel\nas a Vulnerability",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=_text, linespacing=1.3)

    # ── Coordination chain (top) ──
    chain = [
        (0.0, 5.6, 2.0, 1.2, "Private\nInformation",
         "$x_i = \\theta + \\varepsilon_i$"),
        (2.7, 5.6, 2.0, 1.2, "Communication\nChannel",
         "network messages"),
        (5.2, 5.6, 1.8, 1.2, "Coordination\nOutcome",
         "$A > \\theta$ ?"),
    ]
    for x, y, w, h, main, sub in chain:
        _box(ax, (x, y), w, h, "", fc=_fill, ec=_border, lw=1.5,
             text_color=_text)
        cx, cy = x + w/2, y + h/2
        ax.text(cx, cy + 0.2, main, ha="center", va="center",
                fontsize=11, fontweight="bold", color=_text)
        ax.text(cx, cy - 0.28, sub, ha="center", va="center",
                fontsize=9, color=_muted)

    # Chain arrows
    _arrow(ax, (2.0, 6.2), (2.7, 6.2), color=_arrow_c, lw=2.2)
    _arrow(ax, (4.7, 6.2), (5.2, 6.2), color=_arrow_c, lw=2.2)

    ax.text(2.35, 6.5, "informs", ha="center", va="bottom",
            fontsize=9, color=_muted, style="italic")
    ax.text(4.95, 6.5, "aggregates", ha="center", va="bottom",
            fontsize=9, color=_muted, style="italic")

    # ── Regime instruments (below chain) ──
    instruments = [
        (0.0,  3.0, 2.0, 1.6,
         "Censorship",
         "Pools signals\nnear $\\theta^*$;\nremoves private\ninformation",
         "degrades"),
        (2.7,  3.0, 2.0, 1.6,
         "Surveillance",
         "Preference\nfalsification;\nself-censored\nmessages",
         "poisons"),
        (5.2,  3.0, 1.8, 1.6,
         "Propaganda",
         "$k$ regime plants\nsend pro-regime\nmessages;\ndilutes signal",
         "dilutes"),
    ]

    for x, y, w, h, title, desc, verb in instruments:
        _box(ax, (x, y), w, h, "", fc=_red_bg, ec=_red, lw=1.5,
             text_color=_text)
        ax.text(x + w/2, y + h - 0.25, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color=_red)
        ax.text(x + w/2, y + 0.55, desc, ha="center", va="center",
                fontsize=9, color=_text)
        # Attack arrow (upward)
        ax.annotate("", xy=(x + w/2, 5.6), xytext=(x + w/2, 4.6),
                    arrowprops=dict(arrowstyle="-|>", color=_red,
                                    lw=2.2, connectionstyle="arc3,rad=0"))
        ax.text(x + w/2 + 0.2, 5.1, verb, ha="left", va="center",
                fontsize=10, fontweight="bold", color=_red, rotation=90)

    # ── Interaction summary (bottom) ──
    ax.plot([0.0, 7.0], [2.5, 2.5], color="#CCCCCC", lw=0.5)

    sxc_mistral = STATS["regime_control"]["surveillance_x_censorship"]["Mistral Small Creative"]
    idc = STATS["infodesign_comm"]
    surv_delta_base = (sxc_mistral["baseline"] - idc["baseline"]["mean_join"]) * 100
    surv_delta_censor = (sxc_mistral["censor_upper"] - idc["censor_upper"]["mean_join"]) * 100

    ax.text(0.0, 2.0, "Instrument Interactions",
            ha="left", va="center", fontsize=12, fontweight="bold", color=_text)

    ax.text(0.1, 1.3, "Surv. + Propaganda:",
            ha="left", va="center", fontsize=10, fontweight="bold", color=_text)
    ax.text(0.1, 0.85, "Approximately additive",
            ha="left", va="center", fontsize=10, color=_text)

    ax.text(0.1, 0.2, "Surv. + Censorship:",
            ha="left", va="center", fontsize=10, fontweight="bold", color=_text)
    ax.text(0.1, -0.25,
            f"Super-additive: ${surv_delta_censor:.1f}$ pp "
            f"under censorship vs ${surv_delta_base:.1f}$ pp at baseline",
            ha="left", va="center", fontsize=10, color=_text)

    # ── Key insight ──
    ax.text(3.5, -1.3,
            "The regime does not need to change\n"
            "what citizens believe; it needs only to make\n"
            "them uncertain about each other.",
            ha="center", va="center", fontsize=10.5, style="italic",
            color=_red, linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.4", fc="#FAFAFA",
                      ec="#CCCCCC", lw=0.8))

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "diagram_authoritarian_control.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "diagram_authoritarian_control.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  -> diagram_authoritarian_control")


# ======================================================================
if __name__ == "__main__":
    print("Generating diagrams...")
    diagram_pipeline()
    diagram_authoritarian_control()
    print("Done.")
