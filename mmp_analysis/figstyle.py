#!/usr/bin/env python3
"""
figstyle.py -- one visual language for the whole figure set.

Design decisions and why
------------------------
TYPOGRAPHY  Liberation Sans (metric-compatible with Helvetica/Arial, which is
what ACS journals expect) with a single type scale. Titles are sentence case,
semibold rather than bold, and left-aligned to the panel's data area so they
sit on the same optical margin as the axis label.

COLOUR      Okabe-Ito base, chosen for colour-vision deficiency. Each design
axis keeps ONE hue everywhere it appears, so a reader who learns "green =
bridge" in figure A carries it into every later panel. Confounded rules are
deliberately desaturated: they must be visible but must never compete.

DIVERGING   A perceptually even blue-white-red ramp anchored at zero via
TwoSlopeNorm. Zero is near-white so sign is readable at a glance, and the two
arms are balanced in lightness so neither side looks "stronger".

INK         Nothing is pure black (#111827 instead) and nothing is pure white;
a paper-tone canvas reduces glare in print and lets white callout boxes read
as foreground.

SPACING     Panels are placed with explicit gridspec margins rather than
tight_layout, because tight_layout fights suptitles and colorbars and produces
the overlapping titles seen in earlier drafts.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# --------------------------------------------------------------------------
# palette
# --------------------------------------------------------------------------
INK        = "#111827"    # near-black, warmer than #000
INK_SOFT   = "#4B5563"    # secondary text
MUTED      = "#9CA3AF"    # captions, notes
HAIRLINE   = "#E5E7EB"    # grid
PAPER      = "#FFFFFF"
PANEL_TINT = "#F9FAFB"    # very light fill for inactive cells

AXIS_COLOR = {
    "substituent": "#1B6CA8",   # blue
    "ring":        "#D68910",   # amber
    "bridge":      "#0E8A6B",   # teal-green
    "confounded":  "#C4C8CE",   # desaturated grey
}
AXIS_FILL = {                     # 12% tints for node/箱 backgrounds
    "substituent": "#E8F1F8",
    "ring":        "#FBF1E0",
    "bridge":      "#E4F3EF",
    "confounded":  "#F2F3F5",
}
AXIS_LABEL = {"substituent": "substituent", "ring": "ring",
              "bridge": "bridge", "confounded": "confounded"}
AXIS_ORDER = ["substituent", "ring", "bridge", "confounded"]

ACCENT_POS = "#B23A48"    # warm accent for "attention" annotations
ACCENT_NEG = "#1B6CA8"

# balanced diverging ramp, white-ish at zero
DIVERGE = LinearSegmentedColormap.from_list("bwr_bal", [
    "#2C5F8A", "#6699BB", "#AFC9DC", "#F4F5F6",
    "#EAC0AE", "#D08C6E", "#A8503A"])

PROP_TEX = {"Hf": r"$\Delta H_{f}$", "Q": r"$Q$", "OBgood": r"OB$_{\mathrm{good}}$",
            "Pe": r"$P_{e}$", "D": r"$D$", "P": r"$P$",
            "IS": r"$h_{50}$", "nu": r"$\nu$"}

# --------------------------------------------------------------------------
# rc
# --------------------------------------------------------------------------
_FONT_STACK = ["Liberation Sans", "DejaVu Sans", "FreeSans"]


def apply():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 400,
        "figure.facecolor": PAPER, "savefig.facecolor": PAPER,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.10,

        "font.family": "sans-serif", "font.sans-serif": _FONT_STACK,
        "font.size": 8.0, "mathtext.fontset": "dejavusans",

        "axes.titlesize": 8.8, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 7.0,
        "axes.labelsize": 8.0, "axes.labelcolor": INK_SOFT,
        "axes.labelpad": 3.5,
        "axes.edgecolor": "#9AA1AC", "axes.linewidth": 0.6,
        "axes.facecolor": PAPER,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False,

        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,

        "legend.fontsize": 6.8, "legend.frameon": False,
        "legend.handlelength": 1.1, "legend.handletextpad": 0.45,
        "legend.columnspacing": 0.9, "legend.borderpad": 0.2,

        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def ptex(p):
    return PROP_TEX.get(p, p)


def clean(ax, grid=None, spines=("left", "bottom")):
    """Hairline grid behind the data, only the requested spines."""
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in spines)
    if grid:
        ax.grid(axis=grid, color=HAIRLINE, lw=0.55, zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(length=2.6, pad=2.0)
    return ax


def bare(ax):
    """No spines, no ticks -- for heatmaps and diagrams."""
    ax.set_facecolor(PAPER)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return ax


def panel(ax, letter, x=-0.155, y=1.03):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", va="bottom", ha="left", color=INK)


def title(ax, text, sub=None, pad=None):
    """Panel title with an optional lighter kicker line ABOVE it.

    The kicker is drawn first (higher y) and the title padded down, so the two
    never collide -- an earlier draft placed both at y~1.0 and they overlapped.
    """
    if sub:
        ax.set_title(text, color=INK, pad=pad if pad is not None else 15.0)
        ax.text(0, 1.0, sub, transform=ax.transAxes, fontsize=6.6,
                color=MUTED, va="bottom", ha="left")
    else:
        ax.set_title(text, color=INK, pad=pad if pad is not None else 7.0)


def figtitle(fig, text, sub=None, x=0.012, y=0.985):
    fig.text(x, y, text, fontsize=11.5, fontweight="bold", color=INK,
             ha="left", va="top")
    if sub:
        fig.text(x, y - 0.035, sub, fontsize=7.4, color=INK_SOFT,
                 ha="left", va="top")


def footnote(fig, text, x=0.012, y=0.012):
    fig.text(x, y, text, fontsize=6.4, color=MUTED, ha="left", va="bottom",
             linespacing=1.5)


def heat(ax, M, xlabels, ylabels, fmt="{:+.2f}", cmap=DIVERGE, vlim=None,
         fontsize=6.4, gap=0.045):
    """Heatmap with hairline cell separation and auto-contrast labels."""
    import numpy as np
    v = vlim or (np.nanmax(np.abs(M)) or 1.0)
    im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -v, v), aspect="auto",
                   interpolation="nearest")
    ny, nx = M.shape
    for i in range(ny + 1):
        ax.axhline(i - .5, color=PAPER, lw=1.4, zorder=3)
    for j in range(nx + 1):
        ax.axvline(j - .5, color=PAPER, lw=1.4, zorder=3)
    for i in range(ny):
        for j in range(nx):
            if np.isfinite(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                        fontsize=fontsize, zorder=4,
                        color=PAPER if abs(M[i, j]) > .58 * v else INK)
    ax.set_xticks(range(nx)); ax.set_xticklabels(xlabels)
    ax.set_yticks(range(ny)); ax.set_yticklabels(ylabels)
    bare(ax)
    return im, v


def cbar(fig, im, ax, label, fraction=0.030, pad=0.016):
    cb = fig.colorbar(im, ax=ax, fraction=fraction, pad=pad)
    cb.set_label(label, fontsize=6.8, color=INK_SOFT)
    cb.ax.tick_params(labelsize=6.2, length=2, color=INK_SOFT)
    cb.outline.set_visible(False)
    return cb


def badge(ax, x, y, text, color, fontsize=7.4, transform=None):
    """Value chip on a white ground -- used on the bridge cycle edges."""
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="semibold", color=INK, zorder=6,
            transform=transform or ax.transData,
            bbox=dict(fc=PAPER, ec=color, lw=0.7,
                      boxstyle="round,pad=0.28", alpha=0.97))
