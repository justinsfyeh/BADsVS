#!/usr/bin/env python3
"""
14_layer_figures.py -- publication figure set for the layered MMPA section.

Design rationale (why these panels and not others)
--------------------------------------------------
The manuscript already reports the screen and the top-10 table. What the MMPA
section has to add is WHY the screen converged where it did and WHAT IS LEFT.
Every panel below is chosen to answer a question a reader actually has after
reading the screening section; panels that only display the pipeline were cut.

MAIN
  figA  Three design layers with distinct roles
        a  effect magnitude vs fragment/core ratio, by layer  (+ size audit)
        b  layer x property mean directed z, significance boxed
        c  per-layer top rules, per-property z lollipops
        d  one exact A->B exemplar per layer
  figB  Fragment roles and residual design space
        a  z_Hf vs z_OBgood trade-off plane by chemotype   <- the core tension
        b  chemotype x property atlas
        c  candidate maturity (already / actionable) by layer

SUPPORTING
  S1  correlation divergence per layer, with attenuation caveat
  S2  method audit: frag_ratio, FDR effect, mirror dedup
  S3  composite sensitivity (h50 in / out / nu)
  S4  deduplicated volcano: effect vs -log10 p
  S5  regression-to-the-mean control

Every loader is fault tolerant: missing files or columns skip the panel with a
printed note rather than crashing, so the script is usable while the pipeline
is still being re-run.

Usage:
  <tartarus python> 14_layer_figures.py
  <tartarus python> 14_layer_figures.py --only figA figB
  <tartarus python> 14_layer_figures.py --format pdf
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch

import config
import layers as L
import mmpa_lib as ml

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 400,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 8, "axes.labelcolor": "#1a1a1a",
    "axes.edgecolor": "#3a3a3a", "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.fontsize": 7, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

INK = "#1a1a1a"
MUTED = "#8a8a8a"
GRID = "#e6e6e6"

# Okabe-Ito, colour-vision-deficiency safe
LAYER_COLOR = {"substituent": "#0072B2", "scaffold": "#E69F00",
               "linker": "#009E73", "all": "#5a5a5a"}
LAYER_LABEL = {"substituent": "substituent", "scaffold": "scaffold (ring)",
               "linker": "linker (bridge)", "all": "pooled"}
CHEMO_COLOR = {
    "azide": "#D55E00", "nitro": "#0072B2", "nitramine": "#56B4E9",
    "azoxy": "#009E73", "hydrazo": "#CC79A7", "azo": "#8E6C8A",
    "n_oxide": "#66A61E", "azole_ring": "#E69F00", "amino": "#999999",
    "nitrate_ester": "#7570B3", "hydroxyl": "#B3B3B3", "other": "#CFCFCF",
}
DIVERGE = LinearSegmentedColormap.from_list(
    "bwr_soft", ["#2166AC", "#77A9D0", "#F5F5F5", "#E39A7A", "#B2182B"])

PROP_TEX = {"Hf": r"$\Delta H_f$", "Q": r"$Q$", "OBgood": r"OB$_{good}$",
            "Pe": r"$P_e$", "D": r"$D$", "P": r"$P$",
            "IS": r"$h_{50}$", "nu": r"$\nu$"}


def ptex(p):
    return PROP_TEX.get(p, p)


def style(ax, grid_axis=None):
    ax.tick_params(length=2.5, pad=1.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    return ax


def plabel(ax, letter, x=-0.16, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left", color=INK)


def note(msg):
    print(f"    [skip] {msg}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _csv(path):
    try:
        df = pd.read_csv(path)
        return df if len(df) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


class Data:
    def __init__(self, results: Path, master: Path, props):
        self.R = Path(results)
        self.props = list(props)
        self.master = _csv(master)
        self.tr = _csv(self.R / "transformations.csv")
        self.frag = _csv(self.R / "fragment_goodness.csv")
        self.corr = _csv(self.R / "corr_compare.csv")
        self.layer_rep = _csv(self.R / "layer_report.csv")
        self.sens = _csv(self.R / "composite_sensitivity.csv")
        self.matrix = _csv(self.R / "candidate_rule_matrix.csv")
        self.scalars = _csv(self.R / "candidate_scalars.csv")
        self.rtm = _csv(self.R / "regression_to_mean.csv")
        self.sizes = _csv(self.R / "layer_sizes.csv")

        # restrict to properties actually present
        if len(self.tr):
            self.props = [p for p in self.props if f"z_{p}" in self.tr.columns]
            self._prepare_tr()

    def _prepare_tr(self):
        t = self.tr
        if "layer" not in t.columns:
            lay = [L.assign_layer(a, b) for a, b in zip(t.frag_from, t.frag_to)]
            t["layer"] = [x[0] for x in lay]
            t["cross_layer"] = [x[1] for x in lay]
        if "undirected_key" not in t.columns:
            t["undirected_key"] = [L.undirected_key(a, b, ml.canon_fragment)
                                   for a, b in zip(t.frag_from, t.frag_to)]
        for c in ["frag_ratio", "core_heavy", "composite_gated", "n_pairs"]:
            if c in t.columns:
                t[c] = pd.to_numeric(t[c], errors="coerce")
        self.tr = t

    @property
    def und(self):
        """Undirected rule set (mirror rows collapsed)."""
        t = self.tr
        if not len(t):
            return t
        if "direction" in t.columns and (t.direction == "canonical").any():
            return t[t.direction == "canonical"]
        return t.sort_values("composite_gated", ascending=False) \
                .drop_duplicates("undirected_key", keep="first")

    def layers_present(self, frame=None):
        f = self.und if frame is None else frame
        if not len(f) or "layer" not in f.columns:
            return []
        return [l for l in L.LAYERS if (f.layer == l).any()]

    def sig(self, frame, p):
        c = f"sig_{p}"
        if c not in frame.columns:
            return np.zeros(len(frame), dtype=bool)
        return frame[c].astype(str).str.lower().eq("true").values


# ---------------------------------------------------------------------------
# Structure drawing
# ---------------------------------------------------------------------------
def _draw_pair(ax, frag_a, frag_b, caption):
    """Render A -> B as structures. Falls back to monospace SMILES."""
    ax.axis("off")
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        import io
        from PIL import Image

        def png(fr, colour):
            m = Chem.MolFromSmiles(str(fr).replace("[*:1]", "*").replace("[*:2]", "*"))
            if m is None:
                return None
            d = rdMolDraw2D.MolDraw2DCairo(300, 190)
            o = d.drawOptions()
            o.bondLineWidth = 2
            o.padding = 0.08
            rdMolDraw2D.PrepareAndDrawMolecule(d, m)
            d.FinishDrawing()
            return Image.open(io.BytesIO(d.GetDrawingText()))

        ia, ib = png(frag_a, None), png(frag_b, None)
        if ia is None or ib is None:
            raise ValueError
        w = ia.width + 70 + ib.width
        canvas = Image.new("RGB", (w, max(ia.height, ib.height)), "white")
        canvas.paste(ia, (0, 0))
        canvas.paste(ib, (ia.width + 70, 0))
        ax.imshow(np.asarray(canvas))
        ax.annotate("", xy=(ia.width + 60, ia.height / 2),
                    xytext=(ia.width + 10, ia.height / 2),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, color=INK))
    except Exception:
        ax.text(0.5, 0.62, f"{ml.strip_attachment(frag_a) or 'H'}\n$\\downarrow$\n"
                           f"{ml.strip_attachment(frag_b) or 'H'}",
                ha="center", va="center", fontsize=7, family="monospace")
    ax.text(0.5, -0.10, caption, transform=ax.transAxes, ha="center",
            va="top", fontsize=6.8, color=INK, linespacing=1.45)


# ===========================================================================
# MAIN FIGURE A -- three design layers
# ===========================================================================
def figA(D, out, ext):
    if not len(D.und):
        return note("figA: transformations.csv missing")
    und = D.und
    lays = D.layers_present()
    if not lays:
        return note("figA: no layers present")

    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 1.0, 0.92],
                          hspace=0.62, wspace=0.42,
                          top=0.90, bottom=0.07, left=0.09, right=0.97)

    # ---- a: effect magnitude vs fragment ratio ---------------------------
    ax = fig.add_subplot(gs[0, :2]); style(ax, "y")
    has_ratio = "frag_ratio" in und.columns and und.frag_ratio.notna().any()
    if has_ratio:
        for lay in lays:
            s = und[(und.layer == lay) & und.frag_ratio.notna()]
            if not len(s):
                continue
            ax.scatter(s.frag_ratio, s.composite_gated.abs(), s=7,
                       c=LAYER_COLOR[lay], alpha=.28, lw=0,
                       label=f"{LAYER_LABEL[lay]} (n={len(s)})", zorder=3)
            # binned median trend
            q = pd.qcut(s.frag_ratio, min(8, max(2, len(s) // 30)),
                        duplicates="drop")
            g = s.groupby(q, observed=True).agg(
                x=("frag_ratio", "median"),
                y=("composite_gated", lambda v: np.nanmedian(np.abs(v))))
            ax.plot(g.x, g.y, color=LAYER_COLOR[lay], lw=1.8, zorder=5)
        conv = config.SIZE_CONVENTION["max_frag_ratio"]
        ax.axvline(conv, ls="--", lw=1, color="#B2182B", zorder=4)
        ax.text(conv + .01, .97, f"MMPA convention {conv:.2f}",
                transform=ax.get_xaxis_transform(), fontsize=6.4,
                color="#B2182B", va="top", rotation=90)
        ax.set_xlabel("exchanged fragment / molecule  (heavy-atom ratio)")
        ax.set_ylabel("|gated composite|  (SD units)")
        ax.legend(loc="upper left", handletextpad=.3, borderpad=.2)
    else:
        ax.text(.5, .5, "frag_ratio absent\n(re-run stage 04)", ha="center",
                va="center", color=MUTED, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Larger edits give larger effects — layers are not comparable")
    plabel(ax, "a")

    # ---- a2: rule counts -------------------------------------------------
    ax = fig.add_subplot(gs[0, 2]); style(ax, "x")
    cnt = [len(und[und.layer == l]) for l in lays]
    ax.barh(range(len(lays)), cnt,
            color=[LAYER_COLOR[l] for l in lays], height=.62)
    for i, v in enumerate(cnt):
        ax.text(v, i, f" {v:,}", va="center", fontsize=6.8, color=INK)
    ax.set_yticks(range(len(lays)))
    ax.set_yticklabels([LAYER_LABEL[l] for l in lays], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("undirected rules")
    ax.set_title("Rule supply")

    # ---- b: layer x property mean z --------------------------------------
    ax = fig.add_subplot(gs[1, :2])
    M = np.full((len(lays), len(D.props)), np.nan)
    S = np.zeros_like(M, dtype=bool)
    for i, lay in enumerate(lays):
        s = und[und.layer == lay]
        for j, p in enumerate(D.props):
            z = pd.to_numeric(s.get(f"z_{p}"), errors="coerce")
            if z is None or not z.notna().any():
                continue
            M[i, j] = np.nanmean(z)
            S[i, j] = D.sig(s, p).mean() >= .5
    v = np.nanmax(np.abs(M)) or 1.0
    im = ax.imshow(M, cmap=DIVERGE, norm=TwoSlopeNorm(0, -v, v), aspect="auto")
    for i in range(len(lays)):
        for j in range(len(D.props)):
            if S[i, j]:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       ec=INK, lw=1.3))
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center",
                        fontsize=6.4,
                        color="white" if abs(M[i, j]) > .6 * v else INK)
    ax.set_xticks(range(len(D.props)))
    ax.set_xticklabels([ptex(p) for p in D.props])
    ax.set_yticks(range(len(lays)))
    ax.set_yticklabels([LAYER_LABEL[l] for l in lays], fontsize=7.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=.024, pad=.015)
    cb.set_label("mean directed $z$", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=2)
    ax.set_title("Each layer moves a different property set   "
                 "(box: significant in $\\geq$50% of rules)")
    plabel(ax, "b")

    # ---- c: per-layer top rule lollipops ---------------------------------
    ax = fig.add_subplot(gs[1, 2]); style(ax, "x")
    y = 0; ticks = []; labels = []
    for lay in lays:
        s = und[und.layer == lay].nlargest(3, "composite_gated")
        for _, r in s.iterrows():
            for j, p in enumerate(D.props):
                z = pd.to_numeric(pd.Series([r.get(f"z_{p}")]),
                                  errors="coerce").iloc[0]
                if not np.isfinite(z):
                    continue
                ax.plot([0, z], [y, y], lw=.7, color=GRID, zorder=1)
                ax.scatter(z, y, s=11, color=LAYER_COLOR[lay], lw=0, zorder=3)
            ticks.append(y)
            labels.append(f"{r.chemotype_from[:8]}→{r.chemotype_to[:8]}")
            y += 1
        y += .6
    ax.axvline(0, color=INK, lw=.8)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("directed $z$ per property")
    ax.set_title("Top rules within layer")

    # ---- d: exemplars ----------------------------------------------------
    for k, lay in enumerate(lays[:3]):
        axd = fig.add_subplot(gs[2, k])
        s = und[und.layer == lay]
        if not len(s):
            axd.axis("off"); continue
        r = s.nlargest(1, "composite_gated").iloc[0]
        best = max(D.props, key=lambda p: abs(
            pd.to_numeric(pd.Series([r.get(f"z_{p}")]), errors="coerce").iloc[0]
            if np.isfinite(pd.to_numeric(pd.Series([r.get(f"z_{p}")]),
                                         errors="coerce").iloc[0]) else 0))
        cap = (f"{LAYER_LABEL[lay]}  ·  n={int(r.n_pairs)}\n"
               f"gated {r.composite_gated:+.1f} SD  ·  "
               f"largest: {best} {float(r.get(f'z_{best}', np.nan)):+.2f}")
        _draw_pair(axd, r.frag_from, r.frag_to, cap)
        if k == 0:
            plabel(axd, "d", x=-.05, y=1.0)

    fig.suptitle("Three design layers with distinct roles", x=.02, y=.975,
                 ha="left", fontsize=11, fontweight="bold")
    fig.savefig(out / f"figA_design_layers.{ext}")
    plt.close(fig)
    print(f"    -> figA_design_layers.{ext}")


# ===========================================================================
# MAIN FIGURE B -- fragment roles and residual space
# ===========================================================================
def figB(D, out, ext):
    if not len(D.frag):
        return note("figB: fragment_goodness.csv missing")

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, .72],
                          hspace=.72, wspace=.34,
                          top=.90, bottom=.13, left=.11, right=.97)

    f = D.frag.copy()
    for c in f.columns:
        if c.startswith(("A_", "s_", "t_", "n_")):
            f[c] = pd.to_numeric(f[c], errors="coerce")
    gs_scale = _csv(D.R / "global_scales.csv")
    scale = {r["property"]: (r["std"] if r["std"] else 1.0)
             for _, r in gs_scale.iterrows()} if len(gs_scale) else {}

    def zcol(p):
        a = f.get(f"A_{p}")
        if a is None:
            return None
        return a / scale.get(p, 1.0)

    # ---- a: Hf vs OBgood trade-off plane --------------------------------
    ax = fig.add_subplot(gs[0, 0]); style(ax)
    zx, zy = zcol("Hf"), zcol("OBgood")
    if zx is None or zy is None:
        ax.text(.5, .5, "Hf / OBgood absent", ha="center", va="center",
                color=MUTED, transform=ax.transAxes)
    else:
        n = pd.to_numeric(f.get("n_total_pairs"), errors="coerce").fillna(1)
        keep = zx.notna() & zy.notna() & (n >= 20)
        sizes = 6 + 26 * (np.log10(n[keep]) - np.log10(n[keep]).min()) / \
            max(np.log10(n[keep]).max() - np.log10(n[keep]).min(), 1e-9)
        chem = f.loc[keep, "chemotype"].fillna("other")
        ax.axhline(0, color=MUTED, lw=.6, ls=":")
        ax.axvline(0, color=MUTED, lw=.6, ls=":")
        for c in chem.value_counts().index[:8]:
            m = (chem == c).values
            ax.scatter(zx[keep][m], zy[keep][m], s=sizes[m],
                       c=CHEMO_COLOR.get(c, "#CFCFCF"), alpha=.68, lw=.2,
                       ec="white", label=c, zorder=3)
        ax.legend(loc="lower left", ncol=2, handletextpad=.2, columnspacing=.7,
                  borderpad=.2, markerscale=.9)
        ax.set_xlabel(f"fragment effect on {ptex('Hf')}  ($z$)")
        ax.set_ylabel(f"fragment effect on {ptex('OBgood')}  ($z$)")
        ax.text(.97, .97, "no fragment buys both", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.8, color=MUTED, style="italic")
    ax.set_title("The core design tension")
    plabel(ax, "a")

    # ---- b: chemotype x property atlas ----------------------------------
    ax = fig.add_subplot(gs[0, 1])
    chems = (f.chemotype.value_counts().index[:10].tolist()
             if "chemotype" in f.columns else [])
    M = np.full((len(chems), len(D.props)), np.nan)
    for i, c in enumerate(chems):
        sub = f[f.chemotype == c]
        for j, p in enumerate(D.props):
            a = pd.to_numeric(sub.get(f"A_{p}"), errors="coerce")
            if a is not None and a.notna().any():
                M[i, j] = np.nanmean(a) / scale.get(p, 1.0)
    if np.isfinite(M).any():
        v = np.nanmax(np.abs(M))
        im = ax.imshow(M, cmap=DIVERGE, norm=TwoSlopeNorm(0, -v, v), aspect="auto")
        ax.set_xticks(range(len(D.props)))
        ax.set_xticklabels([ptex(p) for p in D.props])
        ax.set_yticks(range(len(chems))); ax.set_yticklabels(chems, fontsize=7)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.02)
        cb.set_label("mean $z$", fontsize=7); cb.ax.tick_params(labelsize=6.5, length=2)
    ax.set_title("Fragment vocabulary")
    plabel(ax, "b")

    # ---- c: candidate maturity by layer ---------------------------------
    ax = fig.add_subplot(gs[1, :]); style(ax, "x")
    m = D.matrix
    if not len(m) or "role" not in m.columns:
        ax.text(.5, .5, "candidate_rule_matrix.csv missing", ha="center",
                va="center", color=MUTED, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
    else:
        if "layer" not in m.columns:
            m = m.copy()
            m["layer"] = [L.assign_layer(a, b)[0]
                          for a, b in zip(m.frag_from, m.frag_to)]
        cands = sorted(m.candidate_id.unique())
        lays = [l for l in L.LAYERS if (m.layer == l).any()]
        STATE = [("no applicable rule", "#E3E3E3"),
                 ("saturated (already only)", None),
                 ("opportunity remains", None)]
        rows = []
        for lay in lays:
            sub = m[m.layer == lay]
            act = set(sub[sub.role == "actionable"].candidate_id)
            alr = set(sub[sub.role == "already_optimized"].candidate_id)
            n_none = sum(1 for c in cands if c not in act and c not in alr)
            n_sat = sum(1 for c in cands if c in alr and c not in act)
            n_opp = sum(1 for c in cands if c in act)
            rows.append((lay, n_none, n_sat, n_opp))
        ypos = np.arange(len(rows))
        for i, (lay, n0, n1, n2) in enumerate(rows):
            base = LAYER_COLOR[lay]
            ax.barh(i, n0, color="#E3E3E3", lw=0)
            ax.barh(i, n1, left=n0, color=base, alpha=.95, lw=0)
            ax.barh(i, n2, left=n0 + n1, color=base, alpha=.30, lw=0)
            tot = n0 + n1 + n2
            if n1:
                ax.text(n0 + n1 / 2, i, str(n1), ha="center", va="center",
                        fontsize=6.8, color="white", fontweight="bold")
            if n2:
                ax.text(n0 + n1 + n2 / 2, i, str(n2), ha="center", va="center",
                        fontsize=6.8, color=INK)
            ax.text(tot + .6, i, f"{n2/max(tot,1):.0%} open", va="center",
                    fontsize=6.5, color=MUTED)
        ax.set_yticks(ypos)
        ax.set_yticklabels([LAYER_LABEL[l] for l, *_ in rows], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("candidates")
        ax.set_xlim(0, len(cands) * 1.16)
        h = [Patch(fc="#E3E3E3", label="no applicable rule"),
             Patch(fc="#555555", alpha=.95, label="saturated (already optimised)"),
             Patch(fc="#555555", alpha=.30, label="opportunity remains")]
        ax.legend(handles=h, ncol=3, loc="lower center",
                  bbox_to_anchor=(.5, -.42), handlelength=1.2)
    ax.set_title("Where the screen is saturated and where it is not")
    plabel(ax, "c", x=-.06)

    fig.suptitle("Fragment roles and residual design space", x=.02, y=.975,
                 ha="left", fontsize=11, fontweight="bold")
    fig.savefig(out / f"figB_fragment_roles.{ext}")
    plt.close(fig)
    print(f"    -> figB_fragment_roles.{ext}")


# ===========================================================================
# SUPPORTING
# ===========================================================================
def figS1(D, out, ext):
    """Correlation divergence per layer."""
    c = D.corr
    if not len(c) or "divergence" not in c.columns:
        return note("S1: corr_compare.csv missing")
    if "layer" not in c.columns:
        c = c.assign(layer="all")
    c = c.dropna(subset=["divergence"])
    lays = [l for l in ("all",) + tuple(L.LAYERS) if (c.layer == l).any()]

    order = (c[c.layer == lays[0]].reindex(
        c[c.layer == lays[0]].divergence.abs().sort_values(ascending=False).index))
    pairs = [f"{r.p1}–{r.p2}" for r in order.itertuples()][:14]

    fig, axes = plt.subplots(1, len(lays), figsize=(2.15 * len(lays), 4.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, lay in zip(axes, lays):
        style(ax, "x")
        s = c[c.layer == lay].copy()
        s["k"] = s.p1 + "–" + s.p2
        s = s.set_index("k").reindex(pairs).dropna(subset=["divergence"])
        col = [LAYER_COLOR[lay] if v > 0 else "#B2182B" for v in s.divergence]
        ax.barh(range(len(s)), s.divergence, color=col, height=.68)
        ax.axvline(0, color=INK, lw=.8)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(s.index, fontsize=6.2)
        ax.invert_yaxis()
        ax.set_title(LAYER_LABEL[lay], fontsize=8.5)
        ax.set_xlabel("edit $r$ − static $r$")
    fig.suptitle("Population coupling breaks differently in each layer",
                 x=.02, ha="left", fontsize=10.5, fontweight="bold")
    fig.text(.02, -.02,
             "Delta correlations are attenuated by noise in a way static "
             "correlations are not (regression dilution), so a positive\n"
             "divergence is the null expectation, not evidence. Compare "
             "against a permutation null before claiming decoupling.",
             fontsize=6.4, color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(out / f"figS1_divergence_by_layer.{ext}")
    plt.close(fig)
    print(f"    -> figS1_divergence_by_layer.{ext}")


def figS2(D, out, ext):
    """Method audit: size conventions, mirror dedup, FDR."""
    und, full = D.und, D.tr
    if not len(full):
        return note("S2: transformations.csv missing")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))

    ax = axes[0]; style(ax, "y")
    if "frag_ratio" in und.columns and und.frag_ratio.notna().any():
        bins = np.linspace(0, min(1, und.frag_ratio.max() * 1.05), 34)
        for lay in D.layers_present():
            s = und[(und.layer == lay) & und.frag_ratio.notna()]
            ax.hist(s.frag_ratio, bins=bins, color=LAYER_COLOR[lay], alpha=.62,
                    label=LAYER_LABEL[lay], lw=0)
        ax.axvline(config.SIZE_CONVENTION["max_frag_ratio"], ls="--", lw=1,
                   color="#B2182B")
        ax.set_xlabel("fragment / molecule ratio"); ax.set_ylabel("rules")
        ax.legend(fontsize=6)
    ax.set_title("a  Size audit")

    ax = axes[1]; style(ax, "y")
    ax.bar([0, 1], [len(full), len(und)], color=["#BBBBBB", LAYER_COLOR["all"]],
           width=.55)
    for i, v in enumerate([len(full), len(und)]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["directed rows", "undirected"])
    ax.set_ylabel("rules")
    ax.set_title("b  Mirror deduplication")

    ax = axes[2]; style(ax, "y")
    frac = []
    lays = D.layers_present()
    for lay in lays:
        s = und[und.layer == lay]
        any_sig = np.zeros(len(s), dtype=bool)
        for p in D.props:
            any_sig |= D.sig(s, p)
        frac.append(any_sig.mean() if len(s) else 0)
    ax.bar(range(len(lays)), frac, color=[LAYER_COLOR[l] for l in lays], width=.6)
    ax.set_xticks(range(len(lays)))
    ax.set_xticklabels([LAYER_LABEL[l] for l in lays], fontsize=6.5, rotation=18)
    ax.set_ylabel("fraction with $\\geq$1 significant property")
    ax.set_ylim(0, 1)
    ax.set_title("c  After global FDR")

    fig.suptitle("Method audit", x=.02, ha="left", fontsize=10.5,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"figS2_method_audit.{ext}")
    plt.close(fig)
    print(f"    -> figS2_method_audit.{ext}")


def figS3(D, out, ext):
    """Composite sensitivity to including h50 / nu."""
    s = D.sens
    if not len(s):
        return note("S3: composite_sensitivity.csv missing (run 13)")
    jcol = [c for c in s.columns if c.startswith("top") and "jaccard" in c]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    for ax, col, lab in zip(
            axes, ["spearman_vs_default"] + jcol,
            ["Spearman $\\rho$ vs default composite", "top-$N$ overlap (Jaccard)"]):
        style(ax, "y")
        presets = [p for p in s.preset.unique() if p != config.DEFAULT_COMPOSITE]
        lays = [l for l in L.LAYERS if (s.layer == l).any()]
        w = .8 / max(len(presets), 1)
        for k, pr in enumerate(presets):
            v = [float(s[(s.layer == l) & (s.preset == pr)][col].iloc[0])
                 if len(s[(s.layer == l) & (s.preset == pr)]) else np.nan
                 for l in lays]
            ax.bar(np.arange(len(lays)) - .4 + w * (k + .5), v, w * .9,
                   label=pr, lw=0)
        ax.axhline(.9, ls=":", lw=.8, color=MUTED)
        ax.set_xticks(range(len(lays)))
        ax.set_xticklabels([LAYER_LABEL[l] for l in lays], fontsize=6.5, rotation=18)
        ax.set_ylim(0, 1.05); ax.set_ylabel(lab, fontsize=7)
        ax.legend(fontsize=6.5)
    fig.suptitle("Conclusions are insensitive to the out-of-domain $h_{50}$ mapping",
                 x=.02, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"figS3_composite_sensitivity.{ext}")
    plt.close(fig)
    print(f"    -> figS3_composite_sensitivity.{ext}")


def figS4(D, out, ext):
    """Deduplicated volcano: effect vs significance."""
    und = D.und
    if not len(und):
        return note("S4: transformations.csv missing")
    props = [p for p in D.props if f"p_{p}" in und.columns]
    if not props:
        return note("S4: no p-value columns")
    show = props[:6]
    ncol = 3
    nrow = int(np.ceil(len(show) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 2.35 * nrow),
                             sharex=False, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes, show):
        style(ax)
        z = pd.to_numeric(und[f"z_{p}"], errors="coerce")
        pv = pd.to_numeric(und[f"p_{p}"], errors="coerce").clip(lower=1e-300)
        y = -np.log10(pv)
        sig = D.sig(und, p)
        ax.scatter(z[~sig], y[~sig], s=3, c="#D9D9D9", lw=0, zorder=2)
        for lay in D.layers_present():
            m = sig & (und.layer == lay).values
            ax.scatter(z[m], y[m], s=4.5, c=LAYER_COLOR[lay], lw=0, alpha=.75,
                       zorder=3, label=LAYER_LABEL[lay])
        ax.axvline(0, color=INK, lw=.7)
        ax.set_title(ptex(p), fontsize=8.5)
        ax.set_xlabel("directed $z$")
    for ax in axes[len(show):]:
        ax.axis("off")
    axes[0].set_ylabel("$-\\log_{10} p$")
    h = [Line2D([], [], marker="o", ls="", ms=4, color=LAYER_COLOR[l],
                label=LAYER_LABEL[l]) for l in D.layers_present()]
    h.append(Line2D([], [], marker="o", ls="", ms=4, color="#D9D9D9",
                    label="not significant (global BH)"))
    fig.legend(handles=h, ncol=4, loc="lower center", bbox_to_anchor=(.5, -.04))
    fig.suptitle("Transformation catalogue (mirror rows removed)", x=.02,
                 ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"figS4_volcano.{ext}")
    plt.close(fig)
    print(f"    -> figS4_volcano.{ext}")


def figS5(D, out, ext):
    """Regression-to-the-mean control, per-property standardised."""
    r = D.rtm
    if not len(r):
        return note("S5: regression_to_mean.csv missing")
    r = r.copy()
    for c in ["centered", "mean_delta"]:
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r = r.dropna(subset=["centered", "mean_delta"])
    props = [p for p in D.props if (r.property == p).any()]
    ncol = 4
    nrow = int(np.ceil(len(props) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 1.9 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes, props):
        style(ax)
        s = r[r.property == p]
        ax.scatter(s.centered, s.mean_delta, s=9, c=LAYER_COLOR["all"],
                   alpha=.6, lw=0)
        if len(s) > 2 and s.centered.std() > 0:
            b, a = np.polyfit(s.centered, s.mean_delta, 1)
            xs = np.linspace(s.centered.min(), s.centered.max(), 20)
            ax.plot(xs, a + b * xs, color="#B2182B", lw=1.3)
            ax.text(.04, .06, f"slope {b:+.2f}", transform=ax.transAxes,
                    fontsize=6.5, color="#B2182B")
        ax.axhline(0, color=MUTED, lw=.6, ls=":")
        ax.axvline(0, color=MUTED, lw=.6, ls=":")
        ax.set_title(ptex(p), fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes[len(props):]:
        ax.axis("off")
    fig.supxlabel("candidate − database mean", fontsize=7.5)
    fig.supylabel("mean Δ from edits", fontsize=7.5)
    fig.suptitle("Regression to the mean: predicted gains shrink with distance "
                 "from the pool", x=.02, ha="left", fontsize=10.5,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"figS5_regression_to_mean.{ext}")
    plt.close(fig)
    print(f"    -> figS5_regression_to_mean.{ext}")


# ===========================================================================
FIGS = {"figA": figA, "figB": figB, "S1": figS1, "S2": figS2,
        "S3": figS3, "S4": figS4, "S5": figS5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(config.RESULTS))
    ap.add_argument("--master", default=str(config.MASTER_CSV))
    ap.add_argument("--properties", nargs="+", default=config.PROPERTIES)
    ap.add_argument("--output-dir", default=str(config.FIG_DIR))
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--only", nargs="+", default=None, choices=list(FIGS))
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    D = Data(Path(args.results), Path(args.master), args.properties)
    print(f"Results: {args.results}\nProperties in play: {D.props}")
    print(f"Rules: {len(D.tr)} directed / {len(D.und)} undirected")

    for name in (args.only or list(FIGS)):
        print(f"  {name}")
        try:
            FIGS[name](D, out, args.format)
        except Exception as e:                     # keep the batch alive
            note(f"{name}: {type(e).__name__}: {e}")
    print(f"\nFigures in {out}/")


if __name__ == "__main__":
    main()
