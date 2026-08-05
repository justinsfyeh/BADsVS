#!/usr/bin/env python3
"""
15_axis_figures.py -- publication figures for the three-axis MMPA analysis.

Figure logic follows the measured shape of the rule inventory:

  substituent   21 rules /   7 fragments  == C(7,2)   COMPLETE   n ~ 5,000 ea
  bridge         3 rules /   3 fragments  == C(3,2)   COMPLETE   n ~ 4,000 ea
  ring       4,544 rules / 161 fragments  sparse sample
  confounded   860 rules   fragment spans ring AND bridge -- excluded

Two axes are recovered exactly, which independently validates the approach:
MMPA rediscovered the enumerated functional groups and bridges from 108k
molecules without being told the combinatorial rule. A complete, densely
supported rule set is a MATRIX and a CYCLE, not two more scatter clouds --
hence panel a is a 7x7 exchange matrix and panel b a 3-node thermodynamic
cycle carrying its own closure check.

The ring axis is different again: its strongest rules share one starting
fragment and differ in what replaces the methyl, so the apparent "ring" signal
is largely ring DECORATION. Figure B separates that explicitly rather than
ranking ring rules as if they were interchangeable.

Usage:
  <tartarus python> 15_axis_figures.py
  <tartarus python> 15_axis_figures.py --only figA figB --format pdf
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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

import config
import layers as L
import mmpa_lib as ml
import figstyle as fs

warnings.filterwarnings("ignore", category=RuntimeWarning)
fs.apply()

SUB_NAME = {"C": "CH$_3$", "N": "NH$_2$", "O": "OH", "[H]": "H", "": "H",
            "N=[N+]=[N-]": "N$_3$", "[N+](=O)[O-]": "NO$_2$",
            "N[N+](=O)[O-]": "NHNO$_2$"}
BRIDGE_NAME = {"[*:1]N=N[*:2]": ("azo", "–N=N–"),
               "[*:1]NN[*:2]": ("hydrazo", "–NH–NH–"),
               "[*:1]N=[N+]([*:2])[O-]": ("azoxy", "–N=N$^{+}$(O$^{-}$)–")}
SUB_ORDER = ["C", "[H]", "N", "O", "N=[N+]=[N-]", "N[N+](=O)[O-]", "[N+](=O)[O-]"]


def sub_name(frag):
    return SUB_NAME.get(ml.strip_attachment(frag) or "[H]",
                        ml.strip_attachment(frag) or "H")


def note(m):
    print(f"    [skip] {m}")


def _csv(p):
    try:
        d = pd.read_csv(p)
        return d if len(d) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


class Data:
    def __init__(self, results, master, props):
        self.R = Path(results)
        self.props = list(props)
        self.master = _csv(master)
        self.tr = _csv(self.R / "transformations.csv")
        self.frag = _csv(self.R / "fragment_goodness.csv")
        self.corr = _csv(self.R / "corr_compare.csv")
        self.sens = _csv(self.R / "composite_sensitivity.csv")
        self.matrix = _csv(self.R / "candidate_rule_matrix.csv")
        self.rtm = _csv(self.R / "regression_to_mean.csv")
        if len(self.tr):
            self.props = [p for p in self.props if f"z_{p}" in self.tr.columns]
            self._prep()

    def _prep(self):
        t = self.tr
        if "axis" not in t.columns:
            a = [L.assign_axis(x, y) for x, y in zip(t.frag_from, t.frag_to)]
            t["axis"] = [i[0] for i in a]
            t["axis_clean"] = [i[1] for i in a]
        for c in ["frag_ratio", "composite_gated", "n_pairs", "core_heavy"]:
            if c in t.columns:
                t[c] = pd.to_numeric(t[c], errors="coerce")
        self.tr = t

    @property
    def und(self):
        t = self.tr
        if not len(t):
            return t
        if "direction" in t.columns and (t.direction == "canonical").any():
            return t[t.direction == "canonical"]
        return t.drop_duplicates("undirected_key") if "undirected_key" in t else t

    @property
    def both(self):
        return self.tr

    def axes_present(self):
        u = self.und
        return [a for a in fs.AXIS_ORDER if len(u) and (u.axis == a).any()]

    def sig(self, f, p):
        c = f"sig_{p}"
        if c not in f.columns:
            return np.zeros(len(f), dtype=bool)
        return f[c].astype(str).str.lower().eq("true").values


# ===========================================================================
def figA(D, out, ext):
    """Two of three enumeration axes recovered completely."""
    if not len(D.und):
        return note("figA: transformations.csv missing")
    und, both = D.und, D.both
    axes_p = D.axes_present()

    fig = plt.figure(figsize=(7.20, 6.55))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.00, 0.86], width_ratios=[1.06, 1.00],
        left=0.105, right=0.955, top=0.845, bottom=0.085,
        hspace=0.56, wspace=0.32)

    # ---------------- a: substituent exchange matrix -------------------
    ax = fig.add_subplot(gs[0, 0])
    sb = both[both.axis == "substituent"]
    if len(sb):
        present = set(sb.frag_from) | set(sb.frag_to)
        frs = sorted(present,
                     key=lambda f: SUB_ORDER.index(ml.strip_attachment(f) or "[H]")
                     if (ml.strip_attachment(f) or "[H]") in SUB_ORDER else 99)
        names = [sub_name(f) for f in frs]
        k = len(frs)
        M = np.full((k, k), np.nan)
        idx = {f: i for i, f in enumerate(frs)}
        for _, r in sb.iterrows():
            M[idx[r.frag_from], idx[r.frag_to]] = r.composite_gated
        im, v = fs.heat(ax, M, names, names, fmt="{:+.1f}", fontsize=6.3)
        for i in range(k):
            ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fc=fs.PANEL_TINT,
                                   ec=fs.PAPER, lw=1.4, zorder=3.5))
        ax.set_xlabel("to", color=fs.INK_SOFT)
        ax.set_ylabel("from", color=fs.INK_SOFT)
        ax.tick_params(labelsize=7.2)
        fs.cbar(fig, im, ax, "gated composite (SD)", fraction=0.042)
        fs.title(ax, f"Substituent axis: {len(und[und.axis=='substituent'])} "
                     f"rules = C({k},2)",
                 sub=f"complete   ·   median {int(sb.n_pairs.median()):,} "
                     f"pairs per rule")
    else:
        fs.bare(ax)
    fs.panel(ax, "a", x=-0.20)

    # ---------------- b: bridge cycle ----------------------------------
    ax = fig.add_subplot(gs[0, 1])
    fs.bare(ax); ax.set_xticks([]); ax.set_yticks([])
    br = und[und.axis == "bridge"]
    if len(br) and "mean_d_Q" in br.columns:
        nodes = sorted(set(br.frag_from) | set(br.frag_to))[:3]
        coords = [(0.0, 0.90), (-0.88, -0.60), (0.88, -0.60)]
        pos = dict(zip(nodes, coords))
        R = 0.42
        for n in nodes:
            x, y = pos[n]
            ax.add_patch(Circle((x, y), R, fc=fs.AXIS_FILL["bridge"],
                                ec=fs.AXIS_COLOR["bridge"], lw=1.5, zorder=4))
            nm, formula = BRIDGE_NAME.get(n, (ml.strip_attachment(n), ""))
            ax.text(x, y + 0.10, nm, ha="center", va="center", fontsize=7.6,
                    fontweight="semibold", color=fs.INK, zorder=5)
            ax.text(x, y - 0.14, formula, ha="center", va="center",
                    fontsize=6.5, color=fs.INK_SOFT, zorder=5)
        for _, r in br.iterrows():
            (x0, y0), (x1, y1) = pos[r.frag_from], pos[r.frag_to]
            dq = float(r["mean_d_Q"])
            d = np.hypot(x1 - x0, y1 - y0)
            ux, uy = (x1 - x0) / d, (y1 - y0) / d
            a0 = (x0 + ux * R * 1.06, y0 + uy * R * 1.06)
            a1 = (x1 - ux * R * 1.06, y1 - uy * R * 1.06)
            ax.add_patch(FancyArrowPatch(
                a0, a1, arrowstyle="-|>", mutation_scale=11, lw=1.5,
                color=fs.AXIS_COLOR["bridge"], zorder=3,
                shrinkA=0, shrinkB=0))
            fs.badge(ax, (a0[0] + a1[0]) / 2, (a0[1] + a1[1]) / 2,
                     f"{dq:+.0f}", fs.AXIS_COLOR["bridge"])
        m = {(r.frag_from, r.frag_to): float(r["mean_d_Q"]) for _, r in br.iterrows()}

        def g(a, b):
            return m.get((a, b), -m[(b, a)] if (b, a) in m else np.nan)
        a_, b_, c_ = nodes
        leg, direct = g(a_, b_) + g(b_, c_), g(a_, c_)
        ax.set_xlim(-1.58, 1.58); ax.set_ylim(-1.30, 1.60)
        if np.isfinite(leg) and np.isfinite(direct):
            ax.text(0.5, -0.045,
                    f"thermodynamic closure   {leg:+.1f}  vs  {direct:+.1f} "
                    f"cal g$^{{-1}}$   (Δ {abs(leg-direct):.1f}, "
                    f"{abs(leg-direct)/max(abs(direct),1e-9):.1%})",
                    transform=ax.transAxes, ha="center", fontsize=6.7,
                    color=fs.INK_SOFT)
        fs.title(ax, f"Bridge axis: {len(br)} rules = C({len(nodes)},2)",
                 sub=f"complete   ·   $\\Delta Q$ in cal g$^{{-1}}$   ·   "
                     f"n $\\approx$ {int(br.n_pairs.median()):,} pairs per edge")
    fs.panel(ax, "b", x=-0.06)

    # ---------------- c: axis x property -------------------------------
    ax = fig.add_subplot(gs[1, 0])
    show = axes_p
    M = np.full((len(show), len(D.props)), np.nan)
    S = np.zeros_like(M, dtype=bool)
    for i, a in enumerate(show):
        s = und[und.axis == a]
        for j, p in enumerate(D.props):
            z = pd.to_numeric(s.get(f"z_{p}"), errors="coerce")
            if z is not None and z.notna().any():
                M[i, j] = np.nanmean(z)
                S[i, j] = D.sig(s, p).mean() >= .5
    im, v = fs.heat(ax, M, [fs.ptex(p) for p in D.props],
                    [fs.AXIS_LABEL[a] for a in show], fontsize=6.3)
    for i, a in enumerate(show):
        for j in range(len(D.props)):
            if S[i, j]:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       ec=fs.INK, lw=1.0, zorder=5))
        if a == "confounded":
            ax.add_patch(Rectangle((-.5, i - .5), len(D.props), 1, fill=False,
                                   ec=fs.ACCENT_POS, lw=1.1,
                                   ls=(0, (2.5, 1.8)), zorder=6))
    ax.set_yticklabels([f"{fs.AXIS_LABEL[a]}  ({len(und[und.axis==a]):,})"
                        for a in show], fontsize=7.0)
    fs.cbar(fig, im, ax, "mean directed $z$", fraction=0.034)
    fs.title(ax, "Each axis moves a different property set")
    ax.text(0, -0.32, "outline: significant in $\\geq$50% of rules        "
                      "dashed: fragment spans two axes — excluded from claims",
            transform=ax.transAxes, fontsize=6.3, color=fs.MUTED)
    fs.panel(ax, "c", x=-0.20)

    # ---------------- d: coverage --------------------------------------
    ax = fig.add_subplot(gs[1, 1]); fs.clean(ax, grid="x")
    cnt = [len(und[und.axis == a]) for a in axes_p]
    ypos = np.arange(len(axes_p))
    ax.barh(ypos, cnt, color=[fs.AXIS_COLOR[a] for a in axes_p], height=.58,
            zorder=3)
    ax.set_xscale("log")
    for i, (a, n) in enumerate(zip(axes_p, cnt)):
        s = und[und.axis == a]
        nf = len(set(s.frag_from) | set(s.frag_to))
        full = nf * (nf - 1) // 2
        if a == "confounded":
            tag = f"{n:,}   excluded"
        elif n == full and n > 1:
            tag = f"{n:,}   complete"
        else:
            tag = f"{n:,}   of {full:,}"
        ax.text(n * 1.30, i, tag, va="center", fontsize=6.7, color=fs.INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels([fs.AXIS_LABEL[a] for a in axes_p], fontsize=7.4)
    ax.invert_yaxis()
    ax.set_xlim(right=max(cnt) * 26)
    ax.set_xlabel("undirected rules (log scale)")
    fs.title(ax, "Coverage of each axis")
    fs.panel(ax, "d", x=-0.16)

    fs.figtitle(fig, "Two of three enumeration axes recovered completely",
                "MMPA rediscovered the 7 enumerated substituents and 3 bridges "
                "from 108k molecules without being given the combinatorial rule")
    fig.savefig(out / f"figA_axes_recovered.{ext}")
    plt.close(fig)
    print(f"    -> figA_axes_recovered.{ext}")


# ===========================================================================
def figB(D, out, ext):
    """Ring axis: decoration dominates identity."""
    und = D.und
    r = und[und.axis == "ring"] if len(und) else pd.DataFrame()
    if not len(r):
        return note("figB: no ring-axis rules")

    fig = plt.figure(figsize=(7.20, 5.35))
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1.00, 0.90], width_ratios=[0.78, 1.11, 1.11],
        left=0.090, right=0.975, top=0.820, bottom=0.070,
        hspace=0.70, wspace=0.40)

    same = r[r.chemotype_from == r.chemotype_to]
    diff = r[r.chemotype_from != r.chemotype_to]

    # ---------------- a: swap vs decoration ----------------------------
    ax = fig.add_subplot(gs[0, 0]); fs.clean(ax, grid="y")
    parts = [("ring only", same, fs.AXIS_FILL["ring"], "#C79A4B"),
             ("+ decoration", diff, fs.AXIS_COLOR["ring"], fs.AXIS_COLOR["ring"])]
    tops = []
    for i, (lab, s, face, edge) in enumerate(parts):
        if not len(s):
            continue
        v = s.composite_gated.dropna()
        pv = ax.violinplot([v], positions=[i], widths=.78, showextrema=False)
        for b in pv["bodies"]:
            b.set_facecolor(face); b.set_alpha(.85)
            b.set_edgecolor(edge); b.set_linewidth(.7)
        q1, q3 = v.quantile(.25), v.quantile(.75)
        ax.plot([i, i], [q1, q3], color=fs.INK, lw=2.6, solid_capstyle="round",
                zorder=5)
        ax.scatter([i], [v.median()], s=16, color=fs.PAPER, ec=fs.INK, lw=1.0,
                   zorder=6)
        tops.append((i, len(s), v.median()))
    ax.axhline(0, color=fs.MUTED, lw=.7, ls=(0, (2, 2)), zorder=1)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + (yhi - ylo) * .18)
    for i, n, med in tops:
        ax.text(i, yhi + (yhi - ylo) * .02, f"$n$={n:,}\nmed {med:+.2f}",
                ha="center", va="bottom", fontsize=6.4, color=fs.INK_SOFT)
    ax.set_xticks(range(len(parts)))
    ax.set_xticklabels([p[0] for p in parts], fontsize=7.2)
    ax.set_ylabel("gated composite (SD)")
    ax.set_xlim(-.62, len(parts) - .38)
    fs.title(ax, "Decoration, not identity")
    fs.panel(ax, "a", x=-0.40)

    # ---------------- b: ring N-count ----------------------------------
    ax = fig.add_subplot(gs[0, 1:]); fs.clean(ax, grid="y")
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")

        def ringN(f):
            m = Chem.MolFromSmiles(str(f))
            if m is None or not m.GetRingInfo().NumRings():
                return np.nan
            big = max(m.GetRingInfo().AtomRings(), key=len)
            return sum(1 for i in big if m.GetAtomWithIdx(i).GetSymbol() == "N")
        rr = r.assign(dN=r.frag_to.map(ringN) - r.frag_from.map(ringN)) \
              .dropna(subset=["dN"])
        levels = sorted(rr.dN.unique())
        w = .78 / max(len(D.props), 1)
        cmap = plt.get_cmap("cividis")
        cols = [cmap(t) for t in np.linspace(.06, .94, len(D.props))]
        for j, p in enumerate(D.props):
            ys = [pd.to_numeric(rr[rr.dN == d][f"z_{p}"], errors="coerce").mean()
                  for d in levels]
            ax.bar(np.arange(len(levels)) - .39 + w * (j + .5), ys, w * .88,
                   color=cols[j], label=fs.ptex(p), lw=0, zorder=3)
        ax.axhline(0, color=fs.INK, lw=.8, zorder=4)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([f"{int(d):+d}\n$n$={int((rr.dN==d).sum()):,}"
                            for d in levels], fontsize=7.0)
        ax.set_xlabel("change in ring nitrogen count")
        ax.set_ylabel("mean directed $z$")
        ax.legend(ncol=len(D.props), loc="lower center",
                  bbox_to_anchor=(.5, 1.02), fontsize=6.5)
    except Exception as e:
        ax.text(.5, .5, f"rdkit unavailable\n{e}", ha="center", va="center",
                transform=ax.transAxes, color=fs.MUTED)
    ax.set_title("More ring nitrogen improves every property together",
                 pad=22, color=fs.INK)
    fs.panel(ax, "b", x=-0.085, y=1.10)

    # ---------------- c: exemplars -------------------------------------
    top = r.nlargest(3, "composite_gated")
    for k, (_, row) in enumerate(top.iterrows()):
        axd = fig.add_subplot(gs[1, k])
        cap = (f"{row.chemotype_from} $\\rightarrow$ {row.chemotype_to}\n"
               f"gated {row.composite_gated:+.1f} SD  ·  $n$={int(row.n_pairs):,}")
        _draw_pair(axd, row.frag_from, row.frag_to, cap)
        if k == 0:
            fs.panel(axd, "c", x=-0.04, y=0.98)

    fs.figtitle(fig, "Ring axis: what sits on the ring matters more than "
                     "which ring",
                "The strongest ring rules share one starting fragment and "
                "differ in the substituent replacing its methyl group")
    fig.savefig(out / f"figB_ring_axis.{ext}")
    plt.close(fig)
    print(f"    -> figB_ring_axis.{ext}")


def _draw_pair(ax, a, b, caption):
    fs.bare(ax); ax.set_xticks([]); ax.set_yticks([])
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        import io
        from PIL import Image

        def png(fr):
            m = Chem.MolFromSmiles(str(fr).replace("[*:1]", "*").replace("[*:2]", "*"))
            if m is None:
                return None
            d = rdMolDraw2D.MolDraw2DCairo(340, 200)
            o = d.drawOptions()
            o.bondLineWidth = 2
            o.padding = 0.06
            rdMolDraw2D.PrepareAndDrawMolecule(d, m)
            d.FinishDrawing()
            return Image.open(io.BytesIO(d.GetDrawingText()))
        ia, ib = png(a), png(b)
        if ia is None or ib is None:
            raise ValueError
        gapw = 62
        cv = Image.new("RGB", (ia.width + gapw + ib.width,
                               max(ia.height, ib.height)), "white")
        cv.paste(ia, (0, 0)); cv.paste(ib, (ia.width + gapw, 0))
        ax.imshow(np.asarray(cv), interpolation="antialiased")
        yc = ia.height / 2
        ax.add_patch(FancyArrowPatch(
            (ia.width + 13, yc), (ia.width + gapw - 13, yc),
            arrowstyle="-|>", mutation_scale=10, lw=1.3, color=fs.INK_SOFT))
    except Exception:
        ax.text(.5, .6, f"{ml.strip_attachment(a) or 'H'}\n$\\downarrow$\n"
                        f"{ml.strip_attachment(b) or 'H'}", ha="center",
                va="center", fontsize=7, family="monospace")
    ax.text(.5, -0.02, caption, transform=ax.transAxes, ha="center", va="top",
            fontsize=6.8, color=fs.INK_SOFT, linespacing=1.5)


# ===========================================================================
def figS1(D, out, ext):
    c = D.corr
    if not len(c) or "divergence" not in c.columns:
        return note("S1: corr_compare.csv missing")
    if "layer" not in c.columns:
        c = c.assign(layer="all")
    c = c.dropna(subset=["divergence"])
    keys = list(dict.fromkeys(c.layer))
    o = c[c.layer == keys[0]]
    order = [f"{r.p1}–{r.p2}" for r in
             o.reindex(o.divergence.abs().sort_values(ascending=False).index)
             .itertuples()][:14]
    fig, axes = plt.subplots(1, len(keys), figsize=(1.95 * len(keys) + .6, 4.1),
                             sharey=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=.145, right=.98, top=.80, bottom=.17, wspace=.22)
    for ax, k in zip(axes, keys):
        fs.clean(ax, grid="x")
        s = c[c.layer == k].copy()
        s["kk"] = s.p1 + "–" + s.p2
        s = s.set_index("kk").reindex(order).dropna(subset=["divergence"])
        base = fs.AXIS_COLOR.get(k, fs.INK_SOFT)
        col = [base if v > 0 else fs.ACCENT_POS for v in s.divergence]
        ax.barh(range(len(s)), s.divergence, color=col, height=.66, zorder=3)
        ax.axvline(0, color=fs.INK, lw=.8, zorder=4)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(s.index, fontsize=6.4)
        ax.invert_yaxis()
        fs.title(ax, str(k))
        ax.set_xlabel("edit $r$ − static $r$")
    fs.figtitle(fig, "Population coupling breaks differently on each axis")
    fs.footnote(fig, "Delta correlations are attenuated by noise in a way static "
                     "correlations are not (regression dilution), so a positive "
                     "divergence is the null\nexpectation rather than evidence. "
                     "Compare against a permutation null before claiming decoupling.")
    fig.savefig(out / f"figS1_divergence.{ext}")
    plt.close(fig)
    print(f"    -> figS1_divergence.{ext}")


def figS2(D, out, ext):
    und, full = D.und, D.tr
    if not len(full):
        return note("S2: transformations.csv missing")
    axes_p = D.axes_present()
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.40))
    fig.subplots_adjust(left=.075, right=.985, top=.70, bottom=.22, wspace=.44)

    ax = axes[0]; fs.clean(ax, grid="y")
    if "frag_ratio" in und.columns and und.frag_ratio.notna().any():
        bins = np.linspace(0, min(1, und.frag_ratio.max() * 1.05), 30)
        for a in axes_p:
            s = und[(und.axis == a) & und.frag_ratio.notna()]
            if len(s):
                ax.hist(s.frag_ratio, bins=bins, color=fs.AXIS_COLOR[a],
                        alpha=.72, label=fs.AXIS_LABEL[a], lw=0, zorder=3)
        ax.axvline(config.SIZE_CONVENTION["max_frag_ratio"], ls=(0, (2.5, 1.8)),
                   lw=1, color=fs.ACCENT_POS, zorder=4)
        ax.legend(fontsize=5.9)
        ax.set_xlabel("fragment / molecule"); ax.set_ylabel("rules")
    fs.title(ax, "a   Size audit")

    ax = axes[1]; fs.clean(ax, grid="y")
    ax.bar([0, 1], [len(full), len(und)], color=[fs.HAIRLINE, fs.INK_SOFT],
           width=.52, zorder=3)
    for i, v in enumerate([len(full), len(und)]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=6.8,
                color=fs.INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["directed", "undirected"], fontsize=7)
    ax.set_ylabel("rules")
    fs.title(ax, "b   Mirror dedup")

    ax = axes[2]; fs.clean(ax, grid="y")
    fr = []
    for a in axes_p:
        s = und[und.axis == a]
        m = np.zeros(len(s), dtype=bool)
        for p in D.props:
            m |= D.sig(s, p)
        fr.append(m.mean() if len(s) else 0)
    ax.bar(range(len(axes_p)), fr, color=[fs.AXIS_COLOR[a] for a in axes_p],
           width=.58, zorder=3)
    ax.set_xticks(range(len(axes_p)))
    ax.set_xticklabels([fs.AXIS_LABEL[a] for a in axes_p], fontsize=6.2,
                       rotation=24, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("fraction significant")
    fs.title(ax, "c   After global FDR")

    ax = axes[3]
    n_conf = len(und[und.axis == "confounded"])
    ax.pie([len(und) - n_conf, n_conf],
           colors=[fs.INK_SOFT, fs.AXIS_COLOR["confounded"]],
           startangle=90, wedgeprops=dict(width=.42, lw=0),
           labels=[f"single-axis\n{len(und)-n_conf:,}",
                   f"confounded\n{n_conf:,}"],
           textprops=dict(fontsize=6.4, color=fs.INK))
    fs.title(ax, "d   Attributable")

    fs.figtitle(fig, "Method audit", y=.985)
    fig.savefig(out / f"figS2_method_audit.{ext}")
    plt.close(fig)
    print(f"    -> figS2_method_audit.{ext}")


def figS3(D, out, ext):
    s = D.sens
    if not len(s):
        return note("S3: composite_sensitivity.csv missing (run stage 13)")
    j = [c for c in s.columns if c.startswith("top") and "jaccard" in c]
    cols = ["spearman_vs_default"] + j
    labs = ["Spearman $\\rho$ vs default", "top-$N$ overlap (Jaccard)"]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.15 * len(cols), 2.6))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=.10, right=.98, top=.72, bottom=.22, wspace=.30)
    keys = list(dict.fromkeys(s.layer))
    pres = [p for p in s.preset.unique() if p != config.DEFAULT_COMPOSITE]
    pal = [fs.AXIS_COLOR["substituent"], fs.AXIS_COLOR["ring"]]
    for ax, col, lab in zip(axes, cols, labs):
        fs.clean(ax, grid="y")
        w = .74 / max(len(pres), 1)
        for k, pr in enumerate(pres):
            v = [float(s[(s.layer == q) & (s.preset == pr)][col].iloc[0])
                 if len(s[(s.layer == q) & (s.preset == pr)]) else np.nan
                 for q in keys]
            ax.bar(np.arange(len(keys)) - .37 + w * (k + .5), v, w * .88,
                   label=pr, color=pal[k % len(pal)], lw=0, zorder=3)
        ax.axhline(.9, ls=(0, (2, 2)), lw=.8, color=fs.MUTED, zorder=2)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, fontsize=6.6, rotation=20, ha="right")
        ax.set_ylim(0, 1.06); ax.set_ylabel(lab, fontsize=7)
        ax.legend(fontsize=6.4)
    fs.figtitle(fig, "Conclusions are insensitive to the out-of-domain "
                     "$h_{50}$ mapping", y=.985)
    fig.savefig(out / f"figS3_composite_sensitivity.{ext}")
    plt.close(fig)
    print(f"    -> figS3_composite_sensitivity.{ext}")


def figS4(D, out, ext):
    und = D.und
    props = [p for p in D.props if f"p_{p}" in und.columns]
    if not props:
        return note("S4: no p-value columns")
    show = props[:6]
    nrow = int(np.ceil(len(show) / 3))
    fig, axes = plt.subplots(nrow, 3, figsize=(7.2, 2.30 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    fig.subplots_adjust(left=.085, right=.98, top=.80, bottom=.20,
                        hspace=.58, wspace=.16)
    for ax, p in zip(axes, show):
        fs.clean(ax)
        z = pd.to_numeric(und[f"z_{p}"], errors="coerce")
        y = -np.log10(pd.to_numeric(und[f"p_{p}"], errors="coerce")
                      .clip(lower=1e-300))
        sg = D.sig(und, p)
        ax.scatter(z[~sg], y[~sg], s=2.6, c=fs.HAIRLINE, lw=0, zorder=2)
        for a in D.axes_present():
            m = sg & (und.axis == a).values
            ax.scatter(z[m], y[m], s=4.2, c=fs.AXIS_COLOR[a], lw=0, alpha=.8,
                       zorder=3, label=fs.AXIS_LABEL[a])
        ax.axvline(0, color=fs.INK, lw=.7, zorder=4)
        fs.title(ax, fs.ptex(p))
        ax.set_xlabel("directed $z$")
    for ax in axes[len(show):]:
        ax.set_visible(False)
    axes[0].set_ylabel("$-\\log_{10} p$")
    h = [Line2D([], [], marker="o", ls="", ms=4, color=fs.AXIS_COLOR[a],
                label=fs.AXIS_LABEL[a]) for a in D.axes_present()]
    h.append(Line2D([], [], marker="o", ls="", ms=4, color=fs.HAIRLINE,
                    label="not significant"))
    fig.legend(handles=h, ncol=5, loc="lower center", bbox_to_anchor=(.5, .02))
    fs.figtitle(fig, "Transformation catalogue by axis",
                "mirror rows removed; significance from global Benjamini–Hochberg")
    fig.savefig(out / f"figS4_volcano.{ext}")
    plt.close(fig)
    print(f"    -> figS4_volcano.{ext}")


def figS5(D, out, ext):
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
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 1.90 * nrow))
    axes = np.atleast_1d(axes).ravel()
    fig.subplots_adjust(left=.085, right=.98, top=.78, bottom=.16,
                        hspace=.66, wspace=.34)
    for ax, p in zip(axes, props):
        fs.clean(ax)
        s = r[r.property == p]
        ax.scatter(s.centered, s.mean_delta, s=8, c=fs.INK_SOFT, alpha=.5,
                   lw=0, zorder=3)
        if len(s) > 2 and s.centered.std() > 0:
            b, a = np.polyfit(s.centered, s.mean_delta, 1)
            xs = np.linspace(s.centered.min(), s.centered.max(), 20)
            ax.plot(xs, a + b * xs, color=fs.ACCENT_POS, lw=1.3, zorder=4)
            ax.text(.05, .07, f"slope {b:+.2f}", transform=ax.transAxes,
                    fontsize=6.4, color=fs.ACCENT_POS)
        ax.axhline(0, color=fs.MUTED, lw=.55, ls=(0, (2, 2)), zorder=1)
        ax.axvline(0, color=fs.MUTED, lw=.55, ls=(0, (2, 2)), zorder=1)
        fs.title(ax, fs.ptex(p))
        ax.tick_params(labelsize=6.2)
    for ax in axes[len(props):]:
        ax.set_visible(False)
    fig.supxlabel("candidate − database mean", fontsize=7.4, color=fs.INK_SOFT)
    fig.supylabel("mean Δ from edits", fontsize=7.4, color=fs.INK_SOFT)
    fs.figtitle(fig, "Regression to the mean",
                "predicted gains shrink with distance from the pool")
    fig.savefig(out / f"figS5_regression_to_mean.{ext}")
    plt.close(fig)
    print(f"    -> figS5_regression_to_mean.{ext}")


FIGS = {"figA": figA, "figB": figB, "S1": figS1, "S2": figS2, "S3": figS3,
        "S4": figS4, "S5": figS5}


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
    print(f"Results: {args.results}\nProperties: {D.props}")
    if len(D.tr):
        u = D.und
        print("Rules per axis: " + ", ".join(
            f"{a}={len(u[u.axis==a]):,}" for a in D.axes_present()))
    for name in (args.only or list(FIGS)):
        print(f"  {name}")
        try:
            FIGS[name](D, out, args.format)
        except Exception as e:
            note(f"{name}: {type(e).__name__}: {e}")
    print(f"\nFigures in {out}/")


if __name__ == "__main__":
    main()
