#!/usr/bin/env python3
"""
12_figures.py -- journal-quality v3 MMPA figure set.

Visual language: muted ink, Okabe–Ito accents, thin spines, panel letters,
no overlapping annotations. Scientific content matches the transformation-first
narrative (families for overview; exact A→B where chemistry matters).

Usage:
  <tartarus python> 12_figures.py
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Patch, FancyBboxPatch
from scipy.cluster.hierarchy import linkage, leaves_list

import config
import mmpa_lib as ml

# ---------------------------------------------------------------------------
# Journal style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "semibold",
    "axes.labelsize": 8.5,
    "axes.labelcolor": "#222222",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

INK = "#1a1a1a"
MUTE = "#6b6b6b"
RULE = "#d0d0d0"
PAPER = "#ffffff"
PANEL = "#f7f7f5"

# Okabe–Ito inspired (colorblind-safe)
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_VERM = "#D55E00"
C_PURPLE = "#CC79A7"
C_SKY = "#56B4E9"
C_GREY = "#B8B8B8"
C_INK = "#111111"

STATE_ORDER = ["neither", "actionable", "mixed", "already"]
STATE_COLORS = {
    "neither": "#E8E8E8",
    "actionable": C_ORANGE,
    "mixed": C_PURPLE,
    "already": C_BLUE,
}
STATE_CODE = {s: i for i, s in enumerate(STATE_ORDER)}
COL_EMBODIED = tuple(int(C_BLUE[i:i+2], 16) / 255 for i in (1, 3, 5))
COL_ACTIONABLE = tuple(int(C_ORANGE[i:i+2], 16) / 255 for i in (1, 3, 5))

# Soft diverging cmap (no neon)
_DIVERGE_COLORS = ["#2166AC", "#67A9CF", "#D1E5F0", "#F7F7F7",
                   "#FDDBC7", "#EF8A62", "#B2182B"]
DIVERGE = LinearSegmentedColormap.from_list("mmp_div", _DIVERGE_COLORS, N=256)
SEQ = LinearSegmentedColormap.from_list(
    "mmp_seq", ["#FFF7EC", "#FDD49E", "#FC8D59", "#D7301F", "#7F0000"], N=256)

CASE_STUDIES = [
    ("cand_007", "Mature star"),
    ("cand_001", "Mature near-lead"),
    ("cand_022", "Mature ≠ win"),
    ("cand_062", "Mixed / large gap"),
    ("cand_038", "Opportunity-rich"),
    ("cand_023", "Scope limit"),
]

PREFERRED_TRANSITIONS = [
    "azole_ring -> azide", "azole_ring -> nitro", "azole_ring -> nitramine",
    "azole_ring -> n_oxide", "azole_ring -> azole_ring",
    "nitramine -> azide", "nitro -> azide", "azide -> azide",
    "nitramine -> nitro", "nitro -> nitramine", "n_oxide -> azide", "azide -> nitro",
]


def style_ax(ax, grid=False):
    ax.set_facecolor(PAPER)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=3, width=0.6, pad=2)
    if grid:
        ax.grid(True, axis="y", color=RULE, lw=0.5, zorder=0)
        ax.set_axisbelow(True)


def panel_label(ax, letter, x=-0.12, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left", color=INK,
            clip_on=False)


def short(s, n=28):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def fraglabel(s):
    s = str(s)
    for a, b in [
        ("[*:1][N+](=O)[O-]", "NO2"), ("[N+](=O)[O-]", "NO2"),
        ("N=[N+]=[N-]", "N3"), ("[*:1][H]", "H"), ("[*:1]O", "OR"), ("[*:1]", ""),
    ]:
        s = s.replace(a, b)
    return short(s.strip(" .|"), 26)


def fam_arrow(tr):
    return str(tr).replace(" -> ", " → ")


def fam_compact(tr):
    """Short family label for dense axis ticks."""
    s = fam_arrow(tr)
    return (s.replace("azole_ring", "azole")
             .replace("nitramine", "NNO₂")
             .replace("azoxy", "N=N⁺O⁻")
             .replace("n_oxide", "N-ox")
             .replace("azide", "N₃")
             .replace("nitro", "NO₂"))


def _read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class Data:
    def __init__(self, results: Path, master_path: Path, props, direction):
        self.d = direction
        self.results = results

        self.master = pd.read_csv(master_path)
        self.master[config.COL_ID] = self.master[config.COL_ID].astype(str)

        self.fg = pd.read_csv(results / "fragment_goodness.csv")
        self.fg["fragment"] = self.fg["fragment"].astype(str)
        gs = pd.read_csv(results / "global_scales.csv").set_index("property")
        self.props = [
            p for p in props
            if f"A_{p}" in self.fg.columns and p in gs.index and p in self.master.columns
        ]
        missing = [p for p in props if p not in self.props]
        if missing:
            print(f"  WARNING: properties skipped (missing from inputs): {missing}")
        if not self.props:
            raise SystemExit("ERROR: no usable properties for figures")
        props = self.props
        self.s_global = {
            p: (float(gs.loc[p, "std"]) if float(gs.loc[p, "std"]) > 0 else 1.0)
            for p in props
        }
        self.scal = pd.read_csv(results / "candidate_scalars.csv")
        self.scal["candidate_id"] = self.scal["candidate_id"].astype(str)
        self.cmp = pd.read_csv(results / "corr_compare.csv")
        self.stat = pd.read_csv(results / "static_corr.csv")
        self.rtm = pd.read_csv(results / "regression_to_mean.csv")
        self.rtm["candidate_id"] = self.rtm["candidate_id"].astype(str)

        print("  loading transformations…")
        usecols = [
            "frag_from", "frag_to", "chemotype_from", "chemotype_to",
            "n_pairs", "composite_gated", "composite_z", "n_improve",
            "n_worsen", "has_tradeoff",
        ] + [f"z_{p}" for p in props] + [f"sig_{p}" for p in props]
        self.trans = pd.read_csv(results / "transformations.csv", usecols=usecols)
        self.matrix = pd.read_csv(results / "candidate_rule_matrix.csv")
        self.matrix["candidate_id"] = self.matrix["candidate_id"].astype(str)
        self.rec = _read_csv(results / "recommendations.csv")
        if len(self.rec):
            self.rec["candidate_id"] = self.rec["candidate_id"].astype(str)
        self.ctx = _read_csv(results / "context_conditional_rules.csv")
        self.head = pd.read_csv(results / "aggregated" / "headline.csv")
        self.head["candidate_id"] = self.head["candidate_id"].astype(str)

        self.candidate_ids = [
            l.strip() for l in open(config.CANDIDATES_TXT) if l.strip()
        ]
        self.smi = {}
        with open(config.SMI_FILE) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.smi[parts[1]] = parts[0]

        ref, _, desc = ml.load_benchmark(
            self.master, props, direction, config.BENCHMARK
        )
        self.ref = ref
        self.bench_desc = desc
        self.clean_rules = self._clean_rules()
        self.tr_summary = self._transition_summary()
        self.state_table, self.profile = self._candidate_states()

    def _clean_rules(self):
        t = self.trans
        return t[(t["n_pairs"] >= 20) & (t["n_worsen"] == 0) &
                 (t["composite_gated"] > 0)].copy()

    def _transition_summary(self):
        t = self.clean_rules.copy()
        t["tr"] = t["chemotype_from"] + " -> " + t["chemotype_to"]
        rows = []
        for tr, g in t.groupby("tr"):
            rows.append({
                "tr": tr,
                "n_rules": len(g),
                "med_n": g["n_pairs"].median(),
                "max_n": g["n_pairs"].max(),
                "med_gated": g["composite_gated"].median(),
                "max_gated": g["composite_gated"].max(),
                "med_improve": g["n_improve"].median(),
                **{f"z_{p}": g[f"z_{p}"].mean() for p in self.props},
                **{f"sigfrac_{p}": (g[f"sig_{p}"].astype(str).str.lower()
                                    .isin(["true", "1", "1.0"])).mean()
                   for p in self.props},
            })
        return pd.DataFrame(rows).sort_values("max_gated", ascending=False)

    def _candidate_states(self):
        m = self.matrix
        f = m[(m["n_pairs"] >= 20) & (m["composite_gated"] >= 5) &
              (m["n_worsen"] == 0)].copy()
        f["tr"] = f["chemotype_from"] + " -> " + f["chemotype_to"]
        f["rule"] = f["frag_from"].astype(str) + ">>" + f["frag_to"].astype(str)

        states = []
        for (cid, tr), g in f.groupby(["candidate_id", "tr"]):
            roles = set(g["role"])
            if roles == {"actionable", "already_optimized"}:
                st = "mixed"
            elif roles == {"actionable"}:
                st = "actionable"
            elif roles == {"already_optimized"}:
                st = "already"
            else:
                st = "neither"
            score = g.groupby("rule")["composite_gated"].max()
            role_by_rule = g.groupby("rule")["role"].apply(set)
            opp = float(sum(score[r] for r in score.index
                            if "actionable" in role_by_rule[r]))
            mat = float(sum(score[r] for r in score.index
                            if "already_optimized" in role_by_rule[r]))
            states.append({"candidate_id": cid, "tr": tr, "state": st,
                           "opp": opp, "mat": mat})
        st = pd.DataFrame(states)

        top_tr = [t for t in PREFERRED_TRANSITIONS if t in set(st["tr"])]
        extras = [t for t in self.tr_summary["tr"] if t not in top_tr]
        top_tr = (top_tr + extras)[:10]

        rows = []
        for cid in self.candidate_ids:
            for tr in top_tr:
                hit = st[(st.candidate_id == cid) & (st.tr == tr)]
                if len(hit):
                    rows.append(hit.iloc[0].to_dict())
                else:
                    rows.append({"candidate_id": cid, "tr": tr, "state": "neither",
                                 "opp": 0.0, "mat": 0.0})
        grid = pd.DataFrame(rows)
        prof = (st.groupby("candidate_id")
                .agg(opp=("opp", "sum"), mat=("mat", "sum"))
                .reindex(self.candidate_ids).fillna(0).reset_index())
        prof = prof.merge(self.scal, on="candidate_id", how="left")
        paired = set()
        ap = self.results / "anchored_pairs.csv"
        if ap.exists():
            paired = set(pd.read_csv(ap, usecols=["candidate_id"])
                         ["candidate_id"].astype(str))
        prof["paired"] = prof["candidate_id"].isin(paired)
        return grid, prof

    def cand_props(self, cid):
        row = self.scal[self.scal.candidate_id == cid]
        vals = {}
        if len(row):
            r = row.iloc[0]
            for p in self.props:
                col = f"val_{p}"
                if col in r.index and pd.notna(r[col]):
                    vals[p] = float(r[col])
        if len(vals) == len(self.props):
            return vals
        smiles = self.smi.get(cid)
        if smiles and config.COL_SMILES in self.master.columns:
            hit = self.master[self.master[config.COL_SMILES] == smiles]
            if len(hit):
                return {p: float(hit.iloc[0][p]) for p in self.props}
        return vals


# ---------------------------------------------------------------------------
# Chemistry drawing
# ---------------------------------------------------------------------------
def _frag_mol_for_draw(frag_smiles):
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    s = str(frag_smiles)
    for tok in ("[*:1]", "[*:2]", "[*:3]", "[*]", "*"):
        s = s.replace(tok, "[H]")
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        mol = Chem.MolFromSmiles(ml.strip_attachment(frag_smiles) or "C")
    return mol


def _frag_query_mol(frag_smiles):
    from rdkit import Chem
    s = ml.strip_attachment(frag_smiles)
    if not s:
        return None
    if s in {"[H]", "H"}:
        return Chem.MolFromSmarts("[H]")
    mol = Chem.MolFromSmiles(s)
    if mol is not None:
        return mol
    s2 = str(frag_smiles).replace("[*:1]", "[*]").replace("[*:2]", "[*]")
    s2 = s2.replace("[*:3]", "[*]").replace("*", "[*]")
    return Chem.MolFromSmiles(s2)


def _match_frag_atoms(mol, frag_smiles, used=None):
    from rdkit import Chem
    if mol is None:
        return ()
    q = _frag_query_mol(frag_smiles)
    if q is None:
        return ()
    hits = list(mol.GetSubstructMatches(q))
    if not hits:
        s = ml.strip_attachment(frag_smiles)
        q2 = Chem.MolFromSmarts(s) if s else None
        if q2 is not None:
            hits = list(mol.GetSubstructMatches(q2))
    if not hits:
        return ()
    used = used or set()
    for hit in hits:
        if not used.intersection(hit):
            return hit
    return hits[0]


def _bonds_within(mol, atoms):
    aset = set(atoms)
    return [b.GetIdx() for b in mol.GetBonds()
            if b.GetBeginAtomIdx() in aset and b.GetEndAtomIdx() in aset]


def _pil_mol(mol, w=200, h=140, highlight_atoms=None, highlight_color=None):
    from rdkit.Chem.Draw import rdMolDraw2D
    from PIL import Image
    if mol is None:
        return Image.new("RGB", (w, h), "white")
    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = False
    opts.atomHighlightsAreCircles = True
    opts.highlightBondWidthMultiplier = 12
    opts.clearBackground = True
    atom_cols, bond_cols = {}, {}
    if highlight_atoms and highlight_color:
        for a in highlight_atoms:
            atom_cols[a] = highlight_color
        aset = set(highlight_atoms)
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in aset and j in aset:
                bond_cols[b.GetIdx()] = highlight_color
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atom_cols.keys()),
        highlightAtomColors=atom_cols,
        highlightBonds=list(bond_cols.keys()),
        highlightBondColors=bond_cols,
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def _draw_mol(ax, smiles, highlight_atoms=None, title=""):
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit import RDLogger
    from PIL import Image
    RDLogger.DisableLog("rdApp.*")
    ax.set_axis_off()
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        ax.text(0.5, 0.5, "parse failed", ha="center", color=MUTE)
        return
    atom_cols = dict(highlight_atoms or {})
    bond_cols = {}
    for color in {atom_cols[a] for a in atom_cols}:
        atoms = [a for a, c in atom_cols.items() if c == color]
        for bi in _bonds_within(mol, atoms):
            bond_cols[bi] = color
    drawer = rdMolDraw2D.MolDraw2DCairo(380, 260)
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = False
    opts.atomHighlightsAreCircles = True
    opts.highlightBondWidthMultiplier = 14
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atom_cols.keys()),
        highlightAtomColors=atom_cols,
        highlightBonds=list(bond_cols.keys()),
        highlightBondColors=bond_cols,
    )
    drawer.FinishDrawing()
    ax.imshow(Image.open(io.BytesIO(drawer.GetDrawingText())))
    if title:
        ax.set_title(title, fontsize=8, fontweight="semibold", pad=3, color=INK)


def _pil_font(size=11):
    from PIL import ImageFont
    for path in (
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_exact_transform(ax, frag_from, frag_to, caption_lines, w=520, h=200):
    from PIL import Image, ImageDraw
    ax.set_axis_off()
    a = _frag_mol_for_draw(frag_from)
    b = _frag_mol_for_draw(frag_to)
    mol_h = 128
    ia = _pil_mol(a, 210, mol_h,
                  list(range(a.GetNumAtoms())) if a else None, COL_ACTIONABLE)
    ib = _pil_mol(b, 210, mol_h,
                  list(range(b.GetNumAtoms())) if b else None, COL_EMBODIED)
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(ia, (6, 6))
    canvas.paste(ib, (w - 216, 6))
    draw = ImageDraw.Draw(canvas)
    font_ab = _pil_font(13)
    font_cap = _pil_font(11)
    font_smi = _pil_font(9)
    y0 = 68
    draw.line([(228, y0), (w - 228, y0)], fill=(50, 50, 50), width=2)
    draw.polygon([(w - 228, y0), (w - 238, y0 - 6), (w - 238, y0 + 6)],
                 fill=(50, 50, 50))
    draw.text((12, 4), "A", fill=(180, 120, 0), font=font_ab)
    draw.text((w - 28, 4), "B", fill=(0, 90, 150), font=font_ab)
    band_y = mol_h + 10
    draw.rectangle([(0, band_y), (w, h)], fill=(247, 247, 245))
    draw.line([(0, band_y), (w, band_y)], fill=(210, 210, 210), width=1)
    y = band_y + 6
    for i, line in enumerate(caption_lines[:3]):
        fnt = font_cap if i == 0 else font_smi
        draw.text((8, y), short(line, 92), fill=(35, 35, 35), font=fnt)
        y += 14 if i == 0 else 12
    ax.imshow(canvas)


def _top_exact_rules_by_family(D, n_families=6):
    ts = D.tr_summary.copy()
    ts = ts[ts["n_rules"] >= 20] if (ts["n_rules"] >= 20).any() else ts
    top_fam = ts.nlargest(n_families, "max_gated")["tr"].tolist()
    clean = D.clean_rules.copy()
    clean["tr"] = clean["chemotype_from"] + " -> " + clean["chemotype_to"]
    rows = []
    for fam in top_fam:
        sub = clean[clean["tr"] == fam]
        if not len(sub):
            continue
        rows.append(sub.sort_values(["composite_gated", "n_pairs"],
                                    ascending=False).iloc[0])
    return rows


def _case_exact_rules(D, cid):
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = D.matrix[
        (D.matrix.candidate_id == cid)
        & (D.matrix.n_pairs >= 10)
        & (D.matrix.composite_gated > 0)
        & (D.matrix.n_worsen == 0)
    ].copy()
    m["tr"] = m.chemotype_from + " -> " + m.chemotype_to
    smiles = D.smi.get(cid, "")
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    atom_cols, used = {}, set()
    embodied_rule = actionable_rule = None

    emb = m[m.role == "already_optimized"].sort_values(
        "composite_gated", ascending=False)
    for _, r in emb.iterrows():
        hit = _match_frag_atoms(mol, r.frag_to, used=set())
        if hit:
            embodied_rule = dict(
                tr=r.tr, score=float(r.composite_gated), n_pairs=int(r.n_pairs),
                frag_from=r.frag_from, frag_to=r.frag_to,
                frag_label=fraglabel(r.frag_to),
                n_improve=int(r.n_improve), n_worsen=int(r.n_worsen),
                highlighted=True)
            for a in hit:
                atom_cols[a] = COL_EMBODIED
                used.add(a)
            break

    act = m[m.role == "actionable"].sort_values(
        "composite_gated", ascending=False)
    for _, r in act.iterrows():
        hit = _match_frag_atoms(mol, r.frag_from, used=used)
        if not hit or used.intersection(hit):
            continue
        actionable_rule = dict(
            tr=r.tr, score=float(r.composite_gated), n_pairs=int(r.n_pairs),
            frag_from=r.frag_from, frag_to=r.frag_to,
            frag_label=fraglabel(r.frag_from),
            n_improve=int(r.n_improve), n_worsen=int(r.n_worsen),
            highlighted=True, same_site=False)
        for a in hit:
            atom_cols[a] = COL_ACTIONABLE
            used.add(a)
        break
    if actionable_rule is None and len(act):
        r = act.iloc[0]
        hit = _match_frag_atoms(mol, r.frag_from, used=set())
        actionable_rule = dict(
            tr=r.tr, score=float(r.composite_gated), n_pairs=int(r.n_pairs),
            frag_from=r.frag_from, frag_to=r.frag_to,
            frag_label=fraglabel(r.frag_from),
            n_improve=int(r.n_improve), n_worsen=int(r.n_worsen),
            same_site=bool(hit and used.intersection(hit)),
            highlighted=False)
    return embodied_rule, actionable_rule, atom_cols


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------
def figure1(D, out):
    props = D.props
    n = len(props)
    M = np.full((n, n), np.nan)
    idx = {p: i for i, p in enumerate(props)}
    for _, r in D.stat.iterrows():
        if r["p1"] in idx and r["p2"] in idx:
            M[idx[r["p2"]], idx[r["p1"]]] = r["pearson"]
    for _, r in D.cmp.iterrows():
        if r["p1"] in idx and r["p2"] in idx:
            M[idx[r["p1"]], idx[r["p2"]]] = r["delta"]

    fig = plt.figure(figsize=(10.8, 4.6), facecolor=PAPER)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.38)

    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax)
    im = ax.imshow(M, cmap=DIVERGE, vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n)); ax.set_xticklabels(props)
    ax.set_yticks(range(n)); ax.set_yticklabels(props)
    for i in range(n):
        ax.text(i, i, props[i], ha="center", va="center", fontsize=8,
                fontweight="bold", color=INK)
        for j in range(n):
            if i == j or np.isnan(M[i, j]):
                continue
            v = M[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.8,
                    color="white" if abs(v) > 0.55 else INK)
    ax.set_title("Property coupling: population vs edits", pad=8)
    ax.text(0.0, -0.16, "Lower triangle: static (108k)   ·   Upper: Δ–Δ from edits",
            transform=ax.transAxes, fontsize=7, color=MUTE)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Pearson r", fontsize=8)
    cb.outline.set_linewidth(0.5)
    panel_label(ax, "a")

    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, grid=True)
    dd = D.cmp.dropna(subset=["divergence"]).copy()
    dd["pair"] = dd["p1"] + "–" + dd["p2"]
    dd = dd.reindex(dd["divergence"].abs().sort_values().index)
    colors = [C_VERM if v > 0 else C_BLUE for v in dd["divergence"]]
    ax2.barh(range(len(dd)), dd["divergence"], color=colors, height=0.72,
             edgecolor="none", zorder=2)
    ax2.set_yticks(range(len(dd)))
    ax2.set_yticklabels(dd["pair"], fontsize=7.5)
    ax2.axvline(0, color=INK, lw=0.7, zorder=3)
    ax2.set_xlabel("Divergence  (edit corr − static corr)")
    ax2.set_title("Where edits break population coupling", pad=8)
    panel_label(ax2, "b", x=-0.18)

    fig.savefig(out / "fig1_coupling.png"); plt.close(fig)
    print("  fig1_coupling.png")


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------
def figure2(D, out):
    props = D.props
    ts = D.tr_summary.copy()
    ts = ts[ts["n_rules"] >= 20].copy() if (D.tr_summary["n_rules"] >= 20).any() \
        else D.tr_summary.copy()
    exemplars = _top_exact_rules_by_family(D, 6)

    fig = plt.figure(figsize=(11.5, 10.4), facecolor=PAPER)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.55, 0.55],
                          hspace=0.55, wspace=0.35)

    # (a) — no on-plot text labels (avoid collisions); legend maps rank → family
    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax)
    x = ts["max_gated"].values
    y = np.log10(ts["n_rules"].astype(float))
    size = 35 + 160 * (ts["med_n"] / max(ts["med_n"].max(), 1))
    sc = ax.scatter(x, y, s=size, c=ts["med_improve"], cmap=SEQ, vmin=2, vmax=7,
                    edgecolor="white", linewidth=0.6, alpha=0.95, zorder=3)
    top = ts.nlargest(5, "max_gated").reset_index(drop=True)
    # Offset rank badges so marker color remains visible
    badge_off = [(10, 8), (-10, 8), (10, -10), (-10, -10), (12, 6)]
    for k, r in top.iterrows():
        ax.scatter([r["max_gated"]], [np.log10(r["n_rules"])],
                   s=55, facecolors="none", edgecolors=INK, linewidths=1.2, zorder=4)
        dx, dy = badge_off[k % len(badge_off)]
        ax.annotate(
            str(k + 1),
            (r["max_gated"], np.log10(r["n_rules"])),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=7, fontweight="bold", ha="center", va="center",
            color=INK, zorder=5,
            bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec=INK, lw=0.7),
            arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.5),
        )

    leg_lines = [f"{k+1}. {fam_arrow(r.tr)}" for k, r in top.iterrows()]
    ax.text(0.02, 0.02, "Top families\n" + "\n".join(leg_lines),
            transform=ax.transAxes, va="bottom", ha="left", fontsize=6.5,
            color=INK, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=RULE, lw=0.6))

    ax.set_xlabel("Max gated composite (Σ sig. z, SD units)")
    ax.set_ylabel("log₁₀ distinct clean exact rules")
    ax.set_title("Family-level landscape", pad=8)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("# props improved (median)", fontsize=7.5)
    cb.outline.set_linewidth(0.4)
    panel_label(ax, "a")

    # (b)
    axb = fig.add_subplot(gs[0, 1])
    style_ax(axb)
    top6 = ts.nlargest(6, "max_gated")
    Z = top6[[f"z_{p}" for p in props]].values.astype(float)
    vmax = max(np.nanmax(np.abs(Z)), 0.5)
    im = axb.imshow(Z, aspect="auto", cmap=DIVERGE, vmin=-vmax, vmax=vmax)
    axb.set_xticks(range(len(props))); axb.set_xticklabels(props, fontsize=8)
    axb.set_yticks(range(len(top6)))
    axb.set_yticklabels([fam_arrow(t) for t in top6["tr"]], fontsize=7)
    for i, (_, r) in enumerate(top6.iterrows()):
        for j, p in enumerate(props):
            if r[f"sigfrac_{p}"] >= 0.5:
                axb.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False,
                    edgecolor=INK, lw=0.9))
    axb.set_title("Family mean directed z  (box = sig ≥50%)", pad=8)
    cb = fig.colorbar(im, ax=axb, fraction=0.046, pad=0.02)
    cb.set_label("mean z", fontsize=7.5)
    cb.outline.set_linewidth(0.4)
    panel_label(axb, "b", x=-0.22)

    # (c) exemplars — title as a dedicated strip above the grid
    fig.text(0.5, 0.565,
             "c   Exact A→B exemplars (best clean rule in each top family; orange=A, blue=B)",
             ha="center", fontsize=9, fontweight="semibold", color=INK)
    gs_ex = gs[1, :].subgridspec(2, 3, wspace=0.18, hspace=0.28)
    for i, r in enumerate(exemplars[:6]):
        ax_e = fig.add_subplot(gs_ex[i // 3, i % 3])
        fam = fam_arrow(f"{r['chemotype_from']} -> {r['chemotype_to']}")
        cap = [
            f"{fam}  ·  gated {r['composite_gated']:+.2f} SD  ·  n={int(r['n_pairs'])}",
            f"A  {fraglabel(r['frag_from'])}",
            f"B  {fraglabel(r['frag_to'])}",
        ]
        _draw_exact_transform(ax_e, r["frag_from"], r["frag_to"], cap, w=500, h=200)


    # (d) context
    axc = fig.add_subplot(gs[2, :])
    style_ax(axc, grid=True)
    if len(D.ctx) and "n_props_sign_flip_vs_global" in D.ctx.columns:
        flips = D.ctx["n_props_sign_flip_vs_global"].fillna(0).astype(int)
        counts = flips.value_counts().reindex(range(0, 7), fill_value=0)
        cols = [C_BLUE if i == 0 else (C_VERM if i >= 3 else C_SKY)
                for i in counts.index]
        axc.bar(counts.index, counts.values, color=cols, width=0.7,
                edgecolor="none", zorder=2)
        axc.set_xticks(range(0, 7))
        axc.set_xlabel("# properties with sign flip vs global rule")
        axc.set_ylabel("# environments")
        axc.set_title("Context reliability at radius 2", pad=6)
        axc.text(
            0.99, 0.92,
            f"n = {len(flips):,}   ·   "
            f"{100*(flips==0).mean():.0f}% stable   ·   "
            f"{100*(flips>=3).mean():.1f}% flip ≥3",
            transform=axc.transAxes, ha="right", va="top", fontsize=7.5,
            color=INK,
            bbox=dict(boxstyle="round,pad=0.35", fc=PANEL, ec=RULE, lw=0.6))
    panel_label(axc, "d", x=-0.05)

    fig.savefig(out / "fig2_transformations.png"); plt.close(fig)
    print("  fig2_transformations.png")


# ---------------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------------
def figure3(D, out):
    grid = D.state_table
    prof = D.profile
    transitions = list(dict.fromkeys(grid["tr"].tolist()))
    cids = D.candidate_ids
    mat = np.full((len(cids), len(transitions)), STATE_CODE["neither"], int)
    ci = {c: i for i, c in enumerate(cids)}
    ti = {t: j for j, t in enumerate(transitions)}
    for _, r in grid.iterrows():
        i, j = ci.get(r.candidate_id), ti.get(r.tr)
        if i is not None and j is not None:
            mat[i, j] = STATE_CODE[r.state]
    try:
        order = leaves_list(linkage(mat.astype(float), method="ward"))
    except Exception:
        order = np.arange(len(cids))
    mat_o = mat[order]
    cids_o = [cids[i] for i in order]
    case_ids = {c for c, _ in CASE_STUDIES}

    fig = plt.figure(figsize=(12.0, 10.0), facecolor=PAPER)
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.55, 0.38, 1.05],
        width_ratios=[1.45, 1.0],
        hspace=0.28, wspace=0.30,
    )

    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax)
    cmap = ListedColormap([STATE_COLORS[s] for s in STATE_ORDER])
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), cmap.N)
    ax.imshow(mat_o, aspect="auto", cmap=cmap, norm=norm,
              interpolation="nearest")
    # Numbered columns + dedicated key row (avoids overlapping long names)
    ax.set_xticks(range(len(transitions)))
    ax.set_xticklabels([str(i + 1) for i in range(len(transitions))], fontsize=8)
    ax.set_xlabel("Transformation family (numbered; key below)", fontsize=7.5)
    yticks, ylabels = [], []
    for i, c in enumerate(cids_o):
        if c in case_ids:
            yticks.append(i); ylabels.append(c.replace("cand_", "★"))
        elif i % 5 == 0:
            yticks.append(i); ylabels.append(c.replace("cand_", ""))
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=6.5)
    for i, c in enumerate(cids_o):
        if c in case_ids:
            ax.axhline(i, color=INK, lw=0.5, alpha=0.35)
    ax.set_title("Candidate × family state  (cell ≥ 1 exact rule)", pad=8)
    handles = [Patch(color=STATE_COLORS[s], label=s) for s in STATE_ORDER]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.08),
              ncol=4, fontsize=7, handlelength=1.0, columnspacing=1.0)
    panel_label(ax, "a")

    ax_key = fig.add_subplot(gs[1, 0])
    ax_key.set_axis_off()
    key_lines = [f"{i+1}  {fam_compact(t)}" for i, t in enumerate(transitions)]
    mid = (len(key_lines) + 1) // 2
    ax_key.text(0.0, 0.95, "Family key", transform=ax_key.transAxes,
                fontsize=7.5, fontweight="semibold", color=INK, va="top")
    ax_key.text(0.0, 0.55, "\n".join(key_lines[:mid]),
                transform=ax_key.transAxes, va="top", ha="left", fontsize=6.5,
                color=INK, linespacing=1.35)
    ax_key.text(0.52, 0.55, "\n".join(key_lines[mid:]),
                transform=ax_key.transAxes, va="top", ha="left", fontsize=6.5,
                color=INK, linespacing=1.35)

    axb = fig.add_subplot(gs[0, 1])
    style_ax(axb)
    p = prof.copy()
    sc = axb.scatter(np.log1p(p.mat), np.log1p(p.opp),
                     c=p.composite_vs_benchmark.fillna(0), cmap=DIVERGE,
                     vmin=-6, vmax=3, s=np.where(p.paired, 42, 70),
                     edgecolor="white", linewidth=0.5, alpha=0.95, zorder=3)
    unp = p[~p.paired]
    if len(unp):
        axb.scatter(np.log1p(unp.mat), np.log1p(unp.opp), s=64,
                    facecolors="none", edgecolors=INK, linewidths=1.1,
                    marker="s", zorder=4)
    offsets = {
        "cand_007": (-58, 22), "cand_001": (12, 18), "cand_022": (12, -22),
        "cand_062": (10, 8), "cand_038": (10, 12), "cand_023": (12, 14),
    }
    short_lab = {
        "cand_007": "007 mature★", "cand_001": "001 near-lead",
        "cand_022": "022 mature≠win", "cand_062": "062 mixed",
        "cand_038": "038 opp-rich", "cand_023": "023 scope",
    }
    for cid, role in CASE_STUDIES:
        hit = p[p.candidate_id == cid]
        if not len(hit):
            continue
        r = hit.iloc[0]
        ox, oy = offsets.get(cid, (5, 5))
        axb.annotate(
            short_lab.get(cid, cid),
            (np.log1p(r.mat), np.log1p(r.opp)),
            fontsize=6.5, fontweight="semibold", color=INK,
            xytext=(ox, oy), textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.55),
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=RULE, lw=0.5),
            zorder=5,
        )
    axb.set_xlabel("Embodied benefit  log(1+Σ gated of B)")
    axb.set_ylabel("Actionable opportunity  log(1+Σ gated of A)")
    axb.set_title("Maturity vs remaining opportunity", pad=8)
    cb = fig.colorbar(sc, ax=axb, fraction=0.046, pad=0.02)
    cb.set_label("vs top-10 benchmark", fontsize=7.5)
    cb.outline.set_linewidth(0.4)
    panel_label(axb, "b", x=-0.16)

    # (c) callouts — compact two-column text
    axc = fig.add_subplot(gs[2, :])
    axc.set_axis_off()
    m = D.matrix[(D.matrix.n_pairs >= 10) & (D.matrix.composite_gated > 0) &
                 (D.matrix.n_worsen == 0)].copy()
    m["tr"] = m.chemotype_from + " -> " + m.chemotype_to
    left, right = [], []
    for i, (cid, role) in enumerate(CASE_STUDIES):
        bucket = left if i < 3 else right
        sub = m[m.candidate_id == cid]
        bucket.append(f"{cid}  ·  {role}")
        if not len(sub):
            bucket.append("    no clean exact rules")
            bucket.append("")
            continue
        emb = sub[sub.role == "already_optimized"].sort_values(
            "composite_gated", ascending=False)
        act = sub[sub.role == "actionable"].sort_values(
            "composite_gated", ascending=False)
        if len(emb):
            r = emb.iloc[0]
            bucket.append(
                f"    B  {fam_arrow(r.tr)}   gated={r.composite_gated:+.1f} SD  "
                f"n={int(r.n_pairs)}   {fraglabel(r.frag_to)}")
        if len(act):
            r = act.iloc[0]
            bucket.append(
                f"    A  {fam_arrow(r.tr)}   gated={r.composite_gated:+.1f} SD  "
                f"n={int(r.n_pairs)}")
            bucket.append(
                f"       {fraglabel(r.frag_from)}  →  {fraglabel(r.frag_to)}")
        bucket.append("")
    axc.text(0.01, 0.98, "\n".join(left), transform=axc.transAxes,
             va="top", ha="left", fontsize=7, family="DejaVu Sans Mono",
             color=INK, linespacing=1.35)
    axc.text(0.52, 0.98, "\n".join(right), transform=axc.transAxes,
             va="top", ha="left", fontsize=7, family="DejaVu Sans Mono",
             color=INK, linespacing=1.35)
    axc.add_patch(FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0, transform=axc.transAxes,
        boxstyle="round,pad=0.01,rounding_size=0.01",
        facecolor=PANEL, edgecolor=RULE, lw=0.7, zorder=-1,
        mutation_aspect=0.1))
    axc.set_title("Exact A/B fragments behind case-study cells",
                  fontsize=9, fontweight="semibold", loc="left", pad=4, color=INK)
    panel_label(axc, "c", x=-0.02, y=1.06)

    fig.savefig(out / "fig3_candidate_states.png"); plt.close(fig)
    print("  fig3_candidate_states.png")


# ---------------------------------------------------------------------------
# Figure 4 — redesigned case cards (no overlap)
# ---------------------------------------------------------------------------
def figure4(D, out):
    """Six full-width case cards — avoids side-by-side molecule/text collisions."""
    fig = plt.figure(figsize=(11.0, 16.5), facecolor=PAPER)
    outer = fig.add_gridspec(6, 1, hspace=0.38)

    for k, (cid, role) in enumerate(CASE_STUDIES):
        card = outer[k].subgridspec(1, 3, width_ratios=[0.95, 0.85, 1.35], wspace=0.20)
        axm = fig.add_subplot(card[0, 0])
        axinfo = fig.add_subplot(card[0, 1])
        axtr = fig.add_subplot(card[0, 2])

        emb, act, highs = _case_exact_rules(D, cid)
        scal = D.scal[D.scal.candidate_id == cid]
        vals = D.cand_props(cid)
        smiles = D.smi.get(cid, "")

        hdr = f"{cid}  ·  {role}"
        if len(scal):
            s = scal.iloc[0]
            hdr += (f"\nPareto={'yes' if s.on_pareto else 'no'}   "
                    f"beat {int(s.n_obj_beat_benchmark)}/7   "
                    f"Δcomp={s.composite_vs_benchmark:+.2f}")
        _draw_mol(axm, smiles, highs, title=hdr)

        axinfo.set_xlim(0, 1); axinfo.set_ylim(0, 1); axinfo.set_axis_off()
        y = 0.98
        axinfo.text(0.0, y, "vs top-10", fontsize=7.5, fontweight="semibold",
                    color=INK, va="top")
        y -= 0.10
        for p in D.props:
            if p not in vals:
                continue
            v, ref = vals[p], D.ref[p]
            beat = D.d[p] * (v - ref) > 0
            axinfo.text(0.0, y, f"{'●' if beat else '○'} {p}",
                        fontsize=7, color=(C_GREEN if beat else MUTE), va="top")
            axinfo.text(0.38, y, f"{v:.1f}", fontsize=7, color=INK, va="top")
            axinfo.text(0.68, y, f"({ref:.1f})", fontsize=6.5, color=MUTE, va="top")
            y -= 0.085
        y -= 0.02
        axinfo.plot([0, 1], [y + 0.03, y + 0.03], color=RULE, lw=0.6,
                    transform=axinfo.transAxes)
        y -= 0.02
        if emb:
            axinfo.text(0.0, y, "Embodied B (blue)", fontsize=7,
                        fontweight="semibold", color=C_BLUE, va="top")
            y -= 0.085
            axinfo.text(0.0, y, fam_arrow(emb["tr"]), fontsize=6.5, color=INK, va="top")
            y -= 0.08
            axinfo.text(0.0, y, f"gated {emb['score']:+.2f} SD · n={emb['n_pairs']}",
                        fontsize=6.5, color=MUTE, va="top")
            y -= 0.10
        if act:
            tag = "orange" if act.get("highlighted") else "not painted"
            axinfo.text(0.0, y, f"Actionable A ({tag})", fontsize=7,
                        fontweight="semibold", color=C_ORANGE, va="top")
            y -= 0.085
            axinfo.text(0.0, y, fam_arrow(act["tr"]), fontsize=6.5, color=INK, va="top")
            y -= 0.08
            axinfo.text(0.0, y, f"gated {act['score']:+.2f} SD · n={act['n_pairs']}",
                        fontsize=6.5, color=MUTE, va="top")
        if not emb and not act:
            axinfo.text(0.0, y, "No clean exact rules in scope",
                        fontsize=7, color=MUTE, va="top")

        if act:
            cap = [
                f"Exact edit · {fam_arrow(act['tr'])} · "
                f"gated={act['score']:+.2f} SD · n={act['n_pairs']}",
                f"A={fraglabel(act['frag_from'])}  →  B={fraglabel(act['frag_to'])}",
            ]
            _draw_exact_transform(axtr, act["frag_from"], act["frag_to"], cap,
                                  w=520, h=200)
        elif emb:
            cap = [
                f"Embodied endpoint · {fam_arrow(emb['tr'])} · "
                f"gated={emb['score']:+.2f} SD · n={emb['n_pairs']}",
                f"A={fraglabel(emb['frag_from'])}  →  B={fraglabel(emb['frag_to'])}",
            ]
            _draw_exact_transform(axtr, emb["frag_from"], emb["frag_to"], cap,
                                  w=520, h=200)
        else:
            axtr.set_axis_off()
            axtr.text(0.5, 0.5, "No clean exact transform in scope",
                      ha="center", va="center", fontsize=8, color=MUTE,
                      transform=axtr.transAxes)

    fig.suptitle(
        "Case studies with exact A→B transforms   "
        "(gated = Σ significant directed z, SD units)",
        fontsize=10, fontweight="semibold", color=INK, y=0.995)
    fig.savefig(out / "fig4_case_studies.png"); plt.close(fig)
    print("  fig4_case_studies.png")


# ---------------------------------------------------------------------------
# Figure 5
# ---------------------------------------------------------------------------
def figure5(D, out):
    props = D.props
    rec = D.rec.copy()
    fig = plt.figure(figsize=(12.2, 6.0), facecolor=PAPER)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.25, 0.95], wspace=0.42)


    axa = fig.add_subplot(gs[0, 0])
    style_ax(axa)
    if len(rec):
        g = (rec.groupby("chemotype_edit")
             .agg(med=("composite_gated", "median"),
                  n_cand=("candidate_id", "nunique"),
                  med_n=("global_n_support", "median"))
             .sort_values("med", ascending=False).head(7))
        Z, labels, sigs = [], [], []
        for edit in g.index:
            sub = rec[rec.chemotype_edit == edit]
            Z.append([sub[f"z_{p}"].astype(float).mean() for p in props])
            sigs.append([(sub[f"sig_{p}"].astype(str).str.lower()
                          .isin(["true", "1", "1.0"])).mean() >= 0.5
                         for p in props])
            labels.append(fam_arrow(edit))
        Z = np.array(Z, float)
        vmax = max(np.nanmax(np.abs(Z)), 0.5)
        im = axa.imshow(Z, aspect="auto", cmap=DIVERGE, vmin=-vmax, vmax=vmax)
        axa.set_xticks(range(len(props))); axa.set_xticklabels(props)
        axa.set_yticks(range(len(labels))); axa.set_yticklabels(labels, fontsize=7)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if sigs[i][j]:
                    axa.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=0.8))
        axa.set_title("Top recommended edit families", pad=8)
        cb = fig.colorbar(im, ax=axa, location="bottom", fraction=0.06,
                          pad=0.14, aspect=30)
        cb.set_label("mean directed z", fontsize=7.5)
        cb.outline.set_linewidth(0.4)
    panel_label(axa, "a")

    axb = fig.add_subplot(gs[0, 1])
    style_ax(axb, grid=True)
    grid = D.state_table
    counts = []
    for tr in list(dict.fromkeys(grid["tr"].tolist()))[:9]:
        sub = grid[grid.tr == tr]
        counts.append({
            "tr": fam_arrow(tr),
            "actionable": int((sub.state == "actionable").sum()),
            "mixed": int((sub.state == "mixed").sum()),
            "already": int((sub.state == "already").sum()),
            "neither": int((sub.state == "neither").sum()),
        })
    C = pd.DataFrame(counts)
    C["opp_n"] = C.actionable + C.mixed
    C = C.sort_values("opp_n", ascending=True)
    y = np.arange(len(C))
    left = np.zeros(len(C))
    for key, col in [("actionable", C_ORANGE), ("mixed", C_PURPLE),
                     ("already", C_BLUE), ("neither", C_GREY)]:
        axb.barh(y, C[key].values, left=left, color=col, height=0.72,
                 edgecolor="white", linewidth=0.4, label=key, zorder=2)
        left = left + C[key].values
    axb.set_yticks(y); axb.set_yticklabels(C["tr"].map(fam_compact), fontsize=7)
    axb.set_xlabel("# candidates")
    axb.set_title("Applicability of key families", pad=8)
    axb.set_xlim(0, max(left.max() * 1.02, 1))
    axb.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4,
               fontsize=6.5, handlelength=0.9, columnspacing=0.9,
               handletextpad=0.35)
    panel_label(axb, "b", x=-0.28)


    axc = fig.add_subplot(gs[0, 2])
    style_ax(axc)
    n_all = len(D.candidate_ids)
    paired = int(D.profile.paired.sum())
    n_rec = D.rec.candidate_id.nunique() if len(D.rec) else 0
    n_strong = int(((D.profile.opp > 0) | (D.profile.mat > 0)).sum())
    local_true = 0
    if len(D.rec) and "has_local_evidence" in D.rec.columns:
        local_true = int(D.rec.has_local_evidence.astype(str).str.lower()
                         .isin(["true", "1", "1.0"]).sum())
    unpaired_ids = sorted(set(D.candidate_ids) -
                          set(D.profile.loc[D.profile.paired, "candidate_id"]))

    metrics = [
        ("Candidates", n_all, C_INK),
        ("Paired in DB", paired, C_BLUE),
        ("Strong-clean membership", n_strong, C_SKY),
        ("With ≥1 recommendation", n_rec, C_ORANGE),
    ]
    yy = np.arange(len(metrics))
    axc.barh(yy, [m[1] for m in metrics], color=[m[2] for m in metrics],
             height=0.62, edgecolor="none", zorder=2)
    axc.set_yticks(yy)
    axc.set_yticklabels([m[0] for m in metrics], fontsize=7)
    axc.set_xlabel("count")
    axc.set_xlim(0, n_all * 1.15)
    for i, (_, v, _) in enumerate(metrics):
        axc.text(v + 1, i, str(v), va="center", fontsize=7.5, color=INK)
    axc.set_title("Coverage", pad=8)
    note = (f"Unpaired: {', '.join(unpaired_ids) if unpaired_ids else '—'}\n"
            f"Local confirmation: {local_true}/{len(D.rec) if len(D.rec) else 0}\n"
            f"Scope: carbon-anchored cuts;\nN–N/azoxy linkers excluded")
    axc.text(0.0, -0.28, note, transform=axc.transAxes, fontsize=6.5,
             color=MUTE, va="top", linespacing=1.35)
    panel_label(axc, "c", x=-0.35)

    fig.savefig(out / "fig5_recommendations.png"); plt.close(fig)
    print("  fig5_recommendations.png")


# ---------------------------------------------------------------------------
# Supplementary
# ---------------------------------------------------------------------------
def supp(D, out):
    props = D.props

    # S1
    Zrows = {}
    for _, r in D.fg.iterrows():
        z = {p: (D.d[p] * r.get(f"A_{p}") / D.s_global[p]
                 if pd.notna(r.get(f"A_{p}")) else np.nan) for p in props}
        z["n"] = r["n_total_pairs"]
        Zrows[r["fragment"]] = z
    Z = pd.DataFrame(Zrows).T
    fig = plt.figure(figsize=(11.0, 4.4), facecolor=PAPER)
    gs = fig.add_gridspec(1, 2, wspace=0.32)
    ax = fig.add_subplot(gs[0, 0]); style_ax(ax)
    sens = "nu" if "nu" in props else ("IS" if "IS" in props else props[-1])
    ax.scatter(Z["Q"], Z[sens],
               s=12 + 140 * (Z["n"].astype(float) / Z["n"].max()),
               c=C_BLUE, edgecolor="white", linewidth=0.3, alpha=0.75)
    ax.axhline(0, color=MUTE, ls=":", lw=0.7); ax.axvline(0, color=MUTE, ls=":", lw=0.7)
    ax.set_xlabel("Fragment goodness z[Q]"); ax.set_ylabel(f"z[{sens}]")
    ax.set_title("Fragment goodness (secondary)", pad=8)
    panel_label(ax, "a")
    axb = fig.add_subplot(gs[0, 1]); style_ax(axb)
    keep = Z.sort_values("n", ascending=False).head(30)
    mat = np.nan_to_num(keep[props].values.astype(float))
    if mat.shape[0] > 2:
        order = leaves_list(linkage(mat, method="ward"))
        mat, labs = mat[order], [fraglabel(keep.index[i]) for i in order]
    else:
        labs = [fraglabel(i) for i in keep.index]
    vmax = max(np.nanmax(np.abs(mat)), 0.5)
    im = axb.imshow(mat, aspect="auto", cmap=DIVERGE, vmin=-vmax, vmax=vmax)
    axb.set_xticks(range(len(props))); axb.set_xticklabels(props, fontsize=8)
    axb.set_yticks(range(len(labs)))
    axb.set_yticklabels(labs, fontsize=5.5, fontfamily="monospace")
    axb.set_title("Top-support fragment archetypes", pad=8)
    fig.colorbar(im, ax=axb, fraction=0.04, pad=0.02).outline.set_linewidth(0.4)
    panel_label(axb, "b", x=-0.35)
    fig.savefig(out / "supp_S1_fragment_atlas.png"); plt.close(fig)
    print("  supp_S1_fragment_atlas.png")

    # S2
    t = D.trans[D.trans.n_pairs >= config.MIN_RULE_SUPPORT]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=PAPER)
    style_ax(ax)
    ax.scatter(t.composite_gated, np.log10(t.n_pairs.astype(float)),
               s=4, c=C_GREY, alpha=0.2, edgecolor="none", rasterized=True)
    clean = D.clean_rules
    ax.scatter(clean.composite_gated, np.log10(clean.n_pairs.astype(float)),
               s=7, c=C_ORANGE, alpha=0.35, edgecolor="none",
               label="clean n≥20, no worsen", rasterized=True)
    ax.axvline(0, color=MUTE, ls=":", lw=0.7)
    ax.set_xlabel("Gated composite (SD units)")
    ax.set_ylabel("log₁₀ n pairs")
    ax.set_title("All directed rules", pad=8)
    ax.legend(loc="upper left")
    fig.savefig(out / "supp_S2_rule_volcano.png"); plt.close(fig)
    print("  supp_S2_rule_volcano.png")

    # S3
    fig = plt.figure(figsize=(11.0, 4.4), facecolor=PAPER)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.32)
    ax = fig.add_subplot(gs[0, 0]); style_ax(ax)
    rtm = D.rtm.dropna(subset=["centered", "mean_delta"])
    for k, p in enumerate(props):
        sub = rtm[rtm.property == p]
        ax.scatter(sub.centered, sub.mean_delta, s=10, alpha=0.7,
                   color=plt.cm.Set2(k / max(1, len(props))), label=p,
                   edgecolor="none")
    ok = rtm.dropna(subset=["centered", "mean_delta"])
    if len(ok) > 2:
        b, a = np.polyfit(ok.centered, ok.mean_delta, 1)
        xs = np.linspace(ok.centered.min(), ok.centered.max(), 40)
        ax.plot(xs, a + b * xs, color=INK, lw=1.3, label=f"slope={b:+.2f}")
    ax.axhline(0, color=MUTE, ls=":", lw=0.7); ax.axvline(0, color=MUTE, ls=":", lw=0.7)
    ax.set_xlabel("Candidate − DB mean"); ax.set_ylabel("Mean Δ from edits")
    ax.set_title("Regression-to-mean", pad=8)
    ax.legend(fontsize=6, ncol=2, loc="best")
    panel_label(ax, "a")

    axb = fig.add_subplot(gs[0, 1]); style_ax(axb, grid=True)
    sb = D.scal.sort_values("composite_vs_benchmark", ascending=True)
    cols = [C_BLUE if n >= len(props) - 1 else C_GREY
            for n in sb.n_obj_beat_benchmark]
    axb.barh(range(len(sb)), sb.composite_vs_benchmark, color=cols,
             height=0.85, edgecolor="none")
    axb.axvline(0, color=INK, lw=0.7)
    axb.set_yticks([])
    axb.set_xlabel("Composite vs top-10 benchmark")
    axb.set_title("All candidates vs benchmark", pad=8)
    case_set = {c for c, _ in CASE_STUDIES}
    for i, (_, r) in enumerate(sb.iterrows()):
        if r.candidate_id in case_set or r.n_obj_beat_benchmark >= len(props) - 1:
            axb.text(r.composite_vs_benchmark, i, f"  {r.candidate_id}",
                     fontsize=5.5, va="center", color=INK)
    panel_label(axb, "b", x=-0.08)
    fig.savefig(out / "supp_S3_rtm_benchmark.png"); plt.close(fig)
    print("  supp_S3_rtm_benchmark.png")

    # S4
    unpaired = [c for c in D.candidate_ids
                if c not in set(D.profile.loc[D.profile.paired, "candidate_id"])]
    if not unpaired:
        unpaired = ["cand_023", "cand_047", "cand_049"]
    fig, axes = plt.subplots(1, len(unpaired),
                             figsize=(3.4 * len(unpaired), 3.2), facecolor=PAPER)
    if len(unpaired) == 1:
        axes = [axes]
    for ax, cid in zip(axes, unpaired):
        _draw_mol(ax, D.smi.get(cid, ""), None, title=f"{cid}  (unpaired)")
    fig.suptitle("Unpaired candidates under current cut-smarts",
                 fontsize=9.5, fontweight="semibold", color=INK)
    fig.savefig(out / "supp_S4_unpaired.png"); plt.close(fig)
    print("  supp_S4_unpaired.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(config.RESULTS))
    ap.add_argument("--master", default=str(config.MASTER_CSV))
    ap.add_argument("--properties", nargs="+", default=config.PROPERTIES)
    ap.add_argument("--out-dir", default=str(config.FIG_DIR))
    args = ap.parse_args()

    res = Path(args.results_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print("Loading data…")
    D = Data(res, Path(args.master), args.properties, config.directions())
    print("Rendering journal figures:")
    figure1(D, out)
    figure2(D, out)
    figure3(D, out)
    figure4(D, out)
    figure5(D, out)
    supp(D, out)
    print(f"All figures → {out}/")


if __name__ == "__main__":
    main()
