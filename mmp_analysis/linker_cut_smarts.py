"""
linker_cut_smarts.py -- acyclic cut patterns for EM bis-azole / linker MMPA.

mmpdb accepts only one cut-smarts string, but RDKit cannot OR two bond
patterns in a single SMARTS.  The pilot builder unions matches from these
patterns via MultiCutFragmentFilter (see linker_fragment.py).

Rules enforced:
  * !@ on every pattern  -> never cut ring bonds (tetrazole N-N safe)
  * !=!#                 -> single bonds only (azide internal doubles excluded)
  * default C-anchored cuts retained for ring substituents
  * additional acyclic N-N for inter-ring linkers / nitramines
"""

from rdkit import Chem

# Default mmpdb carbon-anchored acyclic single-bond cuts.
CUT_C_ACYCLIC = (
    "[#6+0;!$(*=,#[!#6])]!@!=!#[!#0;!#1;!$([CH2]);!$([CH3][CH2])]"
)

# Additional acyclic N-N single bonds (hydrazine linkers, ring N–linker N, etc.).
CUT_NN_ACYCLIC = "[#7]!@!=!#[#7]"

CUT_PATTERNS = [CUT_C_ACYCLIC, CUT_NN_ACYCLIC]

# For logging / fragments header (joined with " | ").
CUT_SMARTS_LABEL = " | ".join(CUT_PATTERNS)

AZIDE_SMARTS = "[N-]=[N+]=[#7]"


def compile_cut_patterns():
    pats = []
    for smarts in CUT_PATTERNS:
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            raise ValueError(f"invalid cut SMARTS: {smarts}")
        pats.append(pat)
    return pats


def cut_bond_records(mol, patterns=None):
    """Return list of dicts describing each matched acyclic cut bond."""
    if patterns is None:
        patterns = compile_cut_patterns()
    seen = set()
    out = []
    for pat in patterns:
        for i, j in mol.GetSubstructMatches(pat, uniquify=True):
            key = tuple(sorted((i, j)))
            if key in seen:
                continue
            seen.add(key)
            b = mol.GetBondBetweenAtoms(i, j)
            a1 = mol.GetAtomWithIdx(i)
            a2 = mol.GetAtomWithIdx(j)
            out.append({
                "i": i, "j": j,
                "a1": a1.GetSymbol(), "a2": a2.GetSymbol(),
                "order": b.GetBondTypeAsDouble(),
                "in_ring": b.IsInRing(),
            })
    return out


def azide_atoms(mol):
    env = set()
    pat = Chem.MolFromSmarts(AZIDE_SMARTS)
    if pat is None:
        return env
    for match in mol.GetSubstructMatches(pat):
        env.update(match)
    return env


def audit_cuts(mol_id, smiles, patterns=None):
    """Flag suspicious cuts: ring bonds or azide-internal N-N."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [{"mol_id": mol_id, "issue": "parse_fail", "detail": smiles}]

    az = azide_atoms(mol)
    issues = []
    for rec in cut_bond_records(mol, patterns):
        i, j = rec["i"], rec["j"]
        if rec["in_ring"]:
            issues.append({
                "mol_id": mol_id, "issue": "ring_cut",
                "detail": f"{rec['a1']}-{rec['a2']} bond {i}-{j}",
            })
        if i in az and j in az:
            issues.append({
                "mol_id": mol_id, "issue": "azide_internal",
                "detail": f"N-N inside azide bond {i}-{j}",
            })
        if rec["order"] != 1.0:
            issues.append({
                "mol_id": mol_id, "issue": "non_single",
                "detail": f"bond order {rec['order']} at {i}-{j}",
            })
    return issues


def fragment_has_azide_split(frag_smiles):
    """True only if an azide motif appears with fewer than three nitrogens."""
    if not frag_smiles or frag_smiles == "None":
        return False
    for part in frag_smiles.replace("[H]", "").split("."):
        part = part.strip()
        if not part:
            continue
        trial = part.replace("*", "[H]")
        mol = Chem.MolFromSmiles(trial)
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol)
        if "[N-]" in smi and "[N+]" in smi:
            n_n = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N")
            if n_n < 3:
                return True
    return False
