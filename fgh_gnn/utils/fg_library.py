"""
All functional groups (and their SMARTS) taken from Daylight [1].

Reference:
    [1] https://www.daylight.com/dayhtml_tutorials/languages/smarts/
        smarts_examples.html
"""

from rdkit import Chem

# C-containing groups
ALLENIC = "allenic"
VINYLIC = "vinylic"
ACETYLENIC = "acetylenic"

# O-containing groups
ETHER = "ether"
HYDROXYL = "hydroxyl"
PEROXIDE = "peroxide"

# (C=O)-containing groups
ACYL_HALIDE = "acyl halide"
ALDEHYDE = "aldehyde"
ANHYDRIDE = "anhydride"
AMIDE = "amide"
AMIDINIUM = "amidinium"
CARBAMATE = "carbamate"  # includes carbamic esters, acids, zwitterions
CARBOXYLIC = "carboxylic"  # includes acid and conjugate base
CYANAMIDE = "cyanamide"
ESTER = "ester"  # hits anhydrides (but not formic)
KETONE = "ketone"

# N-containing groups
AMINE_1_2 = "1-2nd amine"  # primary or secondary; not amide
ENAMINE_ANILINE = "enamine/analine"
AZIDE = "azide"
AZO = "azo"  # includes diazene, azoxy, diazo
IMINE = "imine"  # substituted or un-substituted
IMINIUM = "iminium"
NITRATE = "nitrate"  # also nitrate anion
NITRILE = "nitrile"
NITRO = "nitro"
NITROSO = "nitroso"

# P-containing groups
PHOSPHORIC_ACID = "phosphoric acid"
PHOSPHORIC_ESTER = "phosphoric ester"

# S-containing groups
THIOL = "thiol"
MONOSULFIDE = "monosulfide"
DISULFIDE = "disulfide"
CARBON_THIOESTER = "carbon-thioester"
SULFINATE = "sulfinate"
SULFINIC_ACID = "sulfinic acid"
SULFONE = "sulfone"  # low specificity
SULFOXIDE = "sulfoxide"  # low specificity
SULFATE = "sulfate"  # sulfuric acid monoester
SULFENIC = "sulfenic"  # hits acid and conjugate base

FG_SMARTS = {
    ALLENIC: "[$([CX2](=C)=C)]",
    VINYLIC: "[$([CX3]=[CX3])]",
    ACETYLENIC: "[$([CX2]#C)]",
    ACYL_HALIDE: "[CX3](=[OX1])[F,Cl,Br,I]",
    ALDEHYDE: "[CX3H1](=O)[#6]",
    ANHYDRIDE: "[CX3](=[OX1])[OX2][CX3](=[OX1])",
    AMIDE: "[NX3][CX3](=[OX1])[#6]",
    AMIDINIUM: "[NX3][CX3]=[NX3+]",
    CARBAMATE: "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",
    CARBOXYLIC: "[CX3](=O)[OX1H0-,OX2H1]",
    CYANAMIDE: "[NX3][CX2]#[NX1]",
    ESTER: "[#6][CX3](=O)[OX2H0][#6]",
    KETONE: "[#6][CX3](=O)[#6]",
    ETHER: "[OD2]([#6])[#6]",
    ENAMINE_ANILINE: "[NX3][$(C=C),$(cc)]",
    AMINE_1_2: "[NX3;H2,H1;!$(NC=O)]",
    AZIDE: "[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]",
    AZO: "[NX2]=N",
    IMINE: "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",
    IMINIUM: "[NX3+]=[CX3]",
    NITRATE: "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]",
    NITRILE: "[NX1]#[CX2]",
    NITRO: "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
    NITROSO: "[NX2]=[OX1]",
    HYDROXYL: "[OX2H]",
    PEROXIDE: "[OX2,OX1-][OX2,OX1-]",
    PHOSPHORIC_ACID: "[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])"
                     "([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),"
                     "$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),"
                     "$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),"
                     "$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]",
    PHOSPHORIC_ESTER: "[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),"
                      "$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),"
                      "$([OX2]P)]),$([P+]([OX1-])([OX2][#6])([$([OX2H]),"
                      "$([OX1-]),$([OX2][#6])])[$([OX2H]),$([OX1-]),"
                      "$([OX2][#6]),$([OX2]P)])]",
    THIOL: "[#16X2H]",
    MONOSULFIDE: "[#16X2H0][!#16]",
    DISULFIDE: "[#16X2H0][#16X2H0]",
    CARBON_THIOESTER: "S([#6])[CX3](=O)[#6]",
    SULFINATE: "[$([#16X3](=[OX1])[OX2H0]),$([#16X3+]([OX1-])[OX2H0])]",
    SULFINIC_ACID: "[$([#16X3](=[OX1])[OX2H,OX1H0-]),"
                   "$([#16X3+]([OX1-])[OX2H,OX1H0-])]",
    SULFONE: "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",
    SULFOXIDE: "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",
    SULFATE: "[$([SX4](=O)(=O)(O)O),$([SX4+2]([O-])([O-])(O)O)]",
    SULFENIC: "[#16X2][OX2H,OX1H0-]"
}

FG_MOLS = {name: Chem.MolFromSmarts(s) for name, s in FG_SMARTS.items()}


def has_func_groups(mol):
    return {name: mol.HasSubstructMatch(fg_query)
            for name, fg_query in FG_MOLS.items()}


def analyze_func_groups(mol_batch):
    batch_len = 0
    fg_count = {name: 0 for name, _ in FG_MOLS.items()}

    for mol in mol_batch:
        for name, is_member in has_func_groups(mol).items():
            fg_count[name] += int(is_member)
        batch_len += 1

    fg_freq = {name: (count / batch_len) for name, count in fg_count.items()}

    return fg_count, fg_freq
