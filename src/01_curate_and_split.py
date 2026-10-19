#!/usr/bin/env python3
"""Clean, deterministic rebuild of the external validation set and its stratification.

Addresses Reviewer 3 point 2: distinguishes (i) InChIKey connectivity deduplication,
(ii) Bemis-Murcko scaffold overlap with the ENTIRE BBBP, and (iii) scaffold overlap
with the DEVELOPMENT set specifically.
"""
import warnings, json, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
RDLogger.DisableLog('rdApp.*')

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
D = _os.path.join(_ROOT, 'work') + _os.sep        # intermediate artefacts
DATA = _os.path.join(_ROOT, 'data') + _os.sep     # partition assignments
RESULTS = _os.path.join(_ROOT, 'results') + _os.sep
OUT = _os.path.join(_ROOT, 'figures') + _os.sep
for _p in (D, RESULTS, OUT):
    _os.makedirs(_p, exist_ok=True)


def parse(s):
    if not isinstance(s, str):
        return None
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None


def scaffold_of(s):
    m = parse(s)
    if m is None:
        return None
    try:
        # Must match the definition used to create the published split
        # (MurckoScaffoldSmiles); acyclic molecules yield '' and are kept as
        # their own explicit group rather than dropped.
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return None


def skeleton(s):
    m = parse(s)
    if m is None:
        return None
    try:
        return Chem.MolToInchiKey(m).split('-')[0]
    except Exception:
        return None


# ------------------------------------------------------------------ BBBP
bbbp = pd.read_csv(D + 'BBBP.csv')
bbbp['mol_ok'] = bbbp['smiles'].apply(lambda s: parse(s) is not None)
clean = bbbp[bbbp['mol_ok']].copy()
clean['scaffold'] = clean['smiles'].apply(scaffold_of)
clean['skel'] = clean['smiles'].apply(skeleton)
clean = clean[clean['scaffold'].notna() & clean['skel'].notna()].copy()
print('acyclic (empty-scaffold) BBBP molecules: %d' % (clean['scaffold'] == '').sum())

scaffolds = clean['scaffold'].unique()
tr_sc, tmp_sc = train_test_split(scaffolds, test_size=0.20, random_state=42)
va_sc, te_sc = train_test_split(tmp_sc, test_size=0.50, random_state=42)
train_df = clean[clean['scaffold'].isin(tr_sc)].copy()
valid_df = clean[clean['scaffold'].isin(va_sc)].copy()
test_df = clean[clean['scaffold'].isin(te_sc)].copy()
dev_df = pd.concat([train_df, valid_df])

print('BBBP  raw %d | valid %d | train %d valid %d test %d | dev %d'
      % (len(bbbp), len(clean), len(train_df), len(valid_df), len(test_df), len(dev_df)))

dev_scaffolds = set(dev_df['scaffold'])
all_scaffolds = set(clean['scaffold'])
all_skels = set(clean['skel'])

# ------------------------------------------------------------------ B3DB
b3 = pd.read_csv(D + 'b3.tsv', sep='\t')
n_raw = len(b3)
b3 = b3[['compound_name', 'SMILES', 'BBB+/BBB-', 'reference', 'group']].dropna(
    subset=['SMILES', 'BBB+/BBB-'])
b3['mol_ok'] = b3['SMILES'].apply(lambda s: parse(s) is not None)
b3 = b3[b3['mol_ok']].copy()
n_parsed = len(b3)
b3['y'] = (b3['BBB+/BBB-'].str.strip() == 'BBB+').astype(int)
b3['skel'] = b3['SMILES'].apply(skeleton)
b3 = b3[b3['skel'].notna()].copy()

conflict = b3.groupby('skel')['y'].nunique()
conflict_skels = set(conflict[conflict > 1].index)
b3 = b3[~b3['skel'].isin(conflict_skels)]
b3 = b3.drop_duplicates(subset='skel', keep='first').copy()
n_unique = len(b3)

ext = b3[~b3['skel'].isin(all_skels)].copy()
n_overlap_removed = n_unique - len(ext)
ext['scaffold'] = ext['SMILES'].apply(scaffold_of)
ext = ext[ext['scaffold'].notna()].copy()

ext['scaffold_in_bbbp'] = ext['scaffold'].isin(all_scaffolds)
ext['scaffold_in_dev'] = ext['scaffold'].isin(dev_scaffolds)

print('\nB3DB  raw %d | parsable %d | label-consistent unique skeletons %d'
      % (n_raw, n_parsed, n_unique))
print('      removed as conflicting: %d skeletons' % len(conflict_skels))
print('      removed as overlapping BBBP by skeleton: %d' % n_overlap_removed)
print('EXTERNAL n=%d  pos %d  neg %d' % (len(ext), (ext.y == 1).sum(), (ext.y == 0).sum()))
print('  scaffold shared with ENTIRE BBBP : %d (%.1f%%)'
      % (ext.scaffold_in_bbbp.sum(), 100 * ext.scaffold_in_bbbp.mean()))
print('  scaffold shared with DEVELOPMENT : %d (%.1f%%)'
      % (ext.scaffold_in_dev.sum(), 100 * ext.scaffold_in_dev.mean()))
print('  scaffold NOVEL vs DEVELOPMENT    : %d (%.1f%%)'
      % ((~ext.scaffold_in_dev).sum(), 100 * (~ext.scaffold_in_dev).mean()))
for g, sub in ext.groupby('scaffold_in_dev'):
    print('    %-12s n=%4d  pos %4d  neg %4d  prevalence %.3f'
          % ('dev-overlap' if g else 'dev-novel', len(sub), (sub.y == 1).sum(),
             (sub.y == 0).sum(), sub.y.mean()))

# ------------------------------------------------------------------ features
def morgan(s, radius=2, n_bits=2048):
    m = parse(s)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
    a = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, a)
    return a


DESC = ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds',
        'RingCount', 'AromaticRings', 'FractionCSP3', 'HeavyAtomCount']


def descriptors(s):
    m = parse(s)
    return [Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m),
            Lipinski.NumHDonors(m), Lipinski.NumHAcceptors(m),
            Lipinski.NumRotatableBonds(m), rdMolDescriptors.CalcNumRings(m),
            rdMolDescriptors.CalcNumAromaticRings(m),
            rdMolDescriptors.CalcFractionCSP3(m), m.GetNumHeavyAtoms()]


def build(sm):
    M = np.array([morgan(s) for s in sm])
    Dd = np.nan_to_num(np.array([descriptors(s) for s in sm], dtype=np.float32),
                       nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([M, Dd], axis=1), M, Dd


Xtr_c, Xtr_m, Xtr_d = build(train_df['smiles'])
Xva_c, Xva_m, Xva_d = build(valid_df['smiles'])
Xte_c, Xte_m, Xte_d = build(test_df['smiles'])
Xdv_c, Xdv_m, Xdv_d = build(dev_df['smiles'])
Xex_c, Xex_m, Xex_d = build(ext['SMILES'])

np.savez_compressed(
    D + 'clean_features.npz',
    Xtr_c=Xtr_c, Xtr_m=Xtr_m, Xtr_d=Xtr_d, ytr=train_df['p_np'].values.astype(int),
    Xva_c=Xva_c, Xva_m=Xva_m, Xva_d=Xva_d, yva=valid_df['p_np'].values.astype(int),
    Xte_c=Xte_c, Xte_m=Xte_m, Xte_d=Xte_d, yte=test_df['p_np'].values.astype(int),
    Xdv_c=Xdv_c, Xdv_m=Xdv_m, Xdv_d=Xdv_d, ydv=dev_df['p_np'].values.astype(int),
    Xex_c=Xex_c, Xex_m=Xex_m, Xex_d=Xex_d, yex=ext['y'].values.astype(int),
    ext_in_dev=ext['scaffold_in_dev'].values,
    ext_in_bbbp=ext['scaffold_in_bbbp'].values)

for nm, fr in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
    fr[['name', 'smiles', 'p_np', 'scaffold']].to_csv(D + 'clean_%s.csv' % nm, index=False)
ext[['compound_name', 'SMILES', 'y', 'scaffold', 'scaffold_in_bbbp',
     'scaffold_in_dev', 'reference', 'group']].to_csv(D + 'clean_external.csv', index=False)

summary = {
    'bbbp_raw': int(len(bbbp)), 'bbbp_valid': int(len(clean)),
    'train': int(len(train_df)), 'valid': int(len(valid_df)), 'test': int(len(test_df)),
    'dev': int(len(dev_df)),
    'train_pos': int((train_df.p_np == 1).sum()), 'train_neg': int((train_df.p_np == 0).sum()),
    'valid_pos': int((valid_df.p_np == 1).sum()), 'valid_neg': int((valid_df.p_np == 0).sum()),
    'test_pos': int((test_df.p_np == 1).sum()), 'test_neg': int((test_df.p_np == 0).sum()),
    'b3db_raw': int(n_raw), 'b3db_parsed': int(n_parsed), 'b3db_unique': int(n_unique),
    'b3db_conflicting': int(len(conflict_skels)), 'b3db_overlap_removed': int(n_overlap_removed),
    'external_n': int(len(ext)), 'external_pos': int((ext.y == 1).sum()),
    'external_neg': int((ext.y == 0).sum()),
    'ext_scaffold_in_bbbp': int(ext.scaffold_in_bbbp.sum()),
    'ext_scaffold_in_dev': int(ext.scaffold_in_dev.sum()),
    'ext_scaffold_novel_vs_dev': int((~ext.scaffold_in_dev).sum()),
    'ext_novel_pct_vs_dev': float(100 * (~ext.scaffold_in_dev).mean()),
    'n_unique_scaffolds_bbbp': int(clean['scaffold'].nunique()),
    'n_unique_scaffolds_dev': int(dev_df['scaffold'].nunique()),
}
json.dump(summary, open(D + 'clean_summary.json', 'w'), indent=1)
print('\nsaved clean_summary.json')
