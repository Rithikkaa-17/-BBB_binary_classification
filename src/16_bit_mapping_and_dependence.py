#!/usr/bin/env python3
"""Reviewer 2, point 4: map the highest-attribution fingerprint bits back to the
substructures that set them (reporting hash collisions), and record the
dependence of attribution on descriptor value, with Spearman alongside Pearson
so that a monotone-but-nonlinear relationship is not scored as absent.
"""
import warnings, json, os, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import pearsonr, spearmanr
RDLogger.DisableLog('rdApp.*')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
z = np.load(os.path.join(ROOT, 'work', 'clean_features.npz'))
sv = np.load(os.path.join(ROOT, 'work', 'sv42.npy'))          # written by 13_block_attribution
te = pd.read_csv(os.path.join(ROOT, 'work', 'clean_test.csv'))
DESC = ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds',
        'RingCount', 'AromaticRings', 'FractionCSP3', 'HeavyAtomCount']
X = z['Xte_c']; out = {}

# --- linear vs monotone dependence for each descriptor ---
print('=== descriptor attribution: Pearson vs Spearman, and curvature ===', flush=True)
for j, nm in enumerate(DESC):
    v = X[:, 2048 + j]; a = sv[:, 2048 + j]
    pr = pearsonr(v, a)[0]; sr = spearmanr(v, a)[0]
    # quadratic term on standardised value, to test the parabolic case explicitly
    vs = (v - v.mean()) / (v.std() + 1e-9)
    A = np.column_stack([np.ones_like(vs), vs, vs ** 2])
    beta, *_ = np.linalg.lstsq(A, a, rcond=None)
    r2_lin = 1 - ((a - A[:, :2] @ beta[:2]) ** 2).sum() / ((a - a.mean()) ** 2).sum()
    r2_quad = 1 - ((a - A @ beta) ** 2).sum() / ((a - a.mean()) ** 2).sum()
    out.setdefault('dependence', {})[nm] = {'pearson': float(pr), 'spearman': float(sr),
                                            'r2_linear': float(r2_lin), 'r2_quadratic': float(r2_quad),
                                            'quadratic_coef': float(beta[2])}
    print('  %-16s Pearson %+.3f  Spearman %+.3f  R2 lin %.3f -> quad %.3f' % (nm, pr, sr, r2_lin, r2_quad), flush=True)
    np.save(os.path.join(ROOT, 'work', 'dep_%s.npy' % nm), np.column_stack([v, a]))

# --- which substructures set the top-ranked bits, and do they collide? ---
mabs = np.abs(sv).mean(axis=0)
top = [int(i) for i in np.argsort(-mabs) if i < 2048][:10]
print('\n=== top fingerprint bits: substructures and collisions ===', flush=True)
env = {b: [] for b in top}
for s in te['smiles']:
    m = Chem.MolFromSmiles(s)
    if m is None: continue
    bi = {}
    AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048, bitInfo=bi)
    for b in top:
        for (atom, rad) in bi.get(b, []):
            amap = {}
            sub = Chem.PathToSubmol(m, Chem.FindAtomEnvironmentOfRadiusN(m, rad, atom), atomMap=amap) \
                  if rad > 0 else None
            smi = Chem.MolToSmiles(sub, rootedAtAtom=amap.get(atom, -1), canonical=True) if sub is not None \
                  else m.GetAtomWithIdx(atom).GetSymbol()
            env[b].append(smi)
for b in top:
    uniq = pd.Series(env[b]).value_counts()
    out.setdefault('bits', {})[str(b)] = {'mean_abs': float(mabs[b]),
                                          'n_distinct_environments': int(len(uniq)),
                                          'environments': {str(k): int(v) for k, v in uniq.head(5).items()}}
    print('  bit %-5d mean|SHAP| %.5f  distinct environments %d  top: %s'
          % (b, mabs[b], len(uniq), ', '.join(list(uniq.index[:3])) if len(uniq) else 'none in test set'), flush=True)

json.dump(out, open(os.path.join(ROOT, 'results', 'bit_mapping_and_dependence.json'), 'w'), indent=1)
print('\nsaved results/bit_mapping_and_dependence.json')
