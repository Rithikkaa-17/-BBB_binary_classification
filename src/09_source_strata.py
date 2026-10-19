#!/usr/bin/env python3
"""Source-stratified external performance for the final (seed-averaged) models."""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
D = _os.path.join(_ROOT, 'work') + _os.sep
DATA = _os.path.join(_ROOT, 'data') + _os.sep
RESULTS = _os.path.join(_ROOT, 'results') + _os.sep
f = np.load(D + 'final_preds.npz')
ext = pd.read_csv(DATA + 'external_B3DB_filtered.csv')
yex = f['yex']
assert len(ext) == len(yex)
grp = ext['group'].fillna('unspecified').astype(str).values
rng = np.random.default_rng(42)


def boot_ci(y, p, n=5000):
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) > 1:
            vals.append(roc_auc_score(y[s], p[s]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


OUT = {}
for model, key in (('RF + Fused', 'e_RF_Fused'), ('XGB + Fused', 'e_XGB_Fused')):
    p = f[key]
    OUT[model] = {}
    for g in sorted(pd.Series(grp).value_counts().index):
        m = grp == g
        if len(np.unique(yex[m])) < 2:
            continue
        OUT[model][g] = {'n': int(m.sum()), 'prevalence': float(yex[m].mean()),
                         'roc': float(roc_auc_score(yex[m], p[m])),
                         'ci': boot_ci(yex[m], p[m])}
        print('%-14s %-12s n=%4d prev %.3f ROC %.4f  CI %.4f-%.4f'
              % (model, g, m.sum(), yex[m].mean(), OUT[model][g]['roc'],
                 *OUT[model][g]['ci']))

json.dump(OUT, open(RESULTS + 'source_strata_final.json', 'w'), indent=1)
print('\nsaved source_strata_final.json | groups cover %d of %d external molecules'
      % (sum(v['n'] for v in OUT['RF + Fused'].values()), len(yex)))
