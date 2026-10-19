#!/usr/bin/env python3
"""Multi-criteria comparison at each configuration's validation-selected threshold.

Replaces the earlier computation, which scored the threshold-dependent criteria at a
fixed 0.5 and so did not evaluate any configuration at its own operating point.
All quantities here are taken from the same seed-averaged predictions and the same
validation-selected thresholds used in Table 3 and Supplementary Table S1.
"""
import json, numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, matthews_corrcoef,
                             balanced_accuracy_score, brier_score_loss)

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
D = _os.path.join(_ROOT, 'work') + _os.sep
DATA = _os.path.join(_ROOT, 'data') + _os.sep
RESULTS = _os.path.join(_ROOT, 'results') + _os.sep
f = np.load(D + 'final_preds.npz')
FR = json.load(open(RESULTS + 'final_results.json'))['models']
yte, yex, in_dev = f['yte'], f['yex'], f['in_dev'].astype(bool)
ORDER = ['RF + Morgan', 'RF + Descriptors', 'RF + Fused',
         'XGB + Morgan', 'XGB + Descriptors', 'XGB + Fused']

rows = {}
for n in ORDER:
    key = n.replace(' + ', '_')
    pi, pe = f['p_' + key], f['e_' + key]
    thr = FR[n]['external']['threshold']
    assert abs(thr - FR[n]['internal']['threshold']) < 1e-12
    rows[n] = {
        'threshold': float(thr),
        'int_roc': float(roc_auc_score(yte, pi)),
        'ext_roc': float(roc_auc_score(yex, pe)),
        'ext_pr': float(average_precision_score(yex, pe)),
        'ext_mcc': float(matthews_corrcoef(yex, (pe >= thr).astype(int))),
        'ext_bacc': float(balanced_accuracy_score(yex, (pe >= thr).astype(int))),
        'ext_brier': float(brier_score_loss(yex, pe)),
        'ext_roc_novel': float(roc_auc_score(yex[~in_dev], pe[~in_dev])),
    }
    # must agree with the published tables
    assert abs(rows[n]['ext_mcc'] - FR[n]['external']['MCC']) < 1e-9, n
    assert abs(rows[n]['ext_roc'] - FR[n]['external']['ROC-AUC']) < 1e-9, n

criteria = ['int_roc', 'ext_roc', 'ext_pr', 'ext_mcc', 'ext_bacc', 'ext_roc_novel']
names = ORDER
M = np.array([[rows[n][c] for c in criteria] for n in names])
ref = M.max(axis=0)
srd = {}
for i, n in enumerate(names):
    s = 0
    for j in range(len(criteria)):
        col = M[:, j]
        rank_model = (-col).argsort().argsort()[i] + 1
        rank_ref = 1 if col[i] == ref[j] else (-np.append(col, ref[j])).argsort().argsort()[-1] + 1
        s += abs(rank_model - rank_ref)
    srd[n] = int(s)
k, m = len(criteria), len(names)
srd_max = k * (m - 1)

rng = np.random.default_rng(42)
null = []
for _ in range(10000):
    perm = rng.permutation(m)
    s = 0
    for j in range(k):
        s += np.abs(perm - (-M[:, j]).argsort().argsort()).sum()
    null.append(s / m)
p5 = float(np.percentile(null, 5))

obj = np.array([[rows[n]['ext_roc'], rows[n]['ext_mcc'], -rows[n]['ext_brier']] for n in names])
front = [names[i] for i in range(m)
         if not any(all(obj[j] >= obj[i]) and any(obj[j] > obj[i]) for j in range(m) if j != i)]

print('%-20s %5s %8s %8s %8s %8s %8s %8s  SRD  SRD%%'
      % ('configuration', 'thr', 'intROC', 'extROC', 'extPR', 'extMCC', 'extbACC', 'novel'))
for n in sorted(names, key=srd.get):
    r = rows[n]
    print('%-20s %5.2f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f  %3d %5.1f'
          % (n, r['threshold'], r['int_roc'], r['ext_roc'], r['ext_pr'], r['ext_mcc'],
             r['ext_bacc'], r['ext_roc_novel'], srd[n], 100 * srd[n] / srd_max))
print('\nrandom-ranking null 5th percentile: %.2f' % p5)
print('better than chance:', [n for n in names if srd[n] < p5])
print('Pareto front:', front)
print('dominated   :', [n for n in names if n not in front])

json.dump({'rows': rows, 'criteria': criteria, 'srd': srd, 'srd_max': srd_max,
           'srd_null_p5': p5, 'pareto_front': front},
          open(RESULTS + 'multicriteria_final.json', 'w'), indent=1)
print('\nsaved multicriteria_final.json')
