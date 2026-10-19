#!/usr/bin/env python3
"""Fair, like-for-like comparison: 2 algorithms x 3 representations, 5 seeds,
evaluated on the internal test set and the external set, with paired tests,
sum of ranking differences (SRD) and Pareto analysis.

Addresses R2 (significance, SRD, Pareto, 'other algorithms do the same') and
R3 (fairness of model comparison: identical splits, preprocessing, budget).
"""
import warnings, json, itertools, numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, matthews_corrcoef,
                             balanced_accuracy_score, brier_score_loss, f1_score)
from xgboost import XGBClassifier

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
D = _os.path.join(_ROOT, 'work') + _os.sep        # intermediate artefacts
DATA = _os.path.join(_ROOT, 'data') + _os.sep     # partition assignments
RESULTS = _os.path.join(_ROOT, 'results') + _os.sep
OUT = _os.path.join(_ROOT, 'figures') + _os.sep
for _p in (D, RESULTS, OUT):
    _os.makedirs(_p, exist_ok=True)
z = np.load(D + 'clean_features.npz')
ydv, yte, yex = z['ydv'], z['yte'], z['yex']
in_dev = z['ext_in_dev'].astype(bool)
SEEDS = [7, 21, 42, 84, 123]
REPS = {'Morgan': ('Xdv_m', 'Xte_m', 'Xex_m'),
        'Descriptors': ('Xdv_d', 'Xte_d', 'Xex_d'),
        'Fused': ('Xdv_c', 'Xte_c', 'Xex_c')}


def make(alg, seed):
    if alg == 'RF':
        return RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                      random_state=seed, n_jobs=-1)
    return XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.8,
                         colsample_bytree=0.7, min_child_weight=2, gamma=0.05,
                         reg_alpha=0.1, reg_lambda=1.0, objective='binary:logistic',
                         eval_metric='logloss', random_state=seed, n_jobs=-1)


rows, preds = {}, {}
for alg in ['RF', 'XGB']:
    for rep, (kd, kt, ke) in REPS.items():
        name = '%s + %s' % (alg, rep)
        ri, re_, pi5, pe5 = [], [], [], []
        for sd in SEEDS:
            m = make(alg, sd).fit(z[kd], ydv)
            pi = m.predict_proba(z[kt])[:, 1]
            pe = m.predict_proba(z[ke])[:, 1]
            ri.append(roc_auc_score(yte, pi)); re_.append(roc_auc_score(yex, pe))
            pi5.append(pi); pe5.append(pe)
        pi_m = np.mean(pi5, axis=0); pe_m = np.mean(pe5, axis=0)
        preds[name] = (pi_m, pe_m)
        rows[name] = {
            'int_roc_mean': float(np.mean(ri)), 'int_roc_sd': float(np.std(ri, ddof=1)),
            'ext_roc_mean': float(np.mean(re_)), 'ext_roc_sd': float(np.std(re_, ddof=1)),
            'int_pr': float(average_precision_score(yte, pi_m)),
            'ext_pr': float(average_precision_score(yex, pe_m)),
            'ext_mcc': float(matthews_corrcoef(yex, (pe_m >= 0.5).astype(int))),
            'ext_bacc': float(balanced_accuracy_score(yex, (pe_m >= 0.5).astype(int))),
            'ext_brier': float(brier_score_loss(yex, pe_m)),
            'ext_roc_novel': float(roc_auc_score(yex[~in_dev], pe_m[~in_dev])),
        }
        print('%-20s int %.4f+/-%.4f | ext %.4f+/-%.4f | ext-novel %.4f | ext Brier %.4f'
              % (name, rows[name]['int_roc_mean'], rows[name]['int_roc_sd'],
                 rows[name]['ext_roc_mean'], rows[name]['ext_roc_sd'],
                 rows[name]['ext_roc_novel'], rows[name]['ext_brier']))


def paired(y, pa, pb, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), \
        float(min(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0))


print('\nPAIRWISE TESTS ON THE EXTERNAL SET (ROC-AUC, 5,000 paired resamples)')
pair = {}
for a, b in itertools.combinations(rows, 2):
    d, lo, hi, p = paired(yex, preds[a][1], preds[b][1])
    pair['%s vs %s' % (a, b)] = {'delta': d, 'ci': [lo, hi], 'p': p,
                                 'significant': bool(lo > 0 or hi < 0)}
    print('  %-20s vs %-20s %+.4f (%+.4f,%+.4f) p=%.4f %s'
          % (a, b, d, lo, hi, p, 'SIG' if (lo > 0 or hi < 0) else 'ns'))

# ---------------------------------------------------------------- SRD
print('\nSUM OF RANKING DIFFERENCES (Heberger 2010) -- lower is better')
criteria = ['int_roc_mean', 'ext_roc_mean', 'ext_pr', 'ext_mcc', 'ext_bacc', 'ext_roc_novel']
names = list(rows)
M = np.array([[rows[n][c] for c in criteria] for n in names])   # models x criteria
ref = M.max(axis=0)                                             # row-maximum reference
srd = {}
for i, n in enumerate(names):
    s = 0
    for j in range(len(criteria)):
        col = M[:, j]
        rank_model = (-col).argsort().argsort()[i] + 1
        rank_ref = 1 if col[i] == ref[j] else (-np.append(col, ref[j])).argsort().argsort()[-1] + 1
        s += abs(rank_model - rank_ref)
    srd[n] = int(s)
# normalise to percentage of the maximum possible SRD
k = len(criteria); m = len(names)
srd_max = k * (m - 1)
for n in sorted(srd, key=srd.get):
    print('  %-20s SRD %3d   SRD%% %5.1f' % (n, srd[n], 100 * srd[n] / srd_max))

# SRD validation by random ranking (CRRN-style permutation test)
rng = np.random.default_rng(42)
null = []
for _ in range(10000):
    perm = rng.permutation(m)
    s = 0
    for j in range(k):
        col = M[:, j]
        true_rank = (-col).argsort().argsort()
        s += np.abs(perm - true_rank).sum()
    null.append(s / m)
null = np.array(null)
print('  random-ranking null: 5th percentile %.1f (values below this are better than chance)'
      % np.percentile(null, 5))

# ---------------------------------------------------------------- Pareto
print('\nPARETO ANALYSIS (external ROC-AUC, external MCC, calibration = -Brier)')
obj = np.array([[rows[n]['ext_roc_mean'], rows[n]['ext_mcc'], -rows[n]['ext_brier']]
                for n in names])
front = []
for i in range(m):
    dominated = any(all(obj[j] >= obj[i]) and any(obj[j] > obj[i]) for j in range(m) if j != i)
    if not dominated:
        front.append(names[i])
print('  Pareto-optimal (non-dominated):', ', '.join(front))
print('  dominated:', ', '.join(n for n in names if n not in front))

json.dump({'rows': rows, 'pairwise_external': pair, 'srd': srd,
           'srd_max': srd_max, 'srd_null_p5': float(np.percentile(null, 5)),
           'pareto_front': front, 'criteria': criteria},
          open(D + 'fair_comparison.json', 'w'), indent=1)
np.savez(D + 'fair_preds.npz', **{k2.replace(' ', '_'): v[1] for k2, v in preds.items()},
         yex=yex, in_dev=in_dev)
print('\nsaved fair_comparison.json')
