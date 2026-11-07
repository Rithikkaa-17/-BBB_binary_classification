#!/usr/bin/env python3
"""Reviewer 2, point 2: repeat the headline comparison in a second software
environment, so that the fusion effect can be separated from version-specific
behaviour of the fingerprint and tree implementations.

Run this with a different interpreter from the pinned one; it writes a file
tagged with whatever versions it finds.
"""
import warnings, json, os, numpy as np
warnings.filterwarnings('ignore')
import sklearn, xgboost, rdkit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
z = np.load(os.path.join(ROOT, 'work', 'clean_features.npz'))
ydv, yte, yex = z['ydv'], z['yte'], z['yex']
SEEDS = [7, 21, 42, 84, 123]
tag = 'rdkit%s_sklearn%s_xgb%s' % (rdkit.__version__, sklearn.__version__, xgboost.__version__)
print('environment:', tag, flush=True)

def boot_paired(y, pa, pb, n=5000, seed=42):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

out = {'environment': tag, 'versions': {'rdkit': rdkit.__version__,
       'scikit-learn': sklearn.__version__, 'xgboost': xgboost.__version__,
       'numpy': np.__version__}}
for alg in ['RF', 'XGB']:
    preds = {}
    for rep, sfx in [('Morgan', 'm'), ('Descriptors', 'd'), ('Fused', 'c')]:
        ps, ri, re_ = [], [], []
        for sd in SEEDS:
            if alg == 'RF':
                m = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                           random_state=sd, n_jobs=-1)
            else:
                m = XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.8,
                                  colsample_bytree=0.7, min_child_weight=2, gamma=0.05,
                                  reg_alpha=0.1, reg_lambda=1.0, objective='binary:logistic',
                                  eval_metric='logloss', random_state=sd, n_jobs=-1)
            m.fit(z['Xdv_' + sfx], ydv)
            pe = m.predict_proba(z['Xex_' + sfx])[:, 1]
            ps.append(pe); re_.append(roc_auc_score(yex, pe))
            ri.append(roc_auc_score(yte, m.predict_proba(z['Xte_' + sfx])[:, 1]))
        preds[rep] = np.mean(ps, axis=0)
        out.setdefault(alg, {})[rep] = {'internal_mean': float(np.mean(ri)),
                                        'external_mean': float(np.mean(re_)),
                                        'external_sd': float(np.std(re_, ddof=1)),
                                        'external_ensemble': float(roc_auc_score(yex, preds[rep]))}
        print('  %-4s %-12s external ROC-AUC %.4f ± %.4f' % (alg, rep, np.mean(re_), np.std(re_, ddof=1)), flush=True)
    for base in ['Morgan', 'Descriptors']:
        d, lo, hi = boot_paired(yex, preds['Fused'], preds[base])
        out[alg]['fusion_vs_' + base] = {'delta': d, 'ci': [lo, hi], 'separable': bool(lo > 0 or hi < 0)}
        print('    fusion vs %-12s %+.4f [%+.4f, %+.4f] %s' % (base, d, lo, hi,
              'separable' if (lo > 0 or hi < 0) else 'not separable'), flush=True)

dest = os.path.join(ROOT, 'results', 'environment_%s.json' % ('B' if sklearn.__version__ >= '1.9' else 'A'))
json.dump(out, open(dest, 'w'), indent=1)
print('\nsaved', dest)
