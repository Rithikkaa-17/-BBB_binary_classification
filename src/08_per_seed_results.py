#!/usr/bin/env python3
"""Per-seed internal and external ROC-AUC for all six configurations.

Reproduces exactly the protocol of final_results.py and records the per-seed
values that were previously summarised only as mean and standard deviation.
"""
import warnings, json, numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from xgboost import XGBClassifier

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
D = _os.path.join(_ROOT, 'work') + _os.sep
DATA = _os.path.join(_ROOT, 'data') + _os.sep
RESULTS = _os.path.join(_ROOT, 'results') + _os.sep
z = np.load(D + 'clean_features.npz')
ytr, yva, yte, ydv, yex = z['ytr'], z['yva'], z['yte'], z['ydv'], z['yex']
SEEDS = [7, 21, 42, 84, 123]
REPS = {'Morgan': 'm', 'Descriptors': 'd', 'Fused': 'c'}


def make(alg, seed):
    if alg == 'RF':
        return RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                      random_state=seed, n_jobs=-1)
    return XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.8,
                         colsample_bytree=0.7, min_child_weight=2, gamma=0.05,
                         reg_alpha=0.1, reg_lambda=1.0, objective='binary:logistic',
                         eval_metric='logloss', random_state=seed, n_jobs=-1)


def pick_threshold(y, p):
    grid = np.arange(0.05, 0.96, 0.01)
    return float(grid[int(np.argmax([matthews_corrcoef(y, (p >= t).astype(int))
                                     for t in grid]))])


OUT = {}
print('%-20s %-42s %-42s' % ('configuration', 'internal ROC per seed', 'external ROC per seed'))
for alg in ['RF', 'XGB']:
    for rep, sfx in REPS.items():
        name = '%s + %s' % (alg, rep)
        int_s, ext_s, thr_s = [], [], []
        p_te_s, p_ex_s = [], []
        for sd in SEEDS:
            m1 = make(alg, sd).fit(z['Xtr_' + sfx], ytr)
            thr_s.append(pick_threshold(yva, m1.predict_proba(z['Xva_' + sfx])[:, 1]))
            m2 = make(alg, sd).fit(z['Xdv_' + sfx], ydv)
            pte = m2.predict_proba(z['Xte_' + sfx])[:, 1]
            pex = m2.predict_proba(z['Xex_' + sfx])[:, 1]
            p_te_s.append(pte); p_ex_s.append(pex)
            int_s.append(float(roc_auc_score(yte, pte)))
            ext_s.append(float(roc_auc_score(yex, pex)))
        ens_te = float(roc_auc_score(yte, np.mean(p_te_s, axis=0)))
        ens_ex = float(roc_auc_score(yex, np.mean(p_ex_s, axis=0)))
        OUT[name] = {
            'seeds': SEEDS,
            'internal_roc_per_seed': int_s,
            'external_roc_per_seed': ext_s,
            'internal_roc_mean': float(np.mean(int_s)),
            'internal_roc_sd': float(np.std(int_s, ddof=1)),
            'external_roc_mean': float(np.mean(ext_s)),
            'external_roc_sd': float(np.std(ext_s, ddof=1)),
            'internal_roc_ensemble': ens_te,
            'external_roc_ensemble': ens_ex,
            'threshold_per_seed': thr_s,
            'threshold_median': float(np.median(thr_s)),
        }
        print('%-20s %-42s %-42s' % (
            name, ' '.join('%.4f' % v for v in int_s), ' '.join('%.4f' % v for v in ext_s)))
        print('%-20s mean %.4f+/-%.4f ens %.4f        mean %.4f+/-%.4f ens %.4f  thr %s'
              % ('', np.mean(int_s), np.std(int_s, ddof=1), ens_te,
                 np.mean(ext_s), np.std(ext_s, ddof=1), ens_ex,
                 ','.join('%.2f' % t for t in thr_s)))

json.dump(OUT, open(RESULTS + 'per_seed_results.json', 'w'), indent=1)
print('\nsaved per_seed_results.json')
