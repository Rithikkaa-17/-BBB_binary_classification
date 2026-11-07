#!/usr/bin/env python3
"""Reviewer 2, point 2: is the fusion effect an artefact of feature subsampling,
and does it survive a learner that is not a tree ensemble?

Sweeps the fraction of columns a split may consider (random forest) and the
column-sampling rate (gradient boosting), and adds a support vector machine with
a Tanimoto kernel, which has no feature-subsampling step at all.
"""
import warnings, json, os, numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.special import comb
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
z = np.load(os.path.join(ROOT, 'work', 'clean_features.npz'))
ydv, yex = z['ydv'], z['yex']
SEEDS = [7, 21, 42, 84, 123]
out = {}

def boot_paired(y, pa, pb, n=5000, seed=42):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

# --- how often can a split even see a descriptor? ---
print('=== probability a split considers at least one descriptor (2,058 columns, 10 descriptors) ===')
for lab, k in [('sqrt', int(np.sqrt(2058))), ('log2', int(np.log2(2058))),
               ('0.1', int(0.1 * 2058)), ('0.3', int(0.3 * 2058))]:
    p = 1 - comb(2048, k, exact=True) / comb(2058, k, exact=True)
    out.setdefault('split_visibility', {})[lab] = {'k': int(k), 'p_at_least_one_descriptor': float(p)}
    print('  max_features=%-5s  k=%4d  P = %.4f' % (lab, k, p), flush=True)

# --- random forest sweep ---
print('\n=== random forest: external ROC-AUC by max_features (5 seeds) ===', flush=True)
for lab, mf in [('sqrt', 'sqrt'), ('log2', 'log2'), ('0.1', 0.1), ('0.3', 0.3)]:
    row, preds = {}, {}
    for rep, sfx in [('Morgan', 'm'), ('Descriptors', 'd'), ('Fused', 'c')]:
        ps, rocs = [], []
        for sd in SEEDS:
            m = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                       max_features=mf, random_state=sd, n_jobs=-1).fit(z['Xdv_' + sfx], ydv)
            p = m.predict_proba(z['Xex_' + sfx])[:, 1]; ps.append(p); rocs.append(roc_auc_score(yex, p))
        preds[rep] = np.mean(ps, axis=0)
        row[rep] = {'per_seed': rocs, 'mean': float(np.mean(rocs)), 'sd': float(np.std(rocs, ddof=1))}
    for base in ['Morgan', 'Descriptors']:
        d, lo, hi = boot_paired(yex, preds['Fused'], preds[base])
        row['fusion_vs_' + base] = {'delta': d, 'ci': [lo, hi], 'separable': bool(lo > 0 or hi < 0)}
    out.setdefault('rf_max_features', {})[lab] = row
    print('  max_features=%-5s Morgan %.4f Desc %.4f Fused %.4f | fusion-vs-Morgan %+.4f [%+.4f,%+.4f] %s'
          % (lab, row['Morgan']['mean'], row['Descriptors']['mean'], row['Fused']['mean'],
             row['fusion_vs_Morgan']['delta'], *row['fusion_vs_Morgan']['ci'],
             'separable' if row['fusion_vs_Morgan']['separable'] else 'not separable'), flush=True)

# --- gradient boosting sweep ---
print('\n=== gradient boosting: external ROC-AUC by colsample_bytree (5 seeds) ===', flush=True)
for cs in [0.3, 0.7, 1.0]:
    row, preds = {}, {}
    for rep, sfx in [('Morgan', 'm'), ('Descriptors', 'd'), ('Fused', 'c')]:
        ps, rocs = [], []
        for sd in SEEDS:
            m = XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.8,
                              colsample_bytree=cs, min_child_weight=2, gamma=0.05, reg_alpha=0.1,
                              reg_lambda=1.0, objective='binary:logistic', eval_metric='logloss',
                              random_state=sd, n_jobs=-1).fit(z['Xdv_' + sfx], ydv)
            p = m.predict_proba(z['Xex_' + sfx])[:, 1]; ps.append(p); rocs.append(roc_auc_score(yex, p))
        preds[rep] = np.mean(ps, axis=0)
        row[rep] = {'per_seed': rocs, 'mean': float(np.mean(rocs)), 'sd': float(np.std(rocs, ddof=1))}
    for base in ['Morgan', 'Descriptors']:
        d, lo, hi = boot_paired(yex, preds['Fused'], preds[base])
        row['fusion_vs_' + base] = {'delta': d, 'ci': [lo, hi], 'separable': bool(lo > 0 or hi < 0)}
    out.setdefault('xgb_colsample', {})[str(cs)] = row
    print('  colsample=%-4s Morgan %.4f Desc %.4f Fused %.4f | fusion-vs-Morgan %+.4f [%+.4f,%+.4f] %s'
          % (cs, row['Morgan']['mean'], row['Descriptors']['mean'], row['Fused']['mean'],
             row['fusion_vs_Morgan']['delta'], *row['fusion_vs_Morgan']['ci'],
             'separable' if row['fusion_vs_Morgan']['separable'] else 'not separable'), flush=True)

# --- support vector machine with a Tanimoto kernel (no feature subsampling) ---
def tanimoto(A, B):
    A = A.astype(np.float64); B = B.astype(np.float64)
    num = A @ B.T
    a = (A * A).sum(1)[:, None]; b = (B * B).sum(1)[None, :]
    return num / (a + b - num + 1e-9)

print('\n=== support vector machine, Tanimoto kernel (no feature subsampling) ===', flush=True)
svm_row, svm_preds = {}, {}
for rep, sfx in [('Morgan', 'm'), ('Descriptors', 'd'), ('Fused', 'c')]:
    Xd, Xe = z['Xdv_' + sfx].astype(np.float64), z['Xex_' + sfx].astype(np.float64)
    if sfx in ('d', 'c'):
        # scale so the 10 real-valued columns cannot dominate the kernel by magnitude
        sc = StandardScaler().fit(Xd[:, -10:] if sfx == 'c' else Xd)
        if sfx == 'c':
            Xd = np.hstack([Xd[:, :2048], sc.transform(Xd[:, -10:])])
            Xe = np.hstack([Xe[:, :2048], sc.transform(Xe[:, -10:])])
        else:
            Xd, Xe = sc.transform(Xd), sc.transform(Xe)
    Kd = tanimoto(Xd, Xd); Ke = tanimoto(Xe, Xd)
    ps, rocs = [], []
    for sd in SEEDS:
        m = SVC(C=1.0, kernel='precomputed', class_weight='balanced',
                probability=True, random_state=sd).fit(Kd, ydv)
        p = m.predict_proba(Ke)[:, 1]; ps.append(p); rocs.append(roc_auc_score(yex, p))
    svm_preds[rep] = np.mean(ps, axis=0)
    svm_row[rep] = {'per_seed': rocs, 'mean': float(np.mean(rocs)), 'sd': float(np.std(rocs, ddof=1))}
    print('  %-12s external ROC-AUC %.4f ± %.4f' % (rep, svm_row[rep]['mean'], svm_row[rep]['sd']), flush=True)
for base in ['Morgan', 'Descriptors']:
    d, lo, hi = boot_paired(yex, svm_preds['Fused'], svm_preds[base])
    svm_row['fusion_vs_' + base] = {'delta': d, 'ci': [lo, hi], 'separable': bool(lo > 0 or hi < 0)}
    print('  fusion vs %-12s %+.4f [%+.4f, %+.4f] %s' % (base, d, lo, hi,
          'separable' if (lo > 0 or hi < 0) else 'not separable'), flush=True)
out['svm_tanimoto'] = svm_row

json.dump(out, open(os.path.join(ROOT, 'results', 'subsampling_and_learners.json'), 'w'), indent=1)
print('\nsaved results/subsampling_and_learners.json')
