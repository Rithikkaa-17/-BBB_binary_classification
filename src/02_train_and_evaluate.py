#!/usr/bin/env python3
"""Definitive result set for the revised manuscript, single pinned environment.

Protocol (explicit, per Reviewer 4):
  train  (1,632) -> fit
  valid  (159)   -> decision-threshold selection ONLY
  dev    (1,791) -> refit with identical hyperparameters
  test   (248)   -> evaluated once
  external(2,201)-> evaluated once
"""
import warnings, json, itertools, numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, matthews_corrcoef,
                             accuracy_score, precision_score, recall_score, f1_score,
                             balanced_accuracy_score, brier_score_loss, confusion_matrix)
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
ytr, yva, yte, ydv, yex = z['ytr'], z['yva'], z['yte'], z['ydv'], z['yex']
in_dev = z['ext_in_dev'].astype(bool)
in_bbbp = z['ext_in_bbbp'].astype(bool)
SEEDS = [7, 21, 42, 84, 123]
REPS = {'Morgan': 'm', 'Descriptors': 'd', 'Fused': 'c'}
OUT = {}


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
    mccs = [matthews_corrcoef(y, (p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(mccs))]), float(np.max(mccs))


def full(y, p, thr):
    pred = (p >= thr).astype(int)
    return {'Accuracy': accuracy_score(y, pred),
            'BalancedAccuracy': balanced_accuracy_score(y, pred),
            'Precision': precision_score(y, pred, zero_division=0),
            'Recall': recall_score(y, pred, zero_division=0),
            'F1': f1_score(y, pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y, p), 'PR-AUC': average_precision_score(y, p),
            'MCC': matthews_corrcoef(y, pred), 'Brier': brier_score_loss(y, p),
            'ConfusionMatrix': confusion_matrix(y, pred).tolist(), 'threshold': thr}


def boot(y, p, fn, n=5000, seed=42):
    rng = np.random.default_rng(seed); o = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        o.append(fn(y[i], p[i]))
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def paired(y, pa, pb, fn=roc_auc_score, n=5000, seed=42):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(fn(y[i], pa[i]) - fn(y[i], pb[i]))
    d = np.array(d)
    return (float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
            float(min(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0)))


models, thresholds = {}, {}
print('%-20s %-28s %-28s' % ('configuration', 'internal test', 'external'))
for alg in ['RF', 'XGB']:
    for rep, sfx in REPS.items():
        name = '%s + %s' % (alg, rep)
        p_te_s, p_ex_s, thr_s, roc_te_s, roc_ex_s = [], [], [], [], []
        for sd in SEEDS:
            m1 = make(alg, sd).fit(z['Xtr_' + sfx], ytr)
            thr, _ = pick_threshold(yva, m1.predict_proba(z['Xva_' + sfx])[:, 1])
            thr_s.append(thr)
            m2 = make(alg, sd).fit(z['Xdv_' + sfx], ydv)
            pte = m2.predict_proba(z['Xte_' + sfx])[:, 1]
            pex = m2.predict_proba(z['Xex_' + sfx])[:, 1]
            p_te_s.append(pte); p_ex_s.append(pex)
            roc_te_s.append(roc_auc_score(yte, pte)); roc_ex_s.append(roc_auc_score(yex, pex))
        pte = np.mean(p_te_s, axis=0); pex = np.mean(p_ex_s, axis=0)
        thr = float(np.median(thr_s))
        models[name] = (pte, pex); thresholds[name] = thr
        mt, me = full(yte, pte, thr), full(yex, pex, thr)
        OUT.setdefault('models', {})[name] = {
            'threshold_per_seed': thr_s, 'threshold': thr,
            'internal': mt, 'external': me,
            'internal_roc_seed_mean': float(np.mean(roc_te_s)),
            'internal_roc_seed_sd': float(np.std(roc_te_s, ddof=1)),
            'external_roc_seed_mean': float(np.mean(roc_ex_s)),
            'external_roc_seed_sd': float(np.std(roc_ex_s, ddof=1)),
            'external_roc_novel_vs_dev': float(roc_auc_score(yex[~in_dev], pex[~in_dev])),
            'external_roc_overlap_dev': float(roc_auc_score(yex[in_dev], pex[in_dev])),
        }
        print('%-20s ROC %.4f+/-%.4f PR %.4f | ROC %.4f+/-%.4f PR %.4f MCC %.4f thr %.2f'
              % (name, np.mean(roc_te_s), np.std(roc_te_s, ddof=1), mt['PR-AUC'],
                 np.mean(roc_ex_s), np.std(roc_ex_s, ddof=1), me['PR-AUC'], me['MCC'], thr))

# confidence intervals for the two Pareto models
for name in ['RF + Fused', 'XGB + Fused']:
    pte, pex = models[name]
    OUT['models'][name]['internal_ci'] = {
        'ROC-AUC': boot(yte, pte, roc_auc_score),
        'PR-AUC': boot(yte, pte, average_precision_score)}
    OUT['models'][name]['external_ci'] = {
        'ROC-AUC': boot(yex, pex, roc_auc_score),
        'PR-AUC': boot(yex, pex, average_precision_score)}
    for lab, mask in [('novel_vs_dev', ~in_dev), ('overlap_dev', in_dev),
                      ('novel_vs_bbbp', ~in_bbbp), ('overlap_bbbp', in_bbbp)]:
        OUT['models'][name].setdefault('strata', {})[lab] = {
            'n': int(mask.sum()), 'prevalence': float(yex[mask].mean()),
            'roc': float(roc_auc_score(yex[mask], pex[mask])),
            'ci': boot(yex[mask], pex[mask], roc_auc_score)}

print('\nPAIRED TESTS (external ROC-AUC)')
for a, b in itertools.combinations(models, 2):
    d, lo, hi, p = paired(yex, models[a][1], models[b][1])
    OUT.setdefault('pairwise_external', {})['%s vs %s' % (a, b)] = {
        'delta': d, 'ci': [lo, hi], 'p': p, 'significant': bool(lo > 0 or hi < 0)}
    print('  %-18s vs %-18s %+.4f (%+.4f,%+.4f) p=%.4f %s'
          % (a, b, d, lo, hi, p, 'SIG' if (lo > 0 or hi < 0) else 'ns'))

print('\nPAIRED TESTS (internal test ROC-AUC)')
for a, b in itertools.combinations(models, 2):
    d, lo, hi, p = paired(yte, models[a][0], models[b][0])
    OUT.setdefault('pairwise_internal', {})['%s vs %s' % (a, b)] = {
        'delta': d, 'ci': [lo, hi], 'p': p, 'significant': bool(lo > 0 or hi < 0)}
    print('  %-18s vs %-18s %+.4f (%+.4f,%+.4f) p=%.4f %s'
          % (a, b, d, lo, hi, p, 'SIG' if (lo > 0 or hi < 0) else 'ns'))

OUT['protocol'] = {'train': int(len(ytr)), 'valid': int(len(yva)), 'dev': int(len(ydv)),
                   'test': int(len(yte)), 'external': int(len(yex)), 'seeds': SEEDS}
json.dump(OUT, open(D + 'final_results.json', 'w'), indent=1)
np.savez(D + 'final_preds.npz',
         **{('p_' + k.replace(' + ', '_').replace(' ', '')): v[0] for k, v in models.items()},
         **{('e_' + k.replace(' + ', '_').replace(' ', '')): v[1] for k, v in models.items()},
         yte=yte, yex=yex, in_dev=in_dev, in_bbbp=in_bbbp)
print('\nsaved final_results.json')
