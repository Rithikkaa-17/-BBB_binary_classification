#!/usr/bin/env python3
"""New analyses requested by reviewers.

R1-2  Tanimoto similarity between datasets
R1-4  Calibration (Brier, ECE, reliability curve)
R3-2  External stratified by development-set scaffold overlap, with bootstrap CIs
R3-3  External stratified by B3DB source group (label heterogeneity)
R4-5  Class-weighted variant (scale_pos_weight)
R4-9  Per-model seed standard deviations
R2    Pairwise significance tests between tree-based models
"""
import warnings, json, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.metrics import (roc_auc_score, average_precision_score, matthews_corrcoef,
                             brier_score_loss, accuracy_score, f1_score)
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
RDLogger.DisableLog('rdApp.*')

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
D = _os.path.join(_ROOT, 'work') + _os.sep        # intermediate artefacts
DATA = _os.path.join(_ROOT, 'data') + _os.sep     # partition assignments
RESULTS = _os.path.join(_ROOT, 'results') + _os.sep
OUT = _os.path.join(_ROOT, 'figures') + _os.sep
for _p in (D, RESULTS, OUT):
    _os.makedirs(_p, exist_ok=True)
z = np.load(D + 'clean_features.npz')
S = json.load(open(D + 'clean_summary.json'))
THR = 0.86
RES = {}


def xgb(seed=42, spw=None):
    kw = dict(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.8,
              colsample_bytree=0.7, min_child_weight=2, gamma=0.05, reg_alpha=0.1,
              reg_lambda=1.0, objective='binary:logistic', eval_metric='logloss',
              random_state=seed, n_jobs=-1)
    if spw is not None:
        kw['scale_pos_weight'] = spw
    return XGBClassifier(**kw)


def boot_ci(y, p, fn, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        out.append(fn(y[i], p[i]))
    return float(np.mean(out)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def paired_boot(y, pa, pb, fn, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(fn(y[i], pa[i]) - fn(y[i], pb[i]))
    d = np.array(d)
    p_two = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(min(p_two, 1.0))


ydv, yte, yex = z['ydv'], z['yte'], z['yex']

print('=' * 70)
print('R4-5  CLASS-WEIGHTED VARIANT')
print('=' * 70)
spw = (ydv == 0).sum() / (ydv == 1).sum()
print('scale_pos_weight = %.4f (neg/pos in development set)' % spw)
models = {}
for name, s_ in [('unweighted', None), ('weighted', spw)]:
    m = xgb(42, s_).fit(z['Xdv_c'], ydv)
    pt = m.predict_proba(z['Xte_c'])[:, 1]
    pe = m.predict_proba(z['Xex_c'])[:, 1]
    models[name] = (pt, pe)
    print('%-12s internal ROC %.4f PR %.4f MCC@0.86 %.4f | external ROC %.4f PR %.4f MCC@0.5 %.4f'
          % (name, roc_auc_score(yte, pt), average_precision_score(yte, pt),
             matthews_corrcoef(yte, (pt >= THR).astype(int)),
             roc_auc_score(yex, pe), average_precision_score(yex, pe),
             matthews_corrcoef(yex, (pe >= 0.5).astype(int))))
d, lo, hi, pv = paired_boot(yte, models['weighted'][0], models['unweighted'][0], roc_auc_score)
print('weighted - unweighted, internal ROC-AUC: %+.4f (95%% CI %+.4f, %+.4f), p=%.3f' % (d, lo, hi, pv))
d2, lo2, hi2, pv2 = paired_boot(yex, models['weighted'][1], models['unweighted'][1], roc_auc_score)
print('weighted - unweighted, external ROC-AUC: %+.4f (95%% CI %+.4f, %+.4f), p=%.3f' % (d2, lo2, hi2, pv2))
RES['class_weight'] = {'spw': float(spw), 'internal_delta': [d, lo, hi, pv],
                       'external_delta': [d2, lo2, hi2, pv2]}

print()
print('=' * 70)
print('R1-4  CALIBRATION')
print('=' * 70)
p_int, p_ext = models['unweighted']


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(y)
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum():
            e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return e


for nm, y_, p_ in [('internal', yte, p_int), ('external', yex, p_ext)]:
    print('%-9s Brier %.4f | ECE(10 bins) %.4f | mean predicted %.3f vs observed %.3f'
          % (nm, brier_score_loss(y_, p_), ece(y_, p_), p_.mean(), y_.mean()))
    RES.setdefault('calibration', {})[nm] = {
        'brier': float(brier_score_loss(y_, p_)), 'ece': float(ece(y_, p_)),
        'mean_pred': float(p_.mean()), 'observed': float(y_.mean())}
np.savez(D + 'calibration.npz', y_int=yte, p_int=p_int, y_ext=yex, p_ext=p_ext)

print()
print('=' * 70)
print('R3-2  EXTERNAL STRATIFIED BY DEVELOPMENT-SET SCAFFOLD OVERLAP')
print('=' * 70)
in_dev = z['ext_in_dev'].astype(bool)
in_bbbp = z['ext_in_bbbp'].astype(bool)
for lab, mask in [('all external', np.ones(len(yex), bool)),
                  ('scaffold in DEV', in_dev),
                  ('scaffold NOVEL vs DEV', ~in_dev),
                  ('scaffold in any BBBP', in_bbbp),
                  ('scaffold NOVEL vs BBBP', ~in_bbbp)]:
    m_, l_, h_ = boot_ci(yex[mask], p_ext[mask], roc_auc_score)
    print('%-24s n=%4d prevalence %.3f  ROC-AUC %.4f (95%% CI %.4f-%.4f)'
          % (lab, mask.sum(), yex[mask].mean(), roc_auc_score(yex[mask], p_ext[mask]), l_, h_))
    RES.setdefault('external_strata', {})[lab] = {
        'n': int(mask.sum()), 'prevalence': float(yex[mask].mean()),
        'roc': float(roc_auc_score(yex[mask], p_ext[mask])), 'ci': [l_, h_],
        'pr': float(average_precision_score(yex[mask], p_ext[mask]))}

print()
print('=' * 70)
print('R1-2  TANIMOTO SIMILARITY BETWEEN DATASETS')
print('=' * 70)


def fps(smiles_list):
    out = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            out.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
    return out


dev_csv = pd.concat([pd.read_csv(D + 'clean_train.csv'), pd.read_csv(D + 'clean_valid.csv')])
test_csv = pd.read_csv(D + 'clean_test.csv')
ext_csv = pd.read_csv(D + 'clean_external.csv')
fp_dev = fps(dev_csv['smiles'])
fp_test = fps(test_csv['smiles'])
fp_ext = fps(ext_csv['SMILES'])
print('fingerprints: dev %d | test %d | external %d' % (len(fp_dev), len(fp_test), len(fp_ext)))


def nearest_sim(query, ref):
    return np.array([max(DataStructs.BulkTanimotoSimilarity(q, ref)) for q in query])


sim_test = nearest_sim(fp_test, fp_dev)
sim_ext = nearest_sim(fp_ext, fp_dev)
sim_ext_dev_novel = sim_ext[~in_dev]
sim_ext_dev_share = sim_ext[in_dev]
for nm, a in [('internal test vs dev', sim_test), ('external vs dev', sim_ext),
              ('  external dev-novel', sim_ext_dev_novel),
              ('  external dev-overlap', sim_ext_dev_share)]:
    print('%-24s mean %.3f  median %.3f  IQR %.3f-%.3f  frac>=0.4 %.3f'
          % (nm, a.mean(), np.median(a), np.percentile(a, 25), np.percentile(a, 75), (a >= 0.4).mean()))
    RES.setdefault('tanimoto', {})[nm.strip()] = {
        'mean': float(a.mean()), 'median': float(np.median(a)),
        'q1': float(np.percentile(a, 25)), 'q3': float(np.percentile(a, 75)),
        'frac_ge_0.4': float((a >= 0.4).mean())}
np.savez(D + 'tanimoto.npz', sim_test=sim_test, sim_ext=sim_ext, in_dev=in_dev)

print()
print('=' * 70)
print('R3-3  EXTERNAL STRATIFIED BY B3DB SOURCE GROUP')
print('=' * 70)
grp = ext_csv['group'].fillna('unspecified').astype(str).values
for g in pd.Series(grp).value_counts().index[:6]:
    m = grp == g
    if m.sum() >= 50 and len(np.unique(yex[m])) > 1:
        print('%-28s n=%4d prevalence %.3f ROC-AUC %.4f'
              % (g[:28], m.sum(), yex[m].mean(), roc_auc_score(yex[m], p_ext[m])))
        RES.setdefault('source_strata', {})[g[:40]] = {
            'n': int(m.sum()), 'prevalence': float(yex[m].mean()),
            'roc': float(roc_auc_score(yex[m], p_ext[m]))}

print()
print('=' * 70)
print('R4-9 / R2  SEED VARIABILITY AND PAIRWISE SIGNIFICANCE (tree models)')
print('=' * 70)
seeds = [7, 21, 42, 84, 123]
arms = {'RF + Morgan': ('rf', 'Xdv_m', 'Xte_m'),
        'XGBoost + Morgan': ('xgb', 'Xdv_m', 'Xte_m'),
        'XGBoost + descriptors': ('xgb', 'Xdv_d', 'Xte_d'),
        'XGBoost + fused': ('xgb', 'Xdv_c', 'Xte_c')}
seed_roc, final_pred = {}, {}
for name, (kind, kd, kt) in arms.items():
    rs = []
    for sd in seeds:
        if kind == 'rf':
            m = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                       random_state=sd, n_jobs=-1).fit(z[kd], ydv)
        else:
            m = xgb(sd).fit(z[kd], ydv)
        p = m.predict_proba(z[kt])[:, 1]
        rs.append(roc_auc_score(yte, p))
        if sd == 42:
            final_pred[name] = p
    seed_roc[name] = rs
    print('%-24s ROC-AUC %.4f +/- %.4f   (seeds: %s)'
          % (name, np.mean(rs), np.std(rs, ddof=1), ', '.join('%.4f' % r for r in rs)))
    RES.setdefault('seed_sd', {})[name] = {'mean': float(np.mean(rs)),
                                           'sd': float(np.std(rs, ddof=1)),
                                           'values': [float(r) for r in rs]}

print('\npairwise paired-bootstrap tests on the internal test set (ROC-AUC):')
names = list(arms)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        d, lo, hi, pv = paired_boot(yte, final_pred[a], final_pred[b], roc_auc_score)
        sig = 'significant' if (lo > 0 or hi < 0) else 'not significant'
        print('  %-24s vs %-24s %+.4f (%+.4f, %+.4f) p=%.3f  %s'
              % (a, b, d, lo, hi, pv, sig))
        RES.setdefault('pairwise', {})['%s vs %s' % (a, b)] = {
            'delta': d, 'ci': [lo, hi], 'p': pv, 'significant': bool(lo > 0 or hi < 0)}

json.dump(RES, open(D + 'revision_results.json', 'w'), indent=1)
print('\nsaved revision_results.json')
