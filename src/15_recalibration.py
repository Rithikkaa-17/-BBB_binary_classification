#!/usr/bin/env python3
"""Reviewer 1, point 2: how much does a target-population recalibration step
actually recover? Platt scaling and isotonic regression are fitted on a held-out
slice of the external collection and applied to the remainder.
"""
import warnings, json, os, numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
z = np.load(os.path.join(ROOT, 'work', 'clean_features.npz'))
ydv, yex = z['ydv'], z['yex']
SEEDS = [7, 21, 42, 84, 123]

def ece(y, p, bins=10):
    e, n = 0.0, len(y); edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i+1]) if i else (p >= edges[i]) & (p <= edges[i+1])
        if m.sum(): e += m.sum()/n * abs(y[m].mean() - p[m].mean())
    return float(e)

P = []
for sd in SEEDS:
    m = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                               random_state=sd, n_jobs=-1).fit(z['Xdv_c'], ydv)
    P.append(m.predict_proba(z['Xex_c'])[:, 1])
pex = np.mean(P, axis=0)

out = {'uncalibrated': {'brier': float(brier_score_loss(yex, pex)), 'ece': ece(yex, pex),
                        'roc': float(roc_auc_score(yex, pex)),
                        'mean_pred': float(pex.mean()), 'observed': float(yex.mean())}}
print('uncalibrated   Brier %.4f  ECE %.4f  ROC %.4f  mean pred %.3f vs observed %.3f'
      % (out['uncalibrated']['brier'], out['uncalibrated']['ece'], out['uncalibrated']['roc'],
         pex.mean(), yex.mean()), flush=True)

# 5-fold: fit the calibrator on 4/5 of the external set, score the held-out 1/5
for name in ['platt', 'isotonic']:
    pc = np.zeros_like(pex)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(pex, yex):
        if name == 'platt':
            c = LogisticRegression(C=1e6).fit(pex[tr].reshape(-1, 1), yex[tr])
            pc[va] = c.predict_proba(pex[va].reshape(-1, 1))[:, 1]
        else:
            c = IsotonicRegression(out_of_bounds='clip').fit(pex[tr], yex[tr])
            pc[va] = c.predict(pex[va])
    out[name] = {'brier': float(brier_score_loss(yex, pc)), 'ece': ece(yex, pc),
                 'roc': float(roc_auc_score(yex, pc)), 'mean_pred': float(pc.mean())}
    print('%-14s Brier %.4f  ECE %.4f  ROC %.4f  mean pred %.3f'
          % (name, out[name]['brier'], out[name]['ece'], out[name]['roc'], pc.mean()), flush=True)

# how little data does the calibrator need?
print('\ncalibration-set size sweep (isotonic, 20 repeats, scored on the rest):', flush=True)
rng = np.random.default_rng(0)
for n in [50, 100, 200, 500]:
    eces, briers = [], []
    for _ in range(20):
        idx = rng.permutation(len(yex)); tr, va = idx[:n], idx[n:]
        if len(np.unique(yex[tr])) < 2: continue
        c = IsotonicRegression(out_of_bounds='clip').fit(pex[tr], yex[tr])
        q = c.predict(pex[va])
        eces.append(ece(yex[va], q)); briers.append(brier_score_loss(yex[va], q))
    out.setdefault('size_sweep', {})[str(n)] = {'ece_mean': float(np.mean(eces)), 'ece_sd': float(np.std(eces, ddof=1)),
                                                'brier_mean': float(np.mean(briers))}
    print('  n=%4d  ECE %.4f ± %.4f  Brier %.4f' % (n, np.mean(eces), np.std(eces, ddof=1), np.mean(briers)), flush=True)

json.dump(out, open(os.path.join(ROOT, 'results', 'recalibration.json'), 'w'), indent=1)
print('\nsaved results/recalibration.json')
