#!/usr/bin/env python3
"""SHAP attribution and calibration for the two Pareto-optimal configurations."""
import warnings, json, numpy as np
warnings.filterwarnings('ignore')
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
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
DESC = ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds',
        'RingCount', 'AromaticRings', 'FractionCSP3', 'HeavyAtomCount']
FEAT = ['Morgan_%d' % i for i in range(2048)] + DESC
R = {}


def ece(y, p, bins=10):
    e, n = 0.0, len(y)
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum():
            e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(e)


for name, mdl in [('RF + Fused', RandomForestClassifier(n_estimators=500,
                                                        class_weight='balanced',
                                                        random_state=42, n_jobs=-1)),
                  ('XGB + Fused', XGBClassifier(n_estimators=600, max_depth=5,
                                                learning_rate=0.04, subsample=0.8,
                                                colsample_bytree=0.7, min_child_weight=2,
                                                gamma=0.05, reg_alpha=0.1, reg_lambda=1.0,
                                                objective='binary:logistic',
                                                eval_metric='logloss', random_state=42,
                                                n_jobs=-1))]:
    print('=' * 60); print(name)
    m = mdl.fit(z['Xdv_c'], ydv)
    p_te = m.predict_proba(z['Xte_c'])[:, 1]
    p_ex = m.predict_proba(z['Xex_c'])[:, 1]

    # ---- calibration ----
    for lab, y_, p_ in [('internal', yte, p_te), ('external', yex, p_ex)]:
        R.setdefault('calibration', {}).setdefault(name, {})[lab] = {
            'brier': float(brier_score_loss(y_, p_)), 'ece': ece(y_, p_),
            'mean_pred': float(p_.mean()), 'observed': float(y_.mean())}
        print('  %-9s Brier %.4f  ECE %.4f  mean pred %.3f vs observed %.3f'
              % (lab, brier_score_loss(y_, p_), ece(y_, p_), p_.mean(), y_.mean()))

    # ---- SHAP on the internal test set ----
    # interventional perturbation against an explicit background avoids the
    # additivity failure that class-weighted forests cause under the
    # tree_path_dependent estimator, and yields exact Shapley values w.r.t.
    # the background distribution.
    rng = np.random.default_rng(42)
    bg = z['Xdv_c'][rng.choice(len(ydv), 200, replace=False)]
    ex = shap.TreeExplainer(m, data=bg, feature_perturbation='interventional',
                            model_output='probability')
    sv = ex.shap_values(z['Xte_c'], check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    ma = np.abs(sv).mean(axis=0)
    order = np.argsort(ma)[::-1][:15]
    print('  top features:', ', '.join('%s %.4f' % (FEAT[i], ma[i]) for i in order[:6]))
    R.setdefault('shap_top', {})[name] = [
        {'feature': FEAT[i], 'mean_abs': float(ma[i]),
         'is_descriptor': not FEAT[i].startswith('Morgan_')} for i in order]

    # direction of effect for each descriptor
    dirs = {}
    for k, dn in enumerate(DESC):
        fv = z['Xte_c'][:, 2048 + k]
        s = sv[:, 2048 + k]
        r = float(np.corrcoef(fv, s)[0, 1]) if fv.std() > 0 else 0.0
        dirs[dn] = {'r': r, 'mean_abs': float(np.abs(s).mean()),
                    'direction': 'higher value favours non-permeable' if r < 0
                    else 'higher value favours permeable'}
    R.setdefault('shap_direction', {})[name] = dirs
    print('  directions:', ', '.join('%s r=%+.2f' % (k, v['r']) for k, v in
                                     sorted(dirs.items(), key=lambda x: -abs(x[1]['r']))[:5]))
    np.save(D + 'sv_%s.npy' % name.replace(' + ', '_'), sv)
    np.savez(D + 'calib_%s.npz' % name.replace(' + ', '_'),
             y_te=yte, p_te=p_te, y_ex=yex, p_ex=p_ex)

json.dump(R, open(D + 'shap_calib.json', 'w'), indent=1)
print('\nsaved shap_calib.json')
