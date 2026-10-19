#!/usr/bin/env python3
"""Revised figure set. Bar charts replaced by dot-and-interval plots per ACS
guidance cited by Reviewer 2; fonts enlarged per Reviewer 3."""
import json, warnings, numpy as np
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
D = _os.path.join(_ROOT, 'work') + _os.sep        # intermediate artefacts
DATA = _os.path.join(_ROOT, 'data') + _os.sep     # partition assignments
RESULTS = _os.path.join(_ROOT, 'results') + _os.sep
OUT = _os.path.join(_ROOT, 'figures') + _os.sep
for _p in (D, RESULTS, OUT):
    _os.makedirs(_p, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 10,
                     'axes.labelsize': 10.5, 'axes.titlesize': 10.5, 'xtick.labelsize': 9.5,
                     'ytick.labelsize': 9.5, 'legend.fontsize': 9.5, 'axes.linewidth': 0.9,
                     'savefig.dpi': 600, 'figure.dpi': 150})

F = json.load(open(D + 'final_results.json'))
FC = json.load(open(D + 'fair_comparison.json'))
SC = json.load(open(D + 'shap_calib.json'))
P = np.load(D + 'final_preds.npz')
GREEN, BLUE, ORANGE, PURPLE, MUTED, INK, RED = ('#1d6b3a', '#1f7fc4', '#d1743a',
                                                '#7a4fa3', '#8b9399', '#22262a', '#b02a37')
ORDER = ['RF + Fused', 'XGB + Fused', 'RF + Morgan', 'XGB + Morgan',
         'RF + Descriptors', 'XGB + Descriptors']


def despine(ax, keep=('left', 'bottom')):
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(s in keep)


# ===================================================== Fig: model comparison
fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.4), sharey=True)
y = np.arange(len(ORDER))[::-1]
for ax, key, sdk, title in [
        (axes[0], 'internal_roc_seed_mean', 'internal_roc_seed_sd',
         '(a) Internal test set (n = 248)'),
        (axes[1], 'external_roc_seed_mean', 'external_roc_seed_sd',
         '(b) External set (n = 2,201)')]:
    vals = [F['models'][m][key] for m in ORDER]
    sds = [F['models'][m][sdk] for m in ORDER]
    cols = [GREEN if 'Fused' in m else (BLUE if 'Morgan' in m else ORANGE) for m in ORDER]
    ax.errorbar(vals, y, xerr=sds, fmt='o', ms=7, capsize=4, lw=1.3,
                ecolor=INK, mfc='white', mew=1.8, linestyle='none')
    for yi, v, c in zip(y, vals, cols):
        ax.plot(v, yi, 'o', ms=7, color=c, zorder=5)
        ax.text(v, yi + 0.26, '%.4f' % v, ha='center', fontsize=8.4, color=INK)
    ax.axvline(0.5, color=MUTED, lw=0.8, ls=':')
    ax.set_yticks(y); ax.set_yticklabels(ORDER)
    ax.set_xlabel('ROC-AUC (mean ± SD over 5 seeds)')
    ax.set_title(title, loc='left', color=INK)
    ax.grid(axis='x', alpha=0.2, lw=0.6); ax.set_axisbelow(True)
    despine(ax)
axes[0].set_xlim(0.78, 0.95); axes[1].set_xlim(0.68, 0.81)
fig.legend(handles=[Line2D([], [], marker='o', ls='none', color=GREEN, label='Fused'),
                    Line2D([], [], marker='o', ls='none', color=BLUE, label='Fingerprint only'),
                    Line2D([], [], marker='o', ls='none', color=ORANGE, label='Descriptors only')],
           loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=[0, 0.06, 1, 1], w_pad=1.2)
fig.savefig(OUT + 'R_Fig3_model_comparison.png', bbox_inches='tight', facecolor='white')
plt.close(fig)

# ===================================================== Fig: paired tests forest
pairs_ext = F['pairwise_external']
sel = [('RF + Morgan vs RF + Fused', 'RF: fused vs fingerprint', True),
       ('RF + Descriptors vs RF + Fused', 'RF: fused vs descriptors', True),
       ('XGB + Morgan vs XGB + Fused', 'XGB: fused vs fingerprint', True),
       ('XGB + Descriptors vs XGB + Fused', 'XGB: fused vs descriptors', True),
       ('RF + Fused vs XGB + Fused', 'Fused: RF vs XGB', False)]
fig, ax = plt.subplots(figsize=(6.4, 3.1))
yy = np.arange(len(sel))[::-1]
for yi, (k, lab, flip) in zip(yy, sel):
    v = pairs_ext[k]
    d, lo, hi = v['delta'], v['ci'][0], v['ci'][1]
    if flip:
        d, lo, hi = -d, -hi, -lo          # orient so positive = fusion advantage
    ax.errorbar(d, yi, xerr=[[d - lo], [hi - d]], fmt='o', ms=7, capsize=4, lw=1.5,
                color=GREEN if hi < 0 or lo > 0 else MUTED, capthick=1.3)
    ax.text(d, yi + 0.22, '%+.4f' % d, ha='center', fontsize=8.6, color=INK)
    ax.text(0.085, yi, 'p = %.4f' % v['p'] if v['p'] >= 1e-4 else 'p < 0.0001',
            fontsize=8, va='center', color='#555')
ax.axvline(0, color=RED, lw=1.1, ls='--')
ax.set_yticks(yy); ax.set_yticklabels([l for _, l, _ in sel])
ax.set_xlabel('Difference in external ROC-AUC (95% CI, 5,000 paired resamples)')
ax.set_xlim(-0.01, 0.105)
ax.grid(axis='x', alpha=0.2, lw=0.6); ax.set_axisbelow(True)
despine(ax)
fig.tight_layout()
fig.savefig(OUT + 'R_Fig4_paired_tests.png', bbox_inches='tight', facecolor='white')
plt.close(fig)

# ===================================================== Fig: generalisation strata
fig, ax = plt.subplots(figsize=(6.6, 3.2))
labels, vals, los, his, ns = [], [], [], [], []
m = F['models']['RF + Fused']
labels.append('Internal test'); vals.append(m['internal']['ROC-AUC'])
los.append(m['internal_ci']['ROC-AUC'][0]); his.append(m['internal_ci']['ROC-AUC'][1]); ns.append(248)
for lab, key in [('External: scaffold\nseen in development', 'overlap_dev'),
                 ('External: all', None),
                 ('External: scaffold\nnovel vs development', 'novel_vs_dev')]:
    if key is None:
        labels.append(lab); vals.append(m['external']['ROC-AUC'])
        los.append(m['external_ci']['ROC-AUC'][0]); his.append(m['external_ci']['ROC-AUC'][1])
        ns.append(2201)
    else:
        s = m['strata'][key]
        labels.append(lab); vals.append(s['roc'])
        los.append(s['ci'][0]); his.append(s['ci'][1]); ns.append(s['n'])
x = np.arange(len(labels))
cols = [GREEN, '#2f8f5b', BLUE, ORANGE]
for xi, v, lo, hi, c, n in zip(x, vals, los, his, cols, ns):
    ax.errorbar(xi, v, yerr=[[v - lo], [hi - v]], fmt='o', ms=9, color=c,
                capsize=5, lw=1.5, capthick=1.3)
    ax.text(xi, hi + 0.012, '%.4f' % v, ha='center', fontsize=9, color=INK)
    ax.text(xi, lo - 0.022, 'n = %d' % n, ha='center', fontsize=8.2, color='#555')
ax.axhline(0.5, color=MUTED, lw=0.9, ls=':')
ax.text(len(labels) - 0.5, 0.505, 'chance', fontsize=8, color=MUTED, ha='right')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.8)
ax.set_ylabel('ROC-AUC (95% CI)')
ax.set_ylim(0.48, 0.97)
ax.grid(axis='y', alpha=0.2, lw=0.6); ax.set_axisbelow(True)
despine(ax)
fig.tight_layout()
fig.savefig(OUT + 'R_Fig5_generalisation.png', bbox_inches='tight', facecolor='white')
plt.close(fig)

# ===================================================== Fig: ROC / PR / calibration
cal = np.load(D + 'calib_RF_Fused.npz')
fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.9))
ax = axes[0]
for y_, p_, lab, col, ls in [(cal['y_te'], cal['p_te'], 'Internal (n = 248)', GREEN, '-'),
                             (cal['y_ex'], cal['p_ex'], 'External (n = 2,201)', ORANGE, (0, (4, 2)))]:
    fpr, tpr, _ = roc_curve(y_, p_)
    ax.plot(fpr, tpr, color=col, lw=1.9, ls=ls, label='%s\nAUC %.4f' % (lab, roc_auc_score(y_, p_)))
ax.plot([0, 1], [0, 1], color=MUTED, lw=0.9, ls=':')
ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
ax.set_title('(a) ROC', loc='left', color=INK)
ax.legend(frameon=False, loc='lower right', fontsize=8)
ax.grid(alpha=0.18, lw=0.6); ax.set_axisbelow(True); despine(ax)

ax = axes[1]
for y_, p_, lab, col, ls in [(cal['y_te'], cal['p_te'], 'Internal', GREEN, '-'),
                             (cal['y_ex'], cal['p_ex'], 'External', ORANGE, (0, (4, 2)))]:
    pr, rc, _ = precision_recall_curve(y_, p_)
    ax.plot(rc, pr, color=col, lw=1.9, ls=ls,
            label='%s AP %.4f' % (lab, average_precision_score(y_, p_)))
    ax.axhline(y_.mean(), color=col, lw=0.8, ls=':')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_ylim(0.3, 1.02)
ax.set_title('(b) Precision–recall', loc='left', color=INK)
ax.legend(frameon=False, loc='lower left', fontsize=8)
ax.grid(alpha=0.18, lw=0.6); ax.set_axisbelow(True); despine(ax)

ax = axes[2]
for y_, p_, lab, col, mk in [(cal['y_te'], cal['p_te'], 'Internal', GREEN, 'o'),
                             (cal['y_ex'], cal['p_ex'], 'External', ORANGE, 's')]:
    frac, mean_pred = calibration_curve(y_, p_, n_bins=8, strategy='quantile')
    ax.plot(mean_pred, frac, marker=mk, color=col, lw=1.6, ms=6, label=lab)
ax.plot([0, 1], [0, 1], color=MUTED, lw=0.9, ls=':')
ax.text(0.52, 0.42, 'perfect', color=MUTED, fontsize=8, rotation=37)
ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed frequency')
ax.set_title('(c) Calibration', loc='left', color=INK)
ax.legend(frameon=False, loc='upper left', fontsize=8)
ax.grid(alpha=0.18, lw=0.6); ax.set_axisbelow(True); despine(ax)
fig.tight_layout(pad=0.5, w_pad=1.4)
fig.savefig(OUT + 'R_Fig6_roc_pr_calibration.png', bbox_inches='tight', facecolor='white')
plt.close(fig)

# ===================================================== Fig: SRD + Pareto
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.3, 3.2))
srd = FC['srd']; srd_max = FC['srd_max']
names = sorted(srd, key=srd.get)
yy = np.arange(len(names))[::-1]
for yi, n in zip(yy, names):
    pct = 100 * srd[n] / srd_max
    a1.plot(pct, yi, 'o', ms=8, color=GREEN if 'Fused' in n else MUTED)
    a1.plot([0, pct], [yi, yi], color=GREEN if 'Fused' in n else MUTED, lw=1.1, alpha=0.5)
    a1.text(pct + 1.6, yi, '%.1f' % pct, va='center', fontsize=8.4, color=INK)
a1.axvline(100 * FC['srd_null_p5'] / srd_max, color=RED, ls='--', lw=1.1)
a1.text(100 * FC['srd_null_p5'] / srd_max + 1, len(names) - 1.4,
        'random-ranking\n5th percentile', fontsize=7.6, color=RED)
a1.set_yticks(yy); a1.set_yticklabels(names, fontsize=9)
a1.set_xlabel('SRD (% of maximum)'); a1.set_xlim(0, 86)
a1.set_title('(a) Sum of ranking differences', loc='left', color=INK)
a1.grid(axis='x', alpha=0.2, lw=0.6); a1.set_axisbelow(True); despine(a1)

rows = FC['rows']
front = set(FC['pareto_front'])
for n, v in rows.items():
    on = n in front
    a2.plot(v['ext_roc_mean'], v['ext_mcc'], 'o', ms=10 if on else 7,
            color=GREEN if on else MUTED, mfc=GREEN if on else 'white',
            mew=1.6, zorder=5 if on else 3)
    off = {'RF + Fused': (0, 12), 'XGB + Fused': (0, 12),
           'RF + Morgan': (0, -15), 'XGB + Morgan': (-4, -15),
           'RF + Descriptors': (34, -4), 'XGB + Descriptors': (38, 4)}[n]
    a2.annotate(n, (v['ext_roc_mean'], v['ext_mcc']), textcoords='offset points',
                xytext=off, ha='center', fontsize=7.6,
                color=INK if on else '#666')
fx = sorted([rows[n] for n in front], key=lambda r: r['ext_roc_mean'])
a2.plot([r['ext_roc_mean'] for r in fx], [r['ext_mcc'] for r in fx],
        color=GREEN, lw=1.2, ls='--', alpha=0.7)
a2.set_xlabel('External ROC-AUC'); a2.set_ylabel('External MCC')
a2.set_xlim(0.710, 0.800); a2.set_ylim(0.245, 0.368)
a2.set_title('(b) Pareto front', loc='left', color=INK)
a2.grid(alpha=0.2, lw=0.6); a2.set_axisbelow(True); despine(a2)
fig.tight_layout(pad=0.5, w_pad=1.6)
fig.savefig(OUT + 'R_Fig7_srd_pareto.png', bbox_inches='tight', facecolor='white')
plt.close(fig)

# ===================================================== Fig: Tanimoto
T = np.load(D + 'tanimoto.npz')
fig, ax = plt.subplots(figsize=(6.2, 3.1))
for a, lab, col, ls in [(T['sim_test'], 'Internal test vs development', GREEN, '-'),
                        (T['sim_ext'][T['in_dev']], 'External, scaffold seen', BLUE, (0, (4, 2))),
                        (T['sim_ext'][~T['in_dev']], 'External, scaffold novel', ORANGE, (0, (1.6, 1.6)))]:
    xs = np.sort(a); ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.plot(xs, ys, color=col, lw=1.9, ls=ls, label='%s\nmedian %.3f' % (lab, np.median(a)))
ax.axvline(0.4, color=MUTED, lw=0.9, ls=':')
ax.text(0.405, 0.06, 'Tanimoto 0.4', fontsize=8, color=MUTED)
ax.set_xlabel('Nearest-neighbour Tanimoto similarity to the development set')
ax.set_ylabel('Cumulative fraction')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(frameon=False, loc='lower right', fontsize=8.2)
ax.grid(alpha=0.2, lw=0.6); ax.set_axisbelow(True); despine(ax)
fig.tight_layout()
fig.savefig(OUT + 'R_Fig8_tanimoto.png', bbox_inches='tight', facecolor='white')
plt.close(fig)

print('revised figures written')
