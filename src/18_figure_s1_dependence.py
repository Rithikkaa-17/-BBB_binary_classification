#!/usr/bin/env python3
"""Figure S1: attribution against feature value for three descriptors."""
import os, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = '/home/claude/figures_rev/'
os.makedirs(OUT, exist_ok=True)
BM = json.load(open(os.path.join(ROOT, 'results', 'bit_mapping_and_dependence.json')))
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'savefig.dpi': 600})

names = [('TPSA', 'Topological polar surface area (Å$^2$)'),
         ('HBD', 'Hydrogen-bond donor count'),
         ('LogP', 'Wildman–Crippen logP')]
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
for ax, (key, xl) in zip(axes, names):
    d = np.load(os.path.join(ROOT, 'work', 'dep_%s.npy' % key))
    v, a = d[:, 0], d[:, 1]
    ax.scatter(v, a, s=11, alpha=0.55, edgecolor='none', color='#1f7fc4')
    vs = (v - v.mean()) / (v.std() + 1e-9)
    A = np.column_stack([np.ones_like(vs), vs, vs ** 2])
    beta, *_ = np.linalg.lstsq(A, a, rcond=None)
    xs = np.linspace(v.min(), v.max(), 200)
    xss = (xs - v.mean()) / (v.std() + 1e-9)
    ax.plot(xs, beta[0] + beta[1] * xss + beta[2] * xss ** 2, '--', color='#b02a37', lw=1.6)
    ax.axhline(0, color='#888', lw=0.7, zorder=0)
    st = BM['dependence'][key]
    ax.set_xlabel(xl, fontsize=8.5)
    ax.set_title('%s   r = %+.2f   ρ = %+.2f\nR² %.2f → %.2f'
                 % (key, st['pearson'], st['spearman'], st['r2_linear'], st['r2_quadratic']),
                 fontsize=8.5)
    ax.tick_params(labelsize=7.5)
axes[0].set_ylabel('Attribution (SHAP value)', fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT + 'FigureS1_dependence.png', bbox_inches='tight', facecolor='white')
print('saved', OUT + 'FigureS1_dependence.png')
