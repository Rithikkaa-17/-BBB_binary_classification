#!/usr/bin/env python3
"""SHAP beeswarm and molecule panel for the primary (RF + Fused) configuration."""
import io, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog('rdApp.*')

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
D = _os.path.join(_ROOT, 'work') + _os.sep        # intermediate artefacts
DATA = _os.path.join(_ROOT, 'data') + _os.sep     # partition assignments
RESULTS = _os.path.join(_ROOT, 'results') + _os.sep
OUT = _os.path.join(_ROOT, 'figures') + _os.sep
for _p in (D, RESULTS, OUT):
    _os.makedirs(_p, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 10,
                     'savefig.dpi': 600, 'figure.dpi': 150})

sv = np.load(D + 'sv_RF_Fused.npy')
z = np.load(D + 'clean_features.npz')
X = z['Xte_c']
DESC = ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds',
        'RingCount', 'AromaticRings', 'FractionCSP3', 'HeavyAtomCount']
FEAT = ['Morgan_%d' % i for i in range(2048)] + DESC

ma = np.abs(sv).mean(axis=0)
order = np.argsort(ma)[::-1][:14]

# Blue -> grey -> red: the convention readers expect, and unambiguous about which
# end is "low" and which is "high" (Reviewer 2 found the previous scale confusing).
CM = LinearSegmentedColormap.from_list('lowhigh', ['#2166ac', '#cfd4d8', '#b2182b'])

fig, ax = plt.subplots(figsize=(6.8, 5.0))
rng = np.random.default_rng(0)
for row, fi in enumerate(order[::-1]):
    s = sv[:, fi]
    v = X[:, fi].astype(float)
    vn = np.zeros_like(v) if v.max() == v.min() else (v - v.min()) / (v.max() - v.min())
    jitter = rng.uniform(-0.17, 0.17, len(s))
    ax.scatter(s, row + jitter, c=vn, cmap=CM, s=11, alpha=0.85, linewidths=0, vmin=0, vmax=1)
ax.axvline(0, color='#8b9399', lw=1.0)
ax.set_yticks(range(len(order)))
labels = [FEAT[i] for i in order[::-1]]
ax.set_yticklabels(labels, fontsize=9.5)
for t, lab in zip(ax.get_yticklabels(), labels):
    if not lab.startswith('Morgan_'):
        t.set_fontweight('bold')
ax.set_xlabel('SHAP value  (negative → non-permeable,  positive → permeable)', fontsize=10)
ax.set_ylim(-0.6, len(order) - 0.4)
ax.grid(axis='x', alpha=0.18, lw=0.6); ax.set_axisbelow(True)
for s_ in ('top', 'right'):
    ax.spines[s_].set_visible(False)
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CM), ax=ax, fraction=0.03, pad=0.02)
cb.set_ticks([0, 1]); cb.set_ticklabels(['low', 'high'])
cb.set_label('Feature value', fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT + 'R_Fig9_shap_beeswarm.png', bbox_inches='tight', facecolor='white')
plt.close(fig)
print('beeswarm written; top:', ', '.join(FEAT[i] for i in order[:6]))

# ------------------------------------------------------------------ molecules
P = np.load(D + 'final_preds.npz')
p_te = P['p_RF_Fused']
thr = json.load(open(D + 'final_results.json'))['models']['RF + Fused']['threshold']
test = pd.read_csv(D + 'clean_test.csv')
test['prob'] = p_te
test['y'] = z['yte']
test['pred'] = (test['prob'] >= thr).astype(int)


def props(s):
    m = Chem.MolFromSmiles(s)
    return (rdMolDescriptors.CalcTPSA(m), Crippen.MolLogP(m),
            Lipinski.NumHDonors(m), Descriptors.MolWt(m))


def png(s, w=440, h=290):
    m = Chem.MolFromSmiles(s)
    d = rdMolDraw2D.MolDraw2DCairo(w, h)
    o = d.drawOptions(); o.clearBackground = False; o.bondLineWidth = 2
    rdMolDraw2D.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))


seen = set()


def pick(frame, n):
    out = []
    for _, r in frame.iterrows():
        m = Chem.MolFromSmiles(r['smiles'])
        if m is None:
            continue
        Chem.RemoveStereochemistry(m)
        k = Chem.MolToSmiles(m)
        if 8 <= m.GetNumHeavyAtoms() <= 34 and k not in seen:
            seen.add(k); out.append(r)
        if len(out) == n:
            break
    return out


sel = (pick(test[(test.y == 1) & (test.pred == 1)].nlargest(30, 'prob'), 2)
       + pick(test[(test.y == 0) & (test.pred == 0)].nsmallest(30, 'prob'), 2)
       + pick(test[(test.y == 1) & (test.pred == 0)].nsmallest(10, 'prob'), 1)
       + pick(test[(test.y == 0) & (test.pred == 1)].nlargest(10, 'prob'), 1))
labs = ['Permeable, correctly identified', 'Permeable, correctly identified',
        'Non-permeable, correctly identified', 'Non-permeable, correctly identified',
        'Permeable, missed', 'Non-permeable, missed']
cols = [('#1d6b3a', '#e7f3ec')] * 2 + [('#1f7fc4', '#e5f0fa')] * 2 + [('#b02a37', '#fbe6e8')] * 2

fig, axes = plt.subplots(2, 3, figsize=(7.3, 5.0))
for ax, r, lab, (edge, face) in zip(axes.ravel(), sel, labs, cols):
    ax.imshow(png(r['smiles'])); ax.axis('off')
    tpsa, logp, hbd, mw = props(r['smiles'])
    nm = str(r['name']); nm = nm if len(nm) <= 24 else nm[:22] + '…'
    ax.set_title('%s\n%s' % (nm, lab), fontsize=8.2, color=edge, fontweight='bold',
                 linespacing=1.3, pad=3)
    ax.text(0.5, -0.09, 'TPSA %.0f · LogP %.1f · HBD %d · MW %.0f\n'
            'predicted probability %.2f' % (tpsa, logp, hbd, mw, r['prob']),
            transform=ax.transAxes, ha='center', va='top', fontsize=7.8, linespacing=1.35)
    ax.add_patch(FancyBboxPatch((0.012, 0.012), 0.976, 0.976, transform=ax.transAxes,
                                boxstyle='round,pad=0.004,rounding_size=0.02',
                                lw=1.5, edgecolor=edge, facecolor=face, zorder=-1))
fig.tight_layout(pad=0.5, h_pad=2.4, w_pad=0.9)
fig.savefig(OUT + 'R_Fig10_molecules.png', bbox_inches='tight', facecolor='white')
plt.close(fig)
print('molecules:', [str(r['name']) for r in sel], 'threshold %.2f' % thr)
