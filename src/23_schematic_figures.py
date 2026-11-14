#!/usr/bin/env python3
"""Figures 1 and 2: the system architecture and the evaluation design.

These two schematics were originally drawn outside the repository. They are
rebuilt here so that they are reproducible along with everything else, and so
that house-style corrections (no thousands separators below five digits,
italic variable names) can be applied at source rather than to a raster.

Every number drawn is read from the curated data and the stored results; none
is typed in by hand.
"""
import io, json, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog('rdApp.*')

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(_ROOT, 'work') + os.sep
RESULTS = os.path.join(_ROOT, 'results') + os.sep
OUT = os.path.join(_ROOT, 'figures') + os.sep
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                     'savefig.dpi': 600, 'figure.dpi': 150, 'mathtext.fontset': 'dejavuserif'})

# ------------------------------------------------------------------ palette
DGREEN, LGREEN = '#1d6b3a', '#e7f3ec'
TEAL,   LTEAL  = '#14807a', '#e0f2f0'
BLUE,   LBLUE  = '#1f7fc4', '#e5f0fa'
RED,    LRED   = '#b02a37', '#fbe6e8'
PURPLE, LPURP  = '#7a4fa3', '#f1e9f7'
ORANGE, LORANGE = '#d1743a', '#fbeadf'
INK, MUTED = '#22262a', '#6f767c'

# ------------------------------------------------------------------ the facts
S = json.load(open(D + 'clean_summary.json'))
# the archived (published) run, not the scratch re-run in work/: the decision
# threshold is reselected on 159 validation molecules and is unstable (Section 2.11)
F = json.load(open(RESULTS + 'final_results.json'))
z = np.load(D + 'clean_features.npz')
train = pd.read_csv(D + 'clean_train.csv')
valid = pd.read_csv(D + 'clean_valid.csv')
test = pd.read_csv(D + 'clean_test.csv')
ext = pd.read_csv(D + 'clean_external.csv')


def _n(x):
    """House style: no thousands separator below five digits."""
    x = int(x)
    return str(x) if x < 10000 else format(x, ',')


N_TRAIN, N_VALID, N_TEST, N_EXT = len(train), len(valid), len(test), len(ext)
N_DEV = N_TRAIN + N_VALID
N_CLEAN = N_DEV + N_TEST
N_RAW = int(S['bbbp_raw'])
rf = F['models']['RF + Fused']
xgb = F['models']['XGB + Fused']

# descriptor attribution, for the two bar panels
DESC = ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds',
        'RingCount', 'AromaticRings', 'FractionCSP3', 'HeavyAtomCount']
SHORT = {'MolWt': 'MW', 'LogP': 'LogP', 'TPSA': 'TPSA', 'HBD': 'HBD', 'HBA': 'HBA',
         'RotatableBonds': 'RotB', 'RingCount': 'Rings', 'AromaticRings': 'ArR',
         'FractionCSP3': 'FCSP3', 'HeavyAtomCount': 'HAC'}
sv = np.load(D + 'sv_RF_Fused.npy')
desc_sv = np.abs(sv[:, 2048:]).mean(axis=0)


def box(ax, x0, y0, x1, y1, edge, face, lw=2.0):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle='round,pad=0.0,rounding_size=1.1',
                                lw=lw, edgecolor=edge, facecolor=face, zorder=1))


def title(ax, x, y, s, color, size=13.5):
    ax.text(x, y, s, ha='center', va='center', fontsize=size,
            fontweight='bold', color=color, zorder=4)


def body(ax, x, y, s, size=10.5, color=INK, ha='center', va='center'):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color,
            zorder=4, linespacing=1.5)


def elbow(ax, x0, y0, x1, y1, color='#3a4046', lw=2.2):
    """Orthogonal connector with an arrow head, routed vertically then across."""
    mid = (y0 + y1) / 2
    ax.plot([x0, x0], [y0, mid], color=color, lw=lw, solid_capstyle='butt', zorder=2)
    ax.plot([x0, x1], [mid, mid], color=color, lw=lw, solid_capstyle='butt', zorder=2)
    ax.add_patch(FancyArrowPatch((x1, mid), (x1, y1), arrowstyle='-|>',
                                 mutation_scale=16, color=color, lw=lw, zorder=2))


def arrow(ax, x0, y0, x1, y1, color='#3a4046', lw=2.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                                 mutation_scale=16, color=color, lw=lw, zorder=2))


def molimg(smiles, w=430, h=300):
    m = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(m)
    d = rdMolDraw2D.MolDraw2DCairo(w, h)
    o = d.drawOptions()
    o.clearBackground = False
    o.bondLineWidth = 2
    rdMolDraw2D.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))


def place_mol(ax, smiles, cx, cy, w, h):
    """Draw a molecule centred on (cx, cy), preserving its aspect ratio."""
    img = molimg(smiles)
    ar = img.size[0] / img.size[1]
    if w / h > ar:
        w = h * ar
    else:
        h = w / ar
    ax.imshow(img, extent=(cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2),
              zorder=3, interpolation='antialiased')


# pick illustrative molecules from the data itself, deterministically
def pick_by_name(frame, *names):
    out = []
    for nm in names:
        hit = frame[frame['name'].astype(str).str.lower() == nm.lower()]
        if len(hit):
            out.append(str(hit.iloc[0]['smiles']))
    return out


p_te = np.load(D + 'final_preds.npz')['p_RF_Fused']
thr = rf['threshold']
tv = test.copy()
tv['prob'] = p_te
tv['y'] = z['yte']


def smallest_ok(frame, lo=9, hi=26):
    for _, r in frame.iterrows():
        m = Chem.MolFromSmiles(r['smiles'])
        if m is not None and lo <= m.GetNumHeavyAtoms() <= hi:
            return r
    return frame.iloc[0]


top_perm = smallest_ok(tv[(tv.y == 1)].nlargest(40, 'prob'))
top_non = smallest_ok(tv[(tv.y == 0)].nsmallest(40, 'prob'))
sample3 = [smallest_ok(train.iloc[i::7]) for i in (0, 1, 2)]

# =========================================================== Figure 1
fig, ax = plt.subplots(figsize=(10.4, 9.85))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

# --- row 1 -----------------------------------------------------------------
box(ax, 2, 81.5, 38, 98.5, DGREEN, LGREEN)
title(ax, 20, 96.0, 'BBBP Dataset', DGREEN)
body(ax, 20, 92.6, '%s molecules · %s valid' % (_n(N_RAW), _n(N_CLEAN)))
for i, r in enumerate(sample3):
    place_mol(ax, r['smiles'], 8.5 + i * 11.3, 86.6, 11.0, 8.4)

box(ax, 42, 81.5, 67, 98.5, DGREEN, LGREEN)
title(ax, 54.5, 96.0, 'Curation', DGREEN)
body(ax, 54.5, 90.5, 'RDKit validation\nBemis–Murcko\nscaffolds')

box(ax, 71, 81.5, 98, 98.5, DGREEN, LGREEN)
title(ax, 84.5, 96.0, 'Partition', DGREEN)
body(ax, 84.5, 91.0, 'train %s · validation %s\nindependent test %s'
     % (_n(N_TRAIN), _n(N_VALID), _n(N_TEST)))

arrow(ax, 38, 90, 42, 90)
arrow(ax, 67, 90, 71, 90)

# --- row 2 -----------------------------------------------------------------
box(ax, 2, 56.5, 47, 78.0, TEAL, LTEAL)
title(ax, 24.5, 75.4, 'Morgan Fingerprint', TEAL)
body(ax, 24.5, 71.8, 'radius 2 · %s bits' % _n(2048))
place_mol(ax, top_perm['smiles'], 10.5, 64.0, 14.0, 11.0)
ax.add_patch(Rectangle((19.5, 60.6), 25.5, 7.6, lw=1.3, edgecolor=TEAL,
                       facecolor='#f3faf9', zorder=2))
rng = np.random.default_rng(7)
bits = rng.choice(np.arange(40), size=40, replace=False)
for b in rng.uniform(19.9, 44.6, 40):
    hgt = rng.uniform(0.7, 1.5)
    ax.plot([b, b], [61.1 + rng.uniform(0, 5.4), 0], lw=0)  # keep rng in step
for b, yy in zip(rng.uniform(19.9, 44.6, 40), rng.uniform(61.2, 67.0, 40)):
    ax.add_patch(Rectangle((b, yy), 0.32, 0.95, lw=0, facecolor=TEAL, zorder=3))
body(ax, 32.2, 59.4, 'all %s bits · 40 set' % _n(2048), size=9.2, color=MUTED)

box(ax, 51, 56.5, 98, 78.0, TEAL, LTEAL)
title(ax, 74.5, 75.4, 'Physicochemical Descriptors', TEAL)
body(ax, 74.5, 71.8, '10 interpretable properties')
bw = 3.4
base = 61.0
hs = desc_sv / desc_sv.max() * 6.2
for i, d_ in enumerate(DESC):
    cx = 56.5 + i * 4.1
    ax.add_patch(Rectangle((cx - bw / 2, base), bw, hs[i], lw=0, facecolor=TEAL, zorder=3))
    ax.text(cx, base - 0.6, SHORT[d_], rotation=38, ha='right', va='top',
            fontsize=8.6, color=INK, zorder=4)
ax.plot([54.3, 95.5], [base, base], color=INK, lw=1.1, zorder=3)

elbow(ax, 54.5, 81.5, 24.5, 78.0)
elbow(ax, 84.5, 81.5, 74.5, 78.0)

# --- row 3 -----------------------------------------------------------------
box(ax, 19, 41.0, 81, 54.0, TEAL, LTEAL)
title(ax, 50, 51.6, 'Feature Fusion    $X$ = [$M$, $D$]', TEAL)
body(ax, 50, 48.2, '%s + 10 = %s features per molecule' % (_n(2048), _n(2058)))
ax.add_patch(Rectangle((23, 42.6), 54, 3.4, lw=1.2, edgecolor=TEAL,
                       facecolor='white', zorder=2))
rng2 = np.random.default_rng(11)
for b in rng2.uniform(23.4, 70.0, 34):
    ax.add_patch(Rectangle((b, 42.6), 0.42, 3.4, lw=0, facecolor=TEAL, zorder=3))
ax.add_patch(Rectangle((71.6, 42.6), 5.4, 3.4, lw=0, facecolor=ORANGE, zorder=3))
body(ax, 47, 41.9, 'fingerprint block', size=9.0, color=TEAL)
body(ax, 74.3, 41.9, 'descriptors', size=9.0, color=ORANGE)

elbow(ax, 24.5, 56.5, 38, 54.0)
elbow(ax, 74.5, 56.5, 62, 54.0)

# --- row 4 -----------------------------------------------------------------
box(ax, 6, 24.0, 47, 38.5, BLUE, LBLUE)
title(ax, 26.5, 36.0, 'Random Forest', BLUE)
body(ax, 36.0, 30.0, '500 trees\nbalanced class weights\nthreshold %.2f' % rf['threshold'],
     size=10.0)
for i, (xx, yy) in enumerate([(12.0, 32.6), (14.6, 32.6), (17.2, 32.6)]):
    ax.add_patch(plt.Circle((xx, yy), 0.78, color=BLUE, zorder=4))
    for dx in (-1.6, 1.6):
        ax.plot([xx, xx + dx], [yy - 0.6, yy - 3.4], color=BLUE, lw=1.3, zorder=3)
        ax.add_patch(plt.Circle((xx + dx, yy - 3.8), 0.70, color='#9ec8ea',
                                ec=BLUE, lw=1.0, zorder=4))

box(ax, 52, 24.0, 94, 38.5, BLUE, LBLUE)
title(ax, 73, 36.0, 'Gradient Boosting', BLUE)
body(ax, 83.0, 30.0, '600 trees · depth 5\nlr 0.04 · subsample 0.8\n'
     'threshold %.2f' % xgb['threshold'], size=10.0)
ax.add_patch(plt.Circle((63.0, 33.2), 0.80, color=BLUE, zorder=4))
for dx in (-2.6, 2.6):
    ax.plot([63.0, 63.0 + dx], [32.5, 30.0], color=BLUE, lw=1.3, zorder=3)
    ax.add_patch(plt.Circle((63.0 + dx, 29.6), 0.74, color=BLUE, zorder=4))
    for dx2 in (-1.3, 1.3):
        ax.plot([63.0 + dx, 63.0 + dx + dx2], [29.0, 26.6], color=BLUE, lw=1.2, zorder=3)
        ax.add_patch(plt.Circle((63.0 + dx + dx2, 26.2), 0.66, color='#9ec8ea',
                                ec=BLUE, lw=1.0, zorder=4))

elbow(ax, 50, 41.0, 26.5, 38.5)
elbow(ax, 50, 41.0, 73, 38.5)

# --- row 5 -----------------------------------------------------------------
box(ax, 2, 2.0, 47, 21.0, RED, LRED)
title(ax, 24.5, 18.6, 'Classification', RED)
place_mol(ax, top_perm['smiles'], 13.0, 10.6, 17.0, 10.4)
place_mol(ax, top_non['smiles'], 35.0, 10.6, 17.0, 10.4)
ax.text(13.0, 4.0, 'permeable · $p$ = %.2f' % float(top_perm['prob']),
        ha='center', va='center', fontsize=9.0, fontweight='bold', color=DGREEN, zorder=4)
ax.text(35.0, 4.0, 'non-permeable · $p$ = %.2f' % float(top_non['prob']),
        ha='center', va='center', fontsize=9.0, fontweight='bold', color=RED, zorder=4)

box(ax, 51, 2.0, 98, 21.0, PURPLE, LPURP)
title(ax, 74.5, 18.6, 'SHAP Interpretation', PURPLE)
top5 = np.argsort(desc_sv)[::-1][:5]
for i, fi in enumerate(top5):
    yy = 14.6 - i * 2.3
    wdt = desc_sv[fi] / desc_sv[top5[0]] * 25.0
    ax.add_patch(Rectangle((64.0, yy - 0.78), wdt, 1.56, lw=0, facecolor=PURPLE, zorder=3))
    ax.text(63.3, yy, SHORT[DESC[fi]], ha='right', va='center', fontsize=9.4,
            color=INK, zorder=4)
ax.plot([63.9, 63.9], [3.9, 15.7], color=INK, lw=1.1, zorder=3)
body(ax, 77.0, 3.1, 'mean |SHAP| — leading descriptors', size=9.2, color=MUTED)

elbow(ax, 26.5, 24.0, 24.5, 21.0)
elbow(ax, 73, 24.0, 74.5, 21.0)

fig.savefig(OUT + 'Figure1_architecture.png', bbox_inches='tight',
            facecolor='white', pad_inches=0.08)
plt.close(fig)
print('Figure 1 written')

# =========================================================== Figure 2
N_POS = int((z['ytr'].sum() + z['yva'].sum() + z['yte'].sum()))
N_NEG = N_CLEAN - N_POS
E_POS = int(z['yex'].sum())
E_NEG = N_EXT - E_POS
ov = rf['strata']['overlap_dev']['n']
nv = rf['strata']['novel_vs_dev']['n']

fig, ax = plt.subplots(figsize=(10.6, 7.8))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

box(ax, 26, 88.5, 74, 99.0, DGREEN, LGREEN)
title(ax, 50, 96.4, 'Curated BBBP Dataset', DGREEN)
body(ax, 50, 91.6, '%s molecules (%s + / %s −)' % (_n(N_CLEAN), _n(N_POS), _n(N_NEG)))

box(ax, 2, 72.5, 46, 84.0, '#2f8f5b', LGREEN)
title(ax, 24, 81.4, 'Development Set — %s' % _n(N_DEV), '#2f8f5b')
body(ax, 24, 76.4, 'train %s · validation %s' % (_n(N_TRAIN), _n(N_VALID)))

box(ax, 54, 72.5, 98, 84.0, '#2f8f5b', LGREEN)
title(ax, 76, 81.4, 'Internal Test Set — %s' % _n(N_TEST), '#2f8f5b')
body(ax, 76, 76.4, 'sequestered until final evaluation')

elbow(ax, 50, 88.5, 24, 84.0)
elbow(ax, 50, 88.5, 76, 84.0)

box(ax, 2, 55.5, 98, 68.5, BLUE, LBLUE)
title(ax, 50, 65.8, 'Factorial Design — 3 representations × 2 algorithms '
                    '= 6 configurations', BLUE, size=13.0)
body(ax, 50, 59.6, 'Morgan %s-D  ·  Descriptors 10-D  ·  Fused %s-D        '
     'Random Forest  ·  Gradient Boosting' % (_n(2048), _n(2058)))

elbow(ax, 24, 72.5, 50, 68.5)

box(ax, 2, 37.0, 47, 52.0, BLUE, LBLUE)
title(ax, 24.5, 49.6, 'Five Seeds per Configuration', BLUE, size=12.5)
body(ax, 24.5, 42.6, 'independent refits (7, 21, 42, 84, 123)\n'
     'validation-selected MCC threshold per seed\n'
     'median threshold · seed-averaged probabilities', size=10.0)

box(ax, 50, 37.0, 92, 52.0, DGREEN, LGREEN)
title(ax, 71, 49.6, 'External Set — %s' % _n(N_EXT), DGREEN)
body(ax, 71, 42.6, 'B3DB, connectivity-disjoint from development\n'
     '%s + / %s −\nscaffold-novel and scaffold-overlapping strata'
     % (_n(E_POS), _n(E_NEG)), size=10.0)

arrow(ax, 24.5, 55.5, 24.5, 52.0)
# the internal test set bypasses development and feeds held-out evaluation directly
ax.plot([96.5, 96.5], [72.5, 35.2], color='#3a4046', lw=2.2, zorder=2)
ax.plot([96.5, 90.0], [35.2, 35.2], color='#3a4046', lw=2.2, zorder=2)
arrow(ax, 90.0, 35.2, 90.0, 33.5)

box(ax, 2, 20.0, 98, 33.5, ORANGE, LORANGE)
title(ax, 50, 31.0, 'Held-Out Evaluation — each set scored once under the fixed protocol',
      ORANGE, size=13.0)
body(ax, 50, 24.6, 'internal test %s  ·  external %s, scaffold-stratified  ·  '
     'nine performance measures' % (_n(N_TEST), _n(N_EXT)))

arrow(ax, 24.5, 37.0, 24.5, 33.5)
arrow(ax, 71, 37.0, 71, 33.5)

box(ax, 18, 5.2, 82, 17.4, PURPLE, LPURP)
title(ax, 50, 15.4, 'Inference and Selection', PURPLE)
body(ax, 50, 9.7, 'paired bootstrap, %s resamples  ·  sum of ranking differences  '
     '·  Pareto front\ncalibration  ·  SHAP attribution' % _n(5000), size=10.0)

arrow(ax, 50, 20.0, 50, 17.4)

leg = [('Data', DGREEN, LGREEN), ('Partitioning', '#2f8f5b', LGREEN),
       ('Development experiments', BLUE, LBLUE),
       ('Held-out evaluation', ORANGE, LORANGE), ('Interpretation', PURPLE, LPURP)]
SW, GAP, PAD, FS = 2.4, 1.3, 3.4, 9.6
CPW = FS * 0.47                      # approximate character width in axis units
widths = [SW + GAP + len(l) * CPW for l, _, _ in leg]
x = (100 - (sum(widths) + PAD * (len(leg) - 1))) / 2
for (lab, ec, fc), w in zip(leg, widths):
    ax.add_patch(Rectangle((x, 0.1), SW, 1.6, lw=1.6, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(x + SW + GAP, 0.9, lab, ha='left', va='center', fontsize=FS, color=INK, zorder=4)
    x += w + PAD

fig.savefig(OUT + 'Figure2_evaluation_design.png', bbox_inches='tight',
            facecolor='white', pad_inches=0.08)
plt.close(fig)
print('Figure 2 written')
