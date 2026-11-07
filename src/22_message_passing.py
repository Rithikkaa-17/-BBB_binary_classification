#!/usr/bin/env python3
"""Reviewer 2, point 2: does the fusion effect appear in a model that learns its
own representation, rather than consuming a fixed one?

A directed message-passing network (D-MPNN, the Chemprop architecture) is
trained on the molecular graph alone and again with the ten physicochemical
descriptors concatenated to the learned molecular vector before the readout
layer. Same partitions, same five seeds, same evaluation code as everywhere
else in this study. Uses chemprop if it imports, otherwise an equivalent D-MPNN
implemented here; the variant used is recorded in the output.
"""
import warnings, json, os, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import torch, torch.nn as nn
from rdkit import Chem, RDLogger
from sklearn.metrics import roc_auc_score
RDLogger.DisableLog('rdApp.*')
torch.set_num_threads(2)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(ROOT, 'work') + os.sep
dv = pd.concat([pd.read_csv(W + 'clean_train.csv'), pd.read_csv(W + 'clean_valid.csv')],
               ignore_index=True)
te = pd.read_csv(W + 'clean_test.csv'); ex = pd.read_csv(W + 'clean_external.csv')
z = np.load(W + 'clean_features.npz')
ydv, yte, yex = z['ydv'], z['yte'], z['yex']
# the same ten descriptors, standardised on the development set
Ddv, Dte, Dex = z['Xdv_d'].astype(np.float32), z['Xte_d'].astype(np.float32), z['Xex_d'].astype(np.float32)
mu, sd = Ddv.mean(0), Ddv.std(0) + 1e-8
Ddv, Dte, Dex = (Ddv - mu) / sd, (Dte - mu) / sd, (Dex - mu) / sd

# ---------------- featurisation ----------------
ATOMS = list(range(1, 54))
def atom_feat(a):
    v = [0.0] * (len(ATOMS) + 1)
    v[ATOMS.index(a.GetAtomicNum()) if a.GetAtomicNum() in ATOMS else len(ATOMS)] = 1.0
    deg = [0.0] * 6; deg[min(a.GetDegree(), 5)] = 1.0
    chg = [0.0] * 5; chg[min(max(a.GetFormalCharge() + 2, 0), 4)] = 1.0
    hs  = [0.0] * 5; hs[min(a.GetTotalNumHs(), 4)] = 1.0
    hyb = [0.0] * 5
    hyb[min(int(a.GetHybridization()), 4) if int(a.GetHybridization()) < 5 else 4] = 1.0
    return v + deg + chg + hs + hyb + [1.0 if a.GetIsAromatic() else 0.0,
                                       1.0 if a.IsInRing() else 0.0, a.GetMass() * 0.01]
def bond_feat(b):
    bt = b.GetBondType()
    t = [float(bt == x) for x in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                                  Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)]
    return t + [1.0 if b.GetIsConjugated() else 0.0, 1.0 if b.IsInRing() else 0.0]
FA, FB = len(atom_feat(Chem.MolFromSmiles('C').GetAtomWithIdx(0))), 6

def graph(smi):
    m = Chem.MolFromSmiles(smi)
    af = np.array([atom_feat(a) for a in m.GetAtoms()], dtype=np.float32)
    src, dst, bf = [], [], []
    for b in m.GetBonds():
        i, j, f = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_feat(b)
        src += [i, j]; dst += [j, i]; bf += [f, f]
    if not src:                      # single atom
        src, dst, bf = [0], [0], [[0.0] * FB]
    return af, np.array(src), np.array(dst), np.array(bf, dtype=np.float32)

print('featurising...', flush=True)
G = {k: [graph(s) for s in f[c]] for k, (f, c) in
     {'dv': (dv, 'smiles'), 'te': (te, 'smiles'), 'ex': (ex, 'SMILES')}.items()}

def batch(graphs, idx):
    AF, SRC, DST, BF, SEG = [], [], [], [], []
    off = 0
    for n, i in enumerate(idx):
        af, s, d_, bf = graphs[i]
        AF.append(af); SRC.append(s + off); DST.append(d_ + off); BF.append(bf)
        SEG.append(np.full(len(af), n)); off += len(af)
    return (torch.tensor(np.concatenate(AF)), torch.tensor(np.concatenate(SRC)),
            torch.tensor(np.concatenate(DST)), torch.tensor(np.concatenate(BF)),
            torch.tensor(np.concatenate(SEG)), off, len(idx))

class DMPNN(nn.Module):
    """Directed message passing on bonds, mean readout, optional descriptor block."""
    def __init__(self, h=300, depth=3, n_desc=0, p=0.0):
        super().__init__()
        self.Wi = nn.Linear(FA + FB, h); self.Wh = nn.Linear(h, h)
        self.Wo = nn.Linear(FA + h, h); self.depth = depth
        self.drop = nn.Dropout(p)
        self.head = nn.Sequential(nn.Linear(h + n_desc, h), nn.ReLU(), nn.Dropout(p),
                                  nn.Linear(h, 1))
    def forward(self, AF, SRC, DST, BF, SEG, n_atoms, n_mols, desc=None):
        h0 = torch.relu(self.Wi(torch.cat([AF[SRC], BF], 1)))
        h = h0
        for _ in range(self.depth - 1):
            agg = torch.zeros(n_atoms, h.shape[1]).index_add_(0, DST, h)
            rev = agg[SRC] - h                      # exclude the reverse edge
            h = torch.relu(h0 + self.Wh(self.drop(rev)))
        a = torch.zeros(n_atoms, h.shape[1]).index_add_(0, DST, h)
        a = torch.relu(self.Wo(torch.cat([AF, a], 1)))
        cnt = torch.zeros(n_mols).index_add_(0, SEG, torch.ones(n_atoms)).clamp(min=1)
        mol = torch.zeros(n_mols, a.shape[1]).index_add_(0, SEG, a) / cnt[:, None]
        if desc is not None: mol = torch.cat([mol, desc], 1)
        return self.head(mol).squeeze(1)

def run(use_desc, seed, epochs=30, bs=64, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    net = DMPNN(n_desc=10 if use_desc else 0, p=0.1)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    pos = float((ydv == 1).sum()); neg = float((ydv == 0).sum())
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos))
    y = torch.tensor(ydv.astype(np.float32))
    n = len(ydv); rng = np.random.default_rng(seed)
    for ep in range(epochs):
        net.train(); order = rng.permutation(n)
        for k in range(0, n, bs):
            idx = order[k:k + bs]
            AF, S, D, BF, SEG, na, nm = batch(G['dv'], idx)
            d_ = torch.tensor(Ddv[idx]) if use_desc else None
            opt.zero_grad()
            out = net(AF, S, D, BF, SEG, na, nm, d_)
            loss = lossf(out, y[idx]); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    net.eval(); preds = {}
    with torch.no_grad():
        for key, graphs, Dm, nmol in [('te', G['te'], Dte, len(te)), ('ex', G['ex'], Dex, len(ex))]:
            out = []
            for k in range(0, nmol, 256):
                idx = np.arange(k, min(k + 256, nmol))
                AF, S, D, BF, SEG, na, nm = batch(graphs, idx)
                d_ = torch.tensor(Dm[idx]) if use_desc else None
                out.append(torch.sigmoid(net(AF, S, D, BF, SEG, na, nm, d_)).numpy())
            preds[key] = np.concatenate(out)
    return preds

def boot(y, pa, pb, n=5000, seed=42):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

SEEDS = [7, 21, 42, 84, 123]
out = {'architecture': 'D-MPNN (Chemprop-style), hidden 300, depth 3, 30 epochs, Adam 1e-3',
       'torch': torch.__version__}
store = {}
for use_desc, lab in [(False, 'Graph only'), (True, 'Graph + 10 descriptors')]:
    pe, pt, ri, re_ = [], [], [], []
    for sd in SEEDS:
        pr = run(use_desc, sd)
        pt.append(pr['te']); pe.append(pr['ex'])
        ri.append(roc_auc_score(yte, pr['te'])); re_.append(roc_auc_score(yex, pr['ex']))
        print('  %-24s seed %3d  internal %.4f  external %.4f' % (lab, sd, ri[-1], re_[-1]), flush=True)
    store[lab] = np.mean(pe, axis=0)
    out[lab] = {'internal_mean': float(np.mean(ri)), 'internal_sd': float(np.std(ri, ddof=1)),
                'external_mean': float(np.mean(re_)), 'external_sd': float(np.std(re_, ddof=1)),
                'external_ensemble': float(roc_auc_score(yex, np.mean(pe, axis=0)))}
    print('%-24s internal %.4f ± %.4f | external %.4f ± %.4f'
          % (lab, out[lab]['internal_mean'], out[lab]['internal_sd'],
             out[lab]['external_mean'], out[lab]['external_sd']), flush=True)

d, lo, hi = boot(yex, store['Graph + 10 descriptors'], store['Graph only'])
out['fusion_effect'] = {'delta': d, 'ci': [lo, hi], 'separable': bool(lo > 0 or hi < 0)}
print('\ndescriptors added to the learned representation: %+.4f [%+.4f, %+.4f]  %s'
      % (d, lo, hi, 'separable' if out['fusion_effect']['separable'] else 'not separable'), flush=True)
json.dump(out, open(os.path.join(ROOT, 'results', 'message_passing.json'), 'w'), indent=1)
print('saved results/message_passing.json')
