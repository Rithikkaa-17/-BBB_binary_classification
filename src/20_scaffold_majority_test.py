#!/usr/bin/env python3
"""Reviewer 2, point 1: does the model predict the compound class rather than
the molecule? Tested directly by asking whether a molecule's prediction follows
its own annotation or its scaffold group's majority annotation.
"""
import warnings, json, os, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
z = np.load(os.path.join(ROOT, 'work', 'clean_features.npz'))
te = pd.read_csv(os.path.join(ROOT, 'work', 'clean_test.csv'))
ydv, yte = z['ydv'], z['yte']
fr = json.load(open(os.path.join(ROOT, 'results', 'final_results.json')))
THR = fr['models']['RF + Fused']['threshold']

P = [RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=sd,
                            n_jobs=-1).fit(z['Xdv_c'], ydv).predict_proba(z['Xte_c'])[:, 1]
     for sd in [7, 21, 42, 84, 123]]
p = np.mean(P, axis=0)
d = te.assign(y=yte, p=p, pred=(p >= THR).astype(int))

# scaffold-group majority annotation
grp = d.groupby('scaffold')['y']
d['grp_n'] = grp.transform('size')
d['grp_mean'] = grp.transform('mean')
multi = d[d.grp_n >= 2].copy()
multi['majority'] = (multi.grp_mean > 0.5).astype(int)
multi = multi[multi.grp_mean != 0.5]                      # drop ties
multi['is_minority'] = multi.y != multi.majority
multi['follows_majority'] = multi.pred == multi.majority
multi['correct'] = multi.pred == multi.y

out = {'n_test': int(len(d)), 'n_in_multimember_scaffolds': int(len(d[d.grp_n >= 2])),
       'n_scored': int(len(multi))}
print('test molecules in scaffold groups of >=2 with a non-tied majority: %d' % len(multi))
for lab, sub in [('majority members', multi[~multi.is_minority]),
                 ('minority members', multi[multi.is_minority])]:
    out[lab.split()[0]] = {'n': int(len(sub)), 'accuracy': float(sub.correct.mean()),
                           'follows_group_majority': float(sub.follows_majority.mean()),
                           'mean_p': float(sub.p.mean())}
    print('  %-18s n=%3d  accuracy %.3f  prediction follows group majority %.3f'
          % (lab, len(sub), sub.correct.mean(), sub.follows_majority.mean()))

# how well does the prediction track the group label versus the molecule's own label?
out['roc_vs_own_label'] = float(roc_auc_score(multi.y, multi.p))
out['roc_vs_group_majority'] = float(roc_auc_score(multi.majority, multi.p))
print('\nranking the same molecules against their own annotation : ROC-AUC %.4f'
      % out['roc_vs_own_label'])
print('ranking them against their scaffold majority instead     : ROC-AUC %.4f'
      % out['roc_vs_group_majority'])

# the groups that carry both annotations
mixed = multi[(multi.grp_mean > 0) & (multi.grp_mean < 1)]
out['mixed_groups'] = {'n_groups': int(mixed.scaffold.nunique()), 'n_molecules': int(len(mixed)),
                       'accuracy': float(mixed.correct.mean()),
                       'minority_accuracy': float(mixed[mixed.is_minority].correct.mean()),
                       'n_minority': int(mixed.is_minority.sum())}
print('\nscaffold groups carrying both annotations: %d groups, %d molecules'
      % (mixed.scaffold.nunique(), len(mixed)))
print('  accuracy overall %.3f | on the %d minority members %.3f'
      % (mixed.correct.mean(), mixed.is_minority.sum(), mixed[mixed.is_minority].correct.mean()))

json.dump(out, open(os.path.join(ROOT, 'results', 'scaffold_majority_test.json'), 'w'), indent=1)
print('\nsaved results/scaffold_majority_test.json')
