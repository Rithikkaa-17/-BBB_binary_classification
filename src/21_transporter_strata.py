#!/usr/bin/env python3
"""Reviewer 2, point 1: stratify external performance by documented efflux-
transporter substrate status.

Neither collection annotates transporter status, so the flag is assigned by
matching compound names against a literature-curated list of compounds for
which P-glycoprotein (ABCB1) or BCRP (ABCG2) substrate status is documented.
The list is shipped verbatim below so that the annotation can be audited and
extended; coverage is reported rather than assumed, and no compound is flagged
on structural grounds.
"""
import warnings, json, os, re, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# --- documented efflux substrates (P-gp unless noted; B = BCRP) -------------
PGP = """
digoxin dabigatran fexofenadine loperamide domperidone cyclosporine cyclosporin
vinblastine vincristine vinorelbine paclitaxel docetaxel doxorubicin daunorubicin
epirubicin idarubicin etoposide teniposide mitoxantrone topotecan irinotecan
colchicine quinidine quinine verapamil diltiazem nicardipine talinolol celiprolol
erythromycin azithromycin clarithromycin rifampin rifampicin levofloxacin
ritonavir saquinavir indinavir nelfinavir amprenavir lopinavir atazanavir
darunavir maraviroc raltegravir tenofovir
tacrolimus sirolimus everolimus temsirolimus
imatinib dasatinib nilotinib gefitinib erlotinib lapatinib sunitinib sorafenib
vandetanib crizotinib vemurafenib
risperidone paliperidone olanzapine quetiapine aripiprazole amisulpride
cetirizine levocetirizine desloratadine
ondansetron dolasetron
morphine methadone loperamide fentanyl oxycodone
amitriptyline nortriptyline imipramine desipramine doxepin
phenytoin phenobarbital carbamazepine lamotrigine felbamate topiramate
atorvastatin lovastatin simvastatin pravastatin rosuvastatin fluvastatin
cimetidine ranitidine famotidine
prazosin terazosin doxazosin
dexamethasone hydrocortisone aldosterone corticosterone methylprednisolone
tamoxifen toremifene
ivermectin moxidectin abamectin
grepafloxacin sparfloxacin ciprofloxacin norfloxacin ofloxacin
amprenavir nevirapine efavirenz
losartan valsartan olmesartan telmisartan
digitoxin ouabain
rhodamine calcein
methotrexate mitoxantrone topotecan sulfasalazine sulphasalazine
nitrofurantoin prazosin glyburide glibenclamide
imatinib irinotecan rosuvastatin pitavastatin
""".split()
PGP = sorted(set(w.strip().lower() for w in PGP if w.strip()))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
z = np.load(os.path.join(ROOT, 'work', 'clean_features.npz'))
ex = pd.read_csv(os.path.join(ROOT, 'work', 'clean_external.csv'))
te = pd.read_csv(os.path.join(ROOT, 'work', 'clean_test.csv'))
ydv, yex, yte = z['ydv'], z['yex'], z['yte']
THR = json.load(open(os.path.join(ROOT, 'results', 'final_results.json')))['models']['RF + Fused']['threshold']

Pe, Pt = [], []
for sd in [7, 21, 42, 84, 123]:
    m = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                               random_state=sd, n_jobs=-1).fit(z['Xdv_c'], ydv)
    Pe.append(m.predict_proba(z['Xex_c'])[:, 1]); Pt.append(m.predict_proba(z['Xte_c'])[:, 1])
pex, pte = np.mean(Pe, axis=0), np.mean(Pt, axis=0)

def flag(series):
    """Word-boundary match so that 'quinine' does not match 'chloroquine'."""
    out = []
    for n in series.fillna('').astype(str).str.lower():
        tok = set(re.split(r'[^a-z]+', n))
        out.append(any(k in tok for k in PGP))
    return np.array(out)

res = {'list_size': len(PGP), 'list': PGP}
for lab, frame, col, y, p, n_tot in [
        ('external', ex, 'compound_name', yex, pex, len(ex)),
        ('internal', te, 'name', yte, pte, len(te))]:
    f = flag(frame[col])
    d = pd.DataFrame({'y': y, 'p': p, 'flag': f, 'err': (p >= THR).astype(int) != y})
    named = frame[col].notna().sum()
    block = {'n_total': int(n_tot), 'n_named': int(named), 'n_flagged': int(f.sum()),
             'coverage_of_named': float(f.sum() / max(named, 1))}
    print('\n=== %s set ===' % lab, flush=True)
    print('  named %d/%d   flagged as documented efflux substrates: %d (%.1f%% of named)'
          % (named, n_tot, f.sum(), 100 * f.sum() / max(named, 1)), flush=True)
    for nm, sub in [('efflux substrate', d[d.flag]), ('rest', d[~d.flag])]:
        if len(sub) and sub.y.nunique() > 1:
            block[nm.replace(' ', '_')] = {
                'n': int(len(sub)), 'prevalence': float(sub.y.mean()),
                'roc': float(roc_auc_score(sub.y, sub.p)),
                'error_rate': float(sub.err.mean()), 'mean_p': float(sub.p.mean())}
            print('  %-18s n=%4d  prevalence %.3f  ROC-AUC %.4f  error rate %.3f  mean p %.3f'
                  % (nm, len(sub), sub.y.mean(), roc_auc_score(sub.y, sub.p),
                     sub.err.mean(), sub.p.mean()), flush=True)
        elif len(sub):
            block[nm.replace(' ', '_')] = {'n': int(len(sub)), 'prevalence': float(sub.y.mean()),
                                           'roc': None, 'error_rate': float(sub.err.mean())}
            print('  %-18s n=%4d  prevalence %.3f  (single class, ROC undefined)'
                  % (nm, len(sub), sub.y.mean()), flush=True)
    res[lab] = block
    if 'efflux_substrate' in block and block['efflux_substrate'].get('roc') is not None:
        print('  flagged names:', ', '.join(sorted(set(frame[col].astype(str)[f]))[:14]), '...', flush=True)

# --- is the gap on the external set bigger than sampling noise? ---
f = flag(ex['compound_name'])
ya, pa = yex[f], pex[f]          # documented efflux substrates
yb, pb = yex[~f], pex[~f]        # the rest
rng = np.random.default_rng(42); d = []
for _ in range(5000):
    ia = rng.integers(0, len(ya), len(ya)); ib = rng.integers(0, len(yb), len(yb))
    if len(np.unique(ya[ia])) < 2 or len(np.unique(yb[ib])) < 2: continue
    d.append(roc_auc_score(ya[ia], pa[ia]) - roc_auc_score(yb[ib], pb[ib]))
d = np.array(d)
res['external_gap'] = {'delta': float(d.mean()),
                       'ci': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                       'p_two_sided': float(min(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0)),
                       'separable': bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)}
print('\nexternal: efflux substrates minus the rest, ROC-AUC')
print('  %+.4f  95%% CI %+.4f to %+.4f  p = %.4f  %s'
      % (d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5),
         res['external_gap']['p_two_sided'],
         'separable' if res['external_gap']['separable'] else 'not separable'), flush=True)

json.dump(res, open(os.path.join(ROOT, 'results', 'transporter_strata.json'), 'w'), indent=1)
print('\nsaved results/transporter_strata.json')
