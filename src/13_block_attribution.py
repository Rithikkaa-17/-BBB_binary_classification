"""R2 point 4: block-level attribution, collinearity, nonlinearity, bit mapping."""
import warnings, json, numpy as np, os, sys
warnings.filterwarnings('ignore')
import shap
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr, pearsonr

import os as _o; ROOT=_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))
z=np.load(ROOT+'/work/clean_features.npz')
ydv,yte=z['ydv'],z['yte']
DESC=['MolWt','LogP','TPSA','HBD','HBA','RotatableBonds','RingCount','AromaticRings','FractionCSP3','HeavyAtomCount']
SEEDS=[7,21,42,84,123]
out={}

# ---- multi-seed block attribution ----
blocks={'fingerprint':np.arange(0,2048),'descriptor':np.arange(2048,2058)}
per_seed={'fingerprint':[],'descriptor':[]}
desc_rank_per_seed=[]
for sd in SEEDS:
    m=RandomForestClassifier(n_estimators=500,class_weight='balanced',random_state=sd,n_jobs=-1).fit(z['Xdv_c'],ydv)
    rng=np.random.default_rng(sd)
    bg=z['Xdv_c'][rng.choice(len(ydv),200,replace=False)]
    ex=shap.TreeExplainer(m,data=bg,feature_perturbation='interventional')
    sv=ex.shap_values(z['Xte_c'],check_additivity=False)
    if isinstance(sv,list): sv=sv[1]
    elif sv.ndim==3: sv=sv[:,:,1]
    mabs=np.abs(sv).mean(axis=0)           # mean |SHAP| per feature
    for b,idx in blocks.items():
        per_seed[b].append(float(mabs[idx].sum()))
    order=np.argsort(-mabs)
    desc_rank_per_seed.append({DESC[j-2048]:int(np.where(order==j)[0][0])+1 for j in range(2048,2058)})
    if sd==42:
        out['seed42_mean_abs_desc']={DESC[j-2048]:float(mabs[j]) for j in range(2048,2058)}
        out['seed42_top_bits']=[(int(i),float(mabs[i])) for i in order[:20] if i<2048][:10]
        np.save(ROOT+'/work/sv42.npy',sv)
    print('seed',sd,'fp block sum %.4f  desc block sum %.4f'%(per_seed['fingerprint'][-1],per_seed['descriptor'][-1]),flush=True)

for b in blocks:
    a=np.array(per_seed[b])
    out.setdefault('block_total_mean_abs',{})[b]={'per_seed':a.tolist(),'mean':float(a.mean()),'sd':float(a.std(ddof=1))}
fp=np.array(per_seed['fingerprint']); de=np.array(per_seed['descriptor'])
out['descriptor_share']={'per_seed':(de/(de+fp)).tolist(),'mean':float((de/(de+fp)).mean()),
                         'sd':float((de/(de+fp)).std(ddof=1))}
out['per_feature_mean']={'fingerprint':float(fp.mean()/2048),'descriptor':float(de.mean()/10)}
out['descriptor_rank_per_seed']=desc_rank_per_seed
print('\ndescriptor share of total |SHAP|: %.3f ± %.3f'%(out['descriptor_share']['mean'],out['descriptor_share']['sd']))
print('rank stability of MolWt/HeavyAtomCount across seeds:')
for k in ['MolWt','HeavyAtomCount','TPSA','HBD','HBA','LogP']:
    rr=[d[k] for d in desc_rank_per_seed]
    print('  %-16s ranks %s'%(k,rr))
    out.setdefault('rank_range',{})[k]={'ranks':rr,'min':min(rr),'max':max(rr)}
json.dump(out,open(ROOT+'/results/block_attribution.json','w'),indent=1)
print('saved block_attribution.json')
