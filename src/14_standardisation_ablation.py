"""R2 point 3: standardise/desalt/neutralise, add ionisation descriptors, re-run."""
import warnings, json, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors, DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
RDLogger.DisableLog('rdApp.*')

import os as _o; ROOT=_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))
W=ROOT+'/work/'
tr=pd.read_csv(W+'clean_train.csv'); va=pd.read_csv(W+'clean_valid.csv')
te=pd.read_csv(W+'clean_test.csv');  ex=pd.read_csv(W+'clean_external.csv')
dv=pd.concat([tr,va],ignore_index=True)

_lfc=rdMolStandardize.LargestFragmentChooser()
_un =rdMolStandardize.Uncharger()
def standardise(s):
    m=Chem.MolFromSmiles(s)
    if m is None: return None
    m=rdMolStandardize.Cleanup(m)       # sanitise, normalise, reionise
    m=_lfc.choose(m)                    # desalt: keep largest fragment
    m=_un.uncharge(m)                   # neutralise
    return m

DESC=['MolWt','LogP','TPSA','HBD','HBA','RotatableBonds','RingCount','AromaticRings','FractionCSP3','HeavyAtomCount']
ION =['FormalChargeRaw','NumBasicGroups','NumAcidicGroups','NumIonisable','LogD74_est']
_basic=Chem.MolFromSmarts('[NX3;H2,H1,H0;!$(N[#6]=[O,N,S]);!$(N[a]);!$(N=*);!$([N+])]')
_acids=[Chem.MolFromSmarts(x) for x in
        ['[CX3](=O)[OX2H1,OX1-]','[SX4](=O)(=O)[OX2H1,OX1-]','[nH1]1nnnc1','[PX4](=O)([OX2H1,OX1-])']]
_acids=[a for a in _acids if a is not None]

def desc10(m):
    return [Descriptors.MolWt(m),Crippen.MolLogP(m),rdMolDescriptors.CalcTPSA(m),
            Lipinski.NumHDonors(m),Lipinski.NumHAcceptors(m),Lipinski.NumRotatableBonds(m),
            rdMolDescriptors.CalcNumRings(m),rdMolDescriptors.CalcNumAromaticRings(m),
            rdMolDescriptors.CalcFractionCSP3(m),m.GetNumHeavyAtoms()]

def ion5(m, raw_charge):
    nb=len(m.GetSubstructMatches(_basic))
    na=sum(len(m.GetSubstructMatches(a)) for a in _acids)
    logp=Crippen.MolLogP(m)
    # crude logD7.4 surrogate: penalise ionisable centres (Bhal-style offset)
    logd = logp - 1.0*(nb>0) - 2.0*(na>0)
    return [raw_charge, nb, na, nb+na, logd]

def morgan(m,r=2,n=2048):
    fp=AllChem.GetMorganFingerprintAsBitVect(m,r,nBits=n)
    a=np.zeros(n,dtype=np.uint8); DataStructs.ConvertToNumpyArray(fp,a); return a

def build(smiles):
    M=[];D10=[];D5=[]
    for s in smiles:
        raw=Chem.MolFromSmiles(s)
        rc=Chem.GetFormalCharge(raw) if raw is not None else 0
        m=standardise(s)
        if m is None: m=raw
        M.append(morgan(m)); D10.append(desc10(m)); D5.append(ion5(m,rc))
    M=np.array(M); D10=np.nan_to_num(np.array(D10,dtype=np.float32),nan=0.,posinf=0.,neginf=0.)
    D5=np.nan_to_num(np.array(D5,dtype=np.float32),nan=0.,posinf=0.,neginf=0.)
    return M,D10,D5

print('standardising...',flush=True)
sets={}
for nm,fr,col in [('dv',dv,'smiles'),('te',te,'smiles'),('ex',ex,'SMILES')]:
    sets[nm]=build(fr[col]); print('  ',nm,'done',flush=True)
ydv=dv['p_np'].values.astype(int); yte=te['p_np'].values.astype(int); yex=ex['y'].values.astype(int)

# how much did standardisation change?
nchanged=sum(1 for s in dv['smiles'] if Chem.MolToSmiles(standardise(s))!=Chem.MolToSmiles(Chem.MolFromSmiles(s)))
print('development molecules altered by standardisation: %d / %d (%.1f%%)'%(nchanged,len(dv),100*nchanged/len(dv)),flush=True)

def run(Xdv,Xte,Xex,label,seeds=[7,21,42,84,123]):
    ps_t=[];ps_e=[];rt=[];re_=[]
    for sd in seeds:
        m=RandomForestClassifier(n_estimators=500,class_weight='balanced',random_state=sd,n_jobs=-1).fit(Xdv,ydv)
        pt=m.predict_proba(Xte)[:,1]; pe=m.predict_proba(Xex)[:,1]
        ps_t.append(pt);ps_e.append(pe); rt.append(roc_auc_score(yte,pt)); re_.append(roc_auc_score(yex,pe))
    r={'internal_mean':float(np.mean(rt)),'internal_sd':float(np.std(rt,ddof=1)),
       'external_mean':float(np.mean(re_)),'external_sd':float(np.std(re_,ddof=1)),
       'external_ensemble':float(roc_auc_score(yex,np.mean(ps_e,axis=0)))}
    print('  %-34s internal %.4f±%.4f  external %.4f±%.4f'%(label,r['internal_mean'],r['internal_sd'],r['external_mean'],r['external_sd']),flush=True)
    return r,np.mean(ps_e,axis=0)

out={'n_dev_altered':int(nchanged),'n_dev':int(len(dv))}
print('\n=== ablation (RF, fused) ===',flush=True)
# A: standardised, 10 descriptors
Xa=[np.concatenate([sets[k][0],sets[k][1]],axis=1) for k in ('dv','te','ex')]
out['standardised_10desc'],pa=run(*Xa,'standardised + 10 descriptors')
# B: standardised, 10 + 5 ionisation
Xb=[np.concatenate([sets[k][0],sets[k][1],sets[k][2]],axis=1) for k in ('dv','te','ex')]
out['standardised_15desc'],pb=run(*Xb,'standardised + 10 + 5 ionisation')
# C: as-published (unstandardised) for reference, from the stored npz
z=np.load(ROOT+'/work/clean_features.npz')
out['as_published'],pc=run(z['Xdv_c'],z['Xte_c'],z['Xex_c'],'as published (unstandardised)')

def boot(y,pa_,pb_,n=5000,seed=42):
    rng=np.random.default_rng(seed);d=[]
    for _ in range(n):
        i=rng.integers(0,len(y),len(y))
        if len(np.unique(y[i]))<2: continue
        d.append(roc_auc_score(y[i],pa_[i])-roc_auc_score(y[i],pb_[i]))
    d=np.array(d); return float(d.mean()),float(np.percentile(d,2.5)),float(np.percentile(d,97.5))
for lab,p1,p0 in [('standardised_vs_published',pa,pc),('ionisation_vs_standardised',pb,pa)]:
    d,lo,hi=boot(yex,p1,p0)
    out[lab]={'delta':d,'ci':[lo,hi],'separable':bool(lo>0 or hi<0)}
    print('  Δ %-28s %+.4f [%+.4f, %+.4f] %s'%(lab,d,lo,hi,'separable' if (lo>0 or hi<0) else 'not separable'),flush=True)

json.dump(out,open(ROOT+'/results/standardisation_ablation.json','w'),indent=1)
print('\nsaved standardisation_ablation.json')
