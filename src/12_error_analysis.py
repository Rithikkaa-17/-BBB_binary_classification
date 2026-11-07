"""R2 point 1: what the errors actually are. Polarity distribution, label-origin
strata, and an explicitly-sourced efflux-substrate flag."""
import warnings, json, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors as rdm, Lipinski
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, matthews_corrcoef
RDLogger.DisableLog('rdApp.*')
import os as _o; ROOT=_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))
z=np.load(ROOT+'/work/clean_features.npz')
ydv,yte,yex=z['ydv'],z['yte'],z['yex']
te=pd.read_csv(ROOT+'/work/clean_test.csv'); ex=pd.read_csv(ROOT+'/work/clean_external.csv')
SEEDS=[7,21,42,84,123]; out={}

# final predictor: seed-averaged probability, median validation threshold (0.86 per paper)
P=[]
for sd in SEEDS:
    m=RandomForestClassifier(n_estimators=500,class_weight='balanced',random_state=sd,n_jobs=-1).fit(z['Xdv_c'],ydv)
    P.append(m.predict_proba(z['Xte_c'])[:,1])
pte=np.mean(P,axis=0)
fr=json.load(open(ROOT+'/results/final_results.json'))
THR=fr['models']['RF + Fused']['threshold']
pred=(pte>=THR).astype(int)
err=pred!=yte
print('threshold %.2f  errors %d / %d'%(THR,err.sum(),len(yte)),flush=True)

# ---- descriptors for the test set ----
rows=[]
for s in te['smiles']:
    m=Chem.MolFromSmiles(s)
    rows.append([Descriptors.MolWt(m),rdm.CalcTPSA(m),Crippen.MolLogP(m),
                 Lipinski.NumHDonors(m),Lipinski.NumHAcceptors(m),m.GetNumHeavyAtoms()])
D=pd.DataFrame(rows,columns=['MolWt','TPSA','LogP','HBD','HBA','HAC'])
D['name']=te['name'].values; D['y']=yte; D['p']=pte; D['err']=err

# ---- is "intermediate polarity" true? ----
print('\n=== TPSA distribution ===',flush=True)
for lab,sub in [('correct',D[~D.err]),('errors',D[D.err])]:
    q=sub.TPSA.quantile([0,.25,.5,.75,1]).values
    print('  %-8s n=%3d  TPSA min %.1f q1 %.1f med %.1f q3 %.1f max %.1f'%(lab,len(sub),*q),flush=True)
    out.setdefault('tpsa',{})[lab]={'n':int(len(sub)),'min':float(q[0]),'q1':float(q[1]),
                                    'median':float(q[2]),'q3':float(q[3]),'max':float(q[4])}
# define the interquartile band of the correct set as "intermediate"
lo,hi=D[~D.err].TPSA.quantile([.25,.75]).values
inside=((D[D.err].TPSA>=lo)&(D[D.err].TPSA<=hi)).mean()
inside_all=((D.TPSA>=lo)&(D.TPSA<=hi)).mean()
out['intermediate_band']={'lo':float(lo),'hi':float(hi),
                          'frac_errors_inside':float(inside),'frac_all_inside':float(inside_all)}
print('  interquartile band of correct predictions: TPSA %.1f-%.1f'%(lo,hi),flush=True)
print('  errors inside that band: %.1f%%   (all molecules: %.1f%%)'%(100*inside,100*inside_all),flush=True)

print('\n=== every error, by TPSA ===',flush=True)
E=D[D.err].sort_values('TPSA')
for _,r in E.iterrows():
    print('  %-34s y=%d p=%.2f  TPSA %6.1f  MW %6.1f  LogP %+.2f  HBD %d'%(
        str(r['name'])[:34],r.y,r.p,r.TPSA,r.MolWt,r.LogP,r.HBD),flush=True)
out['errors']=[{'name':str(r['name']),'y':int(r.y),'p':float(r.p),'TPSA':float(r.TPSA),
                'MolWt':float(r.MolWt),'LogP':float(r.LogP),'HBD':int(r.HBD)} for _,r in E.iterrows()]

# ---- efflux-substrate / CNS flags (literature-curated, name-matched) ----
PGP={'digoxin','dabigatran','fexofenadine','loperamide','domperidone','cyclosporine','cyclosporin',
     'vinblastine','vincristine','paclitaxel','doxorubicin','daunorubicin','etoposide','quinidine',
     'verapamil','risperidone','cetirizine','colchicine','erythromycin','ritonavir','saquinavir',
     'indinavir','nelfinavir','amprenavir','tacrolimus','sirolimus','topotecan','irinotecan',
     'mitoxantrone','talinolol','ondansetron','morphine','methadone','amitriptyline','nortriptyline',
     'imatinib','phenytoin','rifampin','rifampicin','atorvastatin','ranitidine','cimetidine'}
CNS ={'diazepam','lorazepam','midazolam','phenobarbital','phenytoin','carbamazepine','valproate',
      'haloperidol','chlorpromazine','clozapine','olanzapine','risperidone','fluoxetine','sertraline',
      'paroxetine','citalopram','amitriptyline','imipramine','nortriptyline','caffeine','nicotine',
      'morphine','codeine','fentanyl','methadone','ketamine','propofol','thiopental','levodopa',
      'donepezil','memantine','zolpidem','buspirone','lithium','gabapentin','topiramate','lamotrigine'}
def flag(n,S):
    n=str(n).lower()
    return any(k in n for k in S)
D['pgp']=D.name.apply(lambda n: flag(n,PGP)); D['cns']=D.name.apply(lambda n: flag(n,CNS))
print('\n=== annotation coverage on the 248-molecule internal test set ===',flush=True)
print('  documented efflux substrates matched: %d'%D.pgp.sum(),flush=True)
print('  CNS-active drugs matched           : %d'%D.cns.sum(),flush=True)
for lab,col in [('efflux-substrate','pgp'),('CNS-active','cns')]:
    sub=D[D[col]]
    if len(sub):
        print('  %-18s n=%3d  error rate %.3f  (rest %.3f)'%(lab,len(sub),sub.err.mean(),D[~D[col]].err.mean()),flush=True)
        out.setdefault('flag_strata',{})[col]={'n':int(len(sub)),'error_rate':float(sub.err.mean()),
                                               'rest_error_rate':float(D[~D[col]].err.mean()),
                                               'names':[str(x) for x in sub.name]}

# ---- external: error / performance by label-origin group ----
print('\n=== external set: performance by B3DB source group ===',flush=True)
Pe=[]
for sd in SEEDS:
    m=RandomForestClassifier(n_estimators=500,class_weight='balanced',random_state=sd,n_jobs=-1).fit(z['Xdv_c'],ydv)
    Pe.append(m.predict_proba(z['Xex_c'])[:,1])
pex=np.mean(Pe,axis=0); prede=(pex>=THR).astype(int)
ex=ex.assign(p=pex,pred=prede,err=prede!=yex,y=yex)
for g,sub in ex.groupby('group'):
    if sub.y.nunique()<2: continue
    r=roc_auc_score(sub.y,sub.p)
    print('  group %-2s n=%4d  prevalence %.3f  ROC-AUC %.4f  error rate %.3f'%(g,len(sub),sub.y.mean(),r,sub.err.mean()),flush=True)
    out.setdefault('external_by_group',{})[str(g)]={'n':int(len(sub)),'prevalence':float(sub.y.mean()),
                                                     'roc':float(r),'error_rate':float(sub.err.mean())}
# crossed with scaffold novelty
print('\n=== external: source group x scaffold novelty ===',flush=True)
for g,sub in ex.groupby('group'):
    for nov,s2 in sub.groupby(~sub.scaffold_in_dev.astype(bool)):
        if s2.y.nunique()<2 or len(s2)<25: continue
        print('  group %-2s %-12s n=%4d  ROC-AUC %.4f'%(g,'novel' if nov else 'overlapping',len(s2),roc_auc_score(s2.y,s2.p)),flush=True)
        out.setdefault('group_x_novelty',{})['%s_%s'%(g,'novel' if nov else 'overlap')]={
            'n':int(len(s2)),'roc':float(roc_auc_score(s2.y,s2.p))}
json.dump(out,open(ROOT+'/results/error_analysis.json','w'),indent=1)
print('\nsaved error_analysis.json')
