import json, hashlib, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import fixed_batch_fit as f
root=pathlib.Path('/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64')
split=pathlib.Path('/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-comparison-protocol-20261025/runs/phase-cf/split.json')
expected='2ffc30511188ca27686d7c2d5a73fa2723f822a42a62a9c159987c7e0d54ede1'
if f.digest(split)!=expected: raise ValueError('Current split hash changed')
old=root/'D-plan/fixed16-binding.json'
if f.digest(old)!='7293e9f5e967799ab5393d30ec0bc081e27f44883b2ce2317e5aa3b84b045ea1': raise ValueError('Original binding changed')
b=json.loads(old.read_text())
b.update(policy=f.POLICY,limits=f.LIMITS,indices=[27942,36041,27956,20756,19275,28178,39540,45206,18404,2971,42206,33331,345,25022,27079,41408],output=str(root/'runs/fixed16-current-fit-001'),current_split={'path':str(split),'sha256':expected,'bytes':split.stat().st_size},new_sources={k:f.digest(root/k) for k in ('D-fit/fixed_batch_fit.py','D-fit/fit_selection.py')})
p=root/'D-fit/current-fit-binding.json'
with p.open('x') as h: json.dump(b,h,sort_keys=True,indent=2);h.write('\n')
bh=f.digest(p)
d={'status':'ADOPTED','scope':f.SCOPE,'binding_sha256':bh,'policy':f.POLICY,'limits':f.LIMITS,'reference_release':'HOLD','cap4000_policy':'NOT_ADOPTED','label_guided':True,'prior_exposure_uncertain':True,'pretraining_overlap_uncertain':True,'sampled_monitoring':True,'no_scientific_approval':True,'train_only':True,'supplied_by':'Fawad Karimi','instruction':'ok do it','date':'2026-10-28','holds':['reference_origin','excluded_reference_outliers','geometry_and_source_scope','protocol_evaluation_disclosure','prior_test_access','independent_reference_review']}
q=root/'D-fit/current-fit-authorization.json'
with q.open('x') as h: json.dump(d,h,sort_keys=True,indent=2);h.write('\n')
print('BINDING',bh,'AUTHORIZATION',f.digest(q))
