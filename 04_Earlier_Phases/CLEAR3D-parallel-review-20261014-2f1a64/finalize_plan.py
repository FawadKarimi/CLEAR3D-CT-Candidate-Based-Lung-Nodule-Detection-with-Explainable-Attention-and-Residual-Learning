import hashlib,json,importlib.util,difflib
from pathlib import Path
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n')
spec=importlib.util.spec_from_file_location('plan',R/'D-plan/fixed_batch.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a=R/'accepted';old=json.loads((a/'training-binding.json').read_text());sample=json.loads((a/'runs/diagnostic-epochs18-001/research/state/sampling-0.json').read_text())
server='/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64'
b={'root':server,'scope':m.SCOPE,'policy':m.POLICY,'limits':m.LIMITS,'output':server+'/runs/fixed16-001','indices':m.choose(sample['source_indices']), 'parent_binding':server+'/accepted/training-binding.json','parent_binding_sha256':sha(a/'training-binding.json'),'configuration':{'path':server+'/accepted/configuration-resnet18.json','sha256':sha(a/'configuration-resnet18.json')},'new_sources':{'D-plan/fixed_batch.py':sha(R/'D-plan/fixed_batch.py')},'accepted_failure_zip_sha256':'446915a67d7faff15b9ba4e00eb0d2ed1a0121125a850136e57d247e23f443fa','parent_inputs':old['diagnostic_data']['inputs'],'source_count':133,'interpreter':old['interpreter'],'interpreter_sha256':old['interpreter_sha256'],'pins':old['pins'],'gpu_uuid':'GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90'}
save(R/'D-plan/fixed16-binding.json',b);h=sha(R/'D-plan/fixed16-binding.json')
d={'status':'DRAFT','scope':m.SCOPE,'binding_sha256':h,'policy':m.POLICY,'limits':m.LIMITS,'reference_release':'HOLD','cap4000_policy':'NOT_ADOPTED','supplied_by':None,'instruction':None,'date':None,'label_guided':True,'prior_exposure_uncertain':True,'pretraining_overlap_uncertain':True,'sampled_monitoring':True,'no_scientific_approval':True,'train_only':True,'holds':['reference_origin','excluded_reference_outliers','geometry_and_source_scope','protocol_evaluation_disclosure','prior_test_access','independent_reference_review'],'proposed_text':'DRAFT ONLY: Permit one separately reviewed fixed16 train-only diagnostic; accepted training sidecar and draw metadata, official weights, and only selected training image.npy/valid.npy pairs. No validation pixels, threshold selection, research-policy adoption or reference approval. No retry/resume or budget enlargement. Sampling and constant-LR/no-augmentation diagnostic choices are explicit and confer no generalization claim. This package authenticates neither authorship nor approval.'}
save(R/'D-plan/DRAFT-disposition.json',d)
py=old['interpreter']
command=f'''#!/bin/bash
set -u
ROOT='{server}'
ADOPTED="${{1:?separate adopted disposition path required}}"
ADOPTED_SHA="${{2:?exact adopted disposition SHA256 required}}"
test -d "$ROOT/runs" && test ! -L "$ROOT/runs" || exit 2
test ! -e "$ROOT/runs/fixed16-001" || exit 2
test ! -e "$ROOT/runs/fixed16-001.shell.log" || exit 2
test ! -e "$ROOT/runs/fixed16-001.shell.exit" || exit 2
set -o noclobber
env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH -u LD_PRELOAD PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90 '{py}' -B "$ROOT/D-plan/fixed_batch.py" --execute --binding "$ROOT/D-plan/fixed16-binding.json" --binding-sha256 '{h}' --disposition "$ADOPTED" --disposition-sha256 "$ADOPTED_SHA" --out "$ROOT/runs/fixed16-001" --acknowledge-shared-gpu-permission --acknowledge-sampled-risk > "$ROOT/runs/fixed16-001.shell.log" 2>&1
status=$?
printf '%s\\n' "$status" > "$ROOT/runs/fixed16-001.shell.exit"
exit "$status"
'''
(R/'D-plan/proposed-command.sh').write_text(command,encoding='utf-8',newline='\n')
patch=''
for rel in ['D-plan/fixed_batch.py','C-tests/test_model_path.py','C-tests/test_fixed_batch.py']:
 patch+=''.join(difflib.unified_diff([], (R/rel).read_text().splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+rel))
(R/'exact-additive.patch').write_text(patch,encoding='utf-8',newline='\n')
print(json.dumps({'binding_sha256':h,'indices':b['indices'],'patch_sha256':sha(R/'exact-additive.patch')}))
