"""Pure metadata contract: no Torch import, research access or approval side effect."""
import hashlib,json
from pathlib import Path

PARENT='3490f9711cf39505722072497dda8153bfa49767f11260e02dcae12017d63b08'
SELECTION='36b02a67d3ab06804792c5d5743236e61ab8f45aeb17fd9f73069087f7275cb8'
PREPARATION='47f5a6a740b405d758431e56f241993e48c93b330fa7859d93a7133378fa120a'
WEIGHTS={18:'38b3a17444c5c18a795355a868e6a2d9ce9318054067f5373d337f55c0f61da3',10:'2824951eaef30e115929b89bec72342380b08d14792bda1449974ac849e497c0'}
POLICY={'version':'medicalnet-guarded-v1','backbones':[18,10],'bridges':[[64,32],[128,64],[512,128]],'bridge_seed':2718,'seed':17,'frozen_bn_statistics':True,'bn_affine_trainable':True,'shortcut_A':'forward-preserving gradient restoration','maxpool_backward':'27 fixed offsets; first-order only','pyramid':'layer1,layer2,layer4; nonoverlap mean2 last scale','precision':'FP32','amp':False,'strict_determinism':True,'selection_sha256':SELECTION,'reference_release':'HOLD','no_resume':True}
GIB=1024**3
LIMITS={'setup_seconds':180,'run_seconds':900,'validation_seconds':180,'total_seconds':1080,'rss_bytes':8*GIB,'allocator_bytes':4*GIB,'gpu_bytes':6*GIB,'sample_seconds':1.,'min_ram_bytes':32*GIB,'reserve_ram_bytes':8*GIB,'min_gpu_bytes':10*GIB,'min_disk_bytes':20*GIB}
GENERATED_LIMITS={**LIMITS,'run_seconds':720,'total_seconds':900}

def require(ok,why):
 if not ok:raise ValueError(why)
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p,expected=None):
 p=Path(p).absolute();require(p.is_file() and not any(x.is_symlink() for x in [p,*p.parents]),'Unsafe/missing metadata')
 require(p.stat().st_size<=16*1024**2,'Metadata admission limit')
 if expected:require(digest(p)==expected,'Metadata binding mismatch')
 return json.loads(p.read_text())
def transfer(depth):
 require(type(depth) is int and depth in WEIGHTS,'Only reviewed depths')
 return {**POLICY,'depth':depth,'checkpoint_sha256':WEIGHTS[depth]}
def identity(depth,config,source_hash,selection_hash,run_id,disposition_hash=None):
 return {'run_id':run_id,'config_hash':canonical(config),'transfer':transfer(depth),'transfer_sha256':canonical(transfer(depth)),'selection_sha256':selection_hash,'source_binding_sha256':source_hash,'seed':17,'amp':False,'device':'cuda','disposition_sha256':disposition_hash,'preparation_sha256':PREPARATION}
def disposition(d,source_hash):
 require(d.get('status')=='ADOPTED' and d.get('scope')=='medicalnet-matched-engineering-v1','Separate adopted disposition required')
 require(d.get('reference_release')=='HOLD' and d.get('source_binding_sha256')==source_hash,'Disposition source/HOLD mismatch')
 require(d.get('selection_sha256')==SELECTION and d.get('preparation_sha256')==PREPARATION,'Disposition input mismatch')
 require(d.get('policy')==POLICY and d.get('limits')==LIMITS,'Disposition policy changed')
 from .engineering import HOLD_IDS
 require(set(HOLD_IDS).issubset(d.get('acknowledged_unresolved_holds',[])),'Six unresolved holds required')
 for k in ('label_guided_sampling','prior_exposure_uncertainty','sampled_monitoring','not_scientific_approval','fp32_not_amp_coverage'):
  require(d.get(k) is True,'Missing acknowledgement: '+k)
 require(all(isinstance(d.get(k),str) and d[k].strip() for k in ('supplied_by','instruction','date')),'Actual adoption attribution missing')
 return d
def provenance(p):
 # A signed-off mirror name or an approval boolean is not authentication.
 require(p.get('status')=='OFFICIAL_BYTES_MATCH','Official binary authentication unresolved')
 require(p.get('distribution_url')=='https://drive.google.com/file/d/13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG/view','Unreviewed distribution')
 require(set(p.get('members',{}))=={'10','18'},'Both artifacts required')
 for d,h in WEIGHTS.items():require(p['members'][str(d)]['sha256']==h and p['members'][str(d)]['official_member'].endswith('/resnet_'+str(d)+'.pth'),'Official artifact mismatch')
 require(len(p.get('archive_sha256',''))==64 and p.get('method')=='streamed-official-archive-member-comparison','Missing official archive comparison')
 return p
def generated_acceptance(d,depth,source_hash):
 require(d.get('status')=='PASS' and d.get('depth')==depth and d.get('source_binding_sha256')==source_hash,'Generated qualification mismatch')
 require(d.get('transfer')==transfer(depth) and d.get('device')=='cuda','Wrong generated identity')
 require(d.get('stages')==['pool_ties_backward','shortcut_backward','full_hybrid_repeat'],'Incomplete generated stages')
 require(d.get('exact_repeat') is True and d.get('updates_per_repeat')==1 and d.get('skips')==0,'Generated optimizer/repeat failed')
 return d
