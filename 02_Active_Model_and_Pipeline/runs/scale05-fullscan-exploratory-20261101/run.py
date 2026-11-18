import csv,json,math,os,sys,time,resource
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
sys.path.insert(0,'/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/src')
import numpy as np
from lidc3d.candidates import generate
base=Path('/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared')
rows=list(csv.reader(open('runs/fullscan-cap4000-dev-20261030-remaining-001/evaluation-recovery/froc4000/nodulesWithoutCandidate_cap4000.txt')))
config={'sigmas_mm':[.5,1,1.5,2,3],'threshold':.005,'nms_mm':3,'cap':4000,'gaussian_mode':'reflect','gaussian_truncate':4}
for pattern,target in [('scan_b568682347811817dc97',3.049),('scan_*70488*',2.79)]:
 f=next(base.glob(pattern)); rec=json.load(open(f/'record.json')); refs=[r for r in rows if r[0]==rec['series_uid'] and abs(float(r[5])/2-target)<.01]; assert len(refs)==1
 ref=refs[0]; center=np.array(list(map(float,ref[2:5]))); radius=float(ref[5])/2; aff=np.asarray(rec['affine']); start=time.monotonic()
 print(json.dumps({'event':'start','scan_id':f.name}),flush=True)
 image=np.load(f/'image.npy'); mask=np.load(f/'lung_mask.npy'); selected,diag=generate(image,mask,config,1.0); pool=diag.pop('precap_candidates'); best=None
 for rank,p in enumerate(pool,1):
  x=aff[:3,:3]@np.asarray(p['zyx'])+aff[:3,3]
  if np.linalg.norm(x-center)<radius: best=rank; break
 out={'event':'complete','scan_id':f.name,'seconds':round(time.monotonic()-start,2),'pool_size':len(pool),'rank':best,'diagnostics':diag,'scope':'exploratory two cases selected for prior misses; not a sensitivity estimate','reference_release':'HOLD'}
 print(json.dumps(out),flush=True)
