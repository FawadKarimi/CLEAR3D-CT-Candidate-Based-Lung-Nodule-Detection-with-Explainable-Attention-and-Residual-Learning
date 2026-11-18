"""Exploratory LUNA candidate loader; patient-grouped fit/dev only."""
import json, hashlib
from pathlib import Path
import numpy as np
from lidc3d.data import CandidateDataset

SPLIT_SHA='2ffc30511188ca27686d7c2d5a73fa2723f822a42a62a9c159987c7e0d54ede1'
EXCLUDED={'LIDC-IDRI-0138','LIDC-IDRI-0029'}

def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def get_split(path, expected=SPLIT_SHA):
    if hash_file(path)!=expected: raise ValueError('split hash mismatch')
    s=json.loads(Path(path).read_text())
    fit,dev=set(s['fit'])-EXCLUDED,set(s['dev'])-EXCLUDED
    if fit & dev: raise ValueError('patient overlap')
    return {'fit':fit,'dev':dev}

def candidate_rows(path,role,allowed):
    if role not in ('fit','dev'): raise ValueError('unsupported role')
    rows=[]
    with open(path) as f:
        for line_no,line in enumerate(f,1):
            if len(line)>1048576: raise ValueError('oversized record')
            r=json.loads(line)
            if r['patient'] in EXCLUDED: continue
            if r['patient'] not in allowed: raise ValueError('patient outside split')
            if r['label'] not in (0,1): raise ValueError('invalid label')
            if not all(np.isfinite(float(r[k])) for k in ('x','y','z')): raise ValueError('invalid coordinate')
            rows.append({**r,'id':f'{role}:{line_no}'})
    return rows

def build_dataset(prepared,rows):
    root=Path(prepared).resolve(); scans={}; converted=[]
    for row in rows:
        sid=row['scan']
        if Path(sid).name!=sid or not sid.startswith('scan_'): raise ValueError('invalid scan id')
        if sid not in scans:
            d=root/sid
            if d.is_symlink() or d.resolve().parent!=root: raise ValueError('invalid scan path')
            record=json.loads((d/'record.json').read_text())
            if record['patient_id']!=row['patient'] or record['series_uid']!=row['series']: raise ValueError('scan identity mismatch')
            geo=json.loads((d/'geometry.json').read_text()); a=np.asarray(geo['affine'],dtype=float)
            if a.shape!=(4,4) or not np.isfinite(a).all(): raise ValueError('invalid affine')
            image=np.load(d/'image.npy',mmap_mode='r',allow_pickle=False)
            valid=np.load(d/'valid.npy',mmap_mode='r',allow_pickle=False)
            if image.ndim!=3 or image.shape!=valid.shape: raise ValueError('incompatible volumes')
            scans[sid]={'scan_id':sid,'image_path':str(d/'image.npy'),'valid_path':str(d/'valid.npy'),'inverse':np.linalg.inv(a),'patient':row['patient'],'series':row['series'],'shape':image.shape}
        s=scans[sid]
        if s['patient']!=row['patient'] or s['series']!=row['series']: raise ValueError('inconsistent identity')
        zyx=(s['inverse']@np.array([row['x'],row['y'],row['z'],1.]))[:3]
        if not np.isfinite(zyx).all(): raise ValueError('invalid voxel coordinate')
        center=np.floor(zyx+0.5).astype(int)
        if (center<0).any() or (center>=np.array(s['shape'])).any(): raise ValueError('candidate center outside image')
        converted.append({'scan_id':sid,'id':row['id'],'zyx':zyx.tolist(),'label':row['label']})
    return CandidateDataset(list(scans.values()),converted,48,False,17)

def filter_excluded(rows, references):
    """Exclude class-0 candidates within excluded spheres; retain all positives.

    references maps series UID to iterable of (LPS center, radius in mm).
    Exact-radius points are not inside under the project's strict convention.
    """
    from collections import defaultdict
    from scipy.spatial import cKDTree
    grouped=defaultdict(list)
    for i,r in enumerate(rows):
        if r['label']==0: grouped[r['series']].append(i)
    dropped=set()
    for uid,indices in grouped.items():
        regs=references.get(uid,())
        if not regs: continue
        xyz=np.asarray([[rows[i][k] for k in ('x','y','z')] for i in indices],dtype=float)
        tree=cKDTree(xyz)
        for center,radius in regs:
            center=np.asarray(center,dtype=float);radius=float(radius)
            if center.shape!=(3,) or not np.isfinite(center).all() or not np.isfinite(radius) or radius<=0:raise ValueError('invalid excluded reference')
            for j in tree.query_ball_point(center,radius):
                if np.linalg.norm(xyz[j]-center)<radius:dropped.add(indices[j])
    return [r for i,r in enumerate(rows) if i not in dropped], len(dropped)
