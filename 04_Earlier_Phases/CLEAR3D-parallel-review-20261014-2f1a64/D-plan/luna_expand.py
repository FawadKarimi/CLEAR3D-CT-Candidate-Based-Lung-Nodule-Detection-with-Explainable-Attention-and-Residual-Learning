import csv, json, os, sys
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0,'/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/work/src')
from lidc3d.geometry import crop
PREP='/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared'
ANN='/experiment-storage/chaidir/fawad-dsai/data/reference/luna16-zenodo-3723295/resolved-reference-v1/annotations.csv'
CAND='/experiment-storage/chaidir/fawad-dsai/data/reference/luna16-zenodo-3723295/resolved-reference-v1/candidates_V2.csv'
ROOT='/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64'
OUT=ROOT+'/runs/luna-expand-001'
os.makedirs(OUT, exist_ok=True)
VAL={'LIDC-IDRI-0020','LIDC-IDRI-0428','LIDC-IDRI-0634','LIDC-IDRI-0722','LIDC-IDRI-0865','LIDC-IDRI-0895'}
by={}
for name in os.listdir(PREP):
    d=os.path.join(PREP,name)
    if not os.path.isdir(d): continue
    rec=json.load(open(os.path.join(d,'record.json')))
    if rec['patient_id'] in VAL or rec.get('split') in ('val','test'): continue
    by[rec['series_uid']]=rec
nod=defaultdict(list)
for r in csv.DictReader(open(ANN)):
    rec=by.get(r['seriesuid'])
    if rec is None: continue
    geo=json.load(open(os.path.join(PREP, rec['scan_id'], 'geometry.json')))
    ijk=np.linalg.inv(np.array(geo['affine'], float)) @ np.array([float(r['coordX']), float(r['coordY']), float(r['coordZ']), 1.0])
    nod[rec['series_uid']].append({'patient': rec['patient_id'], 'scan': rec['scan_id'], 'zyx': [float(ijk[0]), float(ijk[1]), float(ijk[2])]})
print('SERIES_WITH_NODULES', len(nod), 'NODULES', sum(len(v) for v in nod.values()), flush=True)

def bright(arr):
    if arr.shape!=(48,48,48): return False
    inner=arr[16:32,16:32,16:32]
    shell=np.concatenate([arr[:16].ravel(), arr[32:].ravel(), arr[16:32,:16].ravel(), arr[16:32,32:].ravel(), arr[16:32,16:32,:16].ravel(), arr[16:32,16:32,32:].ravel()])
    thr=float(shell.mean()+shell.std())
    m=inner>=thr
    if not m[8,8,8]: return False
    seen=np.zeros(m.shape, bool); stack=[(8,8,8)]; seen[8,8,8]=True; n=0
    while stack:
        z,y,x=stack.pop(); n+=1
        for dz,dy,dx in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            a,b,c=z+dz,y+dy,x+dx
            if 0<=a<16 and 0<=b<16 and 0<=c<16 and m[a,b,c] and not seen[a,b,c]:
                seen[a,b,c]=True; stack.append((a,b,c))
    return n>=10

passed=[]
for i, (uid, items) in enumerate(nod.items(), 1):
    d=os.path.join(PREP, items[0]['scan'])
    image=np.load(os.path.join(d,'image.npy'), mmap_mode='r')
    valid=np.load(os.path.join(d,'valid.npy'), mmap_mode='r')
    for it in items:
        z,y,x=[int(round(v)) for v in it['zyx']]
        if not (0<=z<image.shape[0] and 0<=y<image.shape[1] and 0<=x<image.shape[2]): continue
        if valid[z,y,x]==0: continue
        arr,_=crop(np.asarray(image), it['zyx'], 48)
        if not bright(np.asarray(arr)): continue
        pad,_=crop(np.asarray(image), it['zyx'], 24)
        if np.asarray(pad).shape!=(24,24,24): continue
        it['pad']=np.asarray(pad, np.float32)
        passed.append(it)
    if i%100==0: print('AUDITED', i, 'PASS', len(passed), flush=True)
print('AUDIT_DONE', len(passed), flush=True)
by_pat=defaultdict(list)
for it in passed: by_pat[it['patient']].append(it)
hold='LIDC-IDRI-0694'
patients=sorted(by_pat, key=lambda p: (len(by_pat[p]), p))
stop=[]; nstop=0
if hold in by_pat:
    stop.append(hold); nstop=len(by_pat[hold])
for p in patients:
    if p==hold: continue
    if nstop>=30: break
    stop.append(p); nstop+=len(by_pat[p])
stop=set(stop)
grad=[p for p in by_pat if p not in stop]
ng=sum(len(by_pat[p]) for p in grad)
print('GATE', 'grad', ng, 'stop', nstop, 'stop_patients', len(stop), flush=True)
if ng<200 or nstop<30:
    json.dump({'stop':'insufficient_audit_pass','grad':ng,'stop_pos':nstop}, open(OUT+'/summary.json','w'), indent=2)
    print('STOP_GATE', ng, nstop, flush=True)
    raise SystemExit(0)
grad_items=[it for p in grad for it in by_pat[p]]
stop_items=[it for p in stop for it in by_pat[p]]
grad_series={it['scan'] for it in grad_items}
# map scan to series via record
scan_to_series={rec['scan_id']: uid for uid, rec in by.items()}
series_grad={scan_to_series[s] for s in grad_series}
rng=np.random.default_rng(17)
neg_xyz=defaultdict(list)
for r in csv.DictReader(open(CAND)):
    if r['class']!='0' or r['seriesuid'] not in series_grad: continue
    neg_xyz[r['seriesuid']].append((float(r['coordX']), float(r['coordY']), float(r['coordZ'])))
need=7*8*len(grad_items)
pool=[]
for uid, pts in neg_xyz.items():
    if not pts: continue
    take=min(len(pts), max(1, need//max(len(neg_xyz),1)))
    sel=rng.choice(len(pts), size=min(take, len(pts)), replace=False)
    for j in sel: pool.append((uid, pts[int(j)]))
rng.shuffle(pool)
pool=pool[:need]
print('NEG_POOL', len(pool), 'NEED', need, flush=True)
neg_crops=[]
by_uid=defaultdict(list)
for uid, xyz in pool: by_uid[uid].append(xyz)
for nscan, (uid, xyzs) in enumerate(by_uid.items(), 1):
    if nscan % 50 == 0: print('NEG_SCAN', nscan, 'crops', len(neg_crops), flush=True)
    rec=by[uid]
    image=np.load(os.path.join(PREP, rec['scan_id'], 'image.npy'), mmap_mode='r')
    geo=json.load(open(os.path.join(PREP, rec['scan_id'], 'geometry.json')))
    Ainv=np.linalg.inv(np.array(geo['affine'], float))
    for x,y,z in xyzs:
        ijk=Ainv @ np.array([x,y,z,1.0])
        arr,_=crop(np.asarray(image), [float(ijk[0]), float(ijk[1]), float(ijk[2])], 18)
        arr=np.asarray(arr)
        if arr.shape!=(18,18,18): continue
        neg_crops.append(arr.astype(np.float32))
print('NEG_CROPS', len(neg_crops), flush=True)
if len(neg_crops)<7*len(grad_items):
    json.dump({'stop':'insufficient_negatives','neg':len(neg_crops)}, open(OUT+'/summary.json','w'), indent=2)
    print('STOP_NEG', flush=True); raise SystemExit(0)
pos_bank=np.stack([it['pad'] for it in grad_items]).astype(np.float32)
neg_bank=np.stack(neg_crops).astype(np.float32)
stop_pos=np.stack([np.asarray(it['pad'], np.float32)[3:21,3:21,3:21] for it in stop_items])
series_stop={scan_to_series[it['scan']] for it in stop_items}
stop_xyz=[]
for r in csv.DictReader(open(CAND)):
    if r['class']!='0' or r['seriesuid'] not in series_stop: continue
    stop_xyz.append((r['seriesuid'], float(r['coordX']), float(r['coordY']), float(r['coordZ'])))
cap=min(len(stop_xyz), 20*len(stop_items))
sel=rng.choice(len(stop_xyz), size=cap, replace=False)
by_stop=defaultdict(list)
for j in sel:
    uid,x,y,z=stop_xyz[int(j)]; by_stop[uid].append((x,y,z))
stop_neg=[]
for uid, xyzs in by_stop.items():
    rec=by[uid]
    image=np.load(os.path.join(PREP, rec['scan_id'], 'image.npy'), mmap_mode='r')
    geo=json.load(open(os.path.join(PREP, rec['scan_id'], 'geometry.json')))
    Ainv=np.linalg.inv(np.array(geo['affine'], float))
    for x,y,z in xyzs:
        ijk=Ainv @ np.array([x,y,z,1.0])
        arr=np.asarray(crop(np.asarray(image), [float(ijk[0]), float(ijk[1]), float(ijk[2])], 18)[0])
        if arr.shape==(18,18,18): stop_neg.append(arr.astype(np.float32))
if not stop_neg: raise SystemExit('no stop negatives')
stop_neg=np.stack(stop_neg)
for it in grad_items+stop_items: it.pop('pad', None)
print('BANKS', tuple(pos_bank.shape), tuple(neg_bank.shape), tuple(stop_pos.shape), tuple(stop_neg.shape), flush=True)
np.savez(OUT+'/banks.npz', pos=pos_bank, neg=neg_bank, stop_pos=stop_pos, stop_neg=stop_neg)
meta={'grad_pos': int(len(pos_bank)), 'stop_pos': int(len(stop_pos)), 'stop_neg': int(len(stop_neg)), 'grad_patients': sorted(grad), 'stop_patients': sorted(stop)}
json.dump(meta, open(OUT+'/pool.json','w'), indent=2)
free=os.popen('nvidia-smi --query-gpu=memory.free,uuid --format=csv,noheader,nounits').read()
ok=any(p[1]=='GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90' and int(p[0])>=10240 for p in ([x.strip() for x in line.split(',')] for line in free.splitlines()) if len(p)==2)
if not ok:
    json.dump({'stop':'low_gpu', **meta}, open(OUT+'/summary.json','w'), indent=2)
    print('LOWMEM', flush=True); raise SystemExit(0)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ch=(1,32,64,128,128); layers=[]
        for a,b in zip(ch, ch[1:]):
            layers += [nn.Conv3d(a,b,3,padding=1), nn.ReLU(inplace=True), nn.MaxPool3d(2, ceil_mode=True)]
        self.features=nn.Sequential(*layers); self.fc=nn.Linear(128*8, 2)
        for m in self.modules():
            if isinstance(m,(nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self,x): return self.fc(self.features(x).flatten(1))
def zscore(batch, noise):
    m=batch.mean(axis=(1,2,3), keepdims=True); s=batch.std(axis=(1,2,3), keepdims=True)
    out=(batch-m)/(s+1e-6)
    if noise: out=out+rng.normal(0,0.125,out.shape).astype(np.float32)
    return out
def flags(batch):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(batch),128):
            x=torch.from_numpy(zscore(batch[i:i+128], False)[:,None]).to(device)
            out.append(torch.softmax(model(x).float(),1)[:,1].ge(0.5).cpu().numpy())
    return np.concatenate(out)
device=torch.device('cuda'); torch.manual_seed(17); model=Net().to(device)
opt=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([0.25,1.75], device=device))
best=None; best_state=None; stall=0
log=open(OUT+'/train-log.jsonl','w')
npos=len(pos_bank); copies=8
for epoch in range(1,16):
    model.train(); order_pos=rng.permutation(npos)
    xs=[]; ys=[]
    for i in order_pos:
        for _ in range(copies):
            off=rng.integers(-2,3,size=3)
            c0,c1,c2=(12+int(v) for v in off)
            xs.append(pos_bank[i, c0-9:c0+9, c1-9:c1+9, c2-9:c2+9]); ys.append(1)
    take=rng.choice(len(neg_bank), size=7*len(xs), replace=len(neg_bank)<7*len(xs))
    for j in take:
        xs.append(neg_bank[int(j)]); ys.append(0)
    order=rng.permutation(len(xs)); total=0.0; steps=0
    for s in range(0, len(order)-127, 128):
        idx=order[s:s+128]
        batch=zscore(np.stack([xs[j] for j in idx]), True)
        y=torch.tensor([ys[j] for j in idx], device=device)
        opt.zero_grad(set_to_none=True)
        loss=loss_fn(model(torch.from_numpy(batch[:,None]).to(device)), y)
        if not torch.isfinite(loss): raise SystemExit('nonfinite')
        loss.backward(); opt.step(); total+=float(loss.detach()); steps+=1
    sp=flags(stop_pos); sn=flags(stop_neg)
    tp=int(sp.sum()); fn=int((~sp).sum()); fp=int(sn.sum()); tn=int((~sn).sum())
    rec={'epoch':epoch,'loss':total/max(steps,1),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'sum':fp+fn}
    log.write(json.dumps(rec)+chr(10)); log.flush()
    print('EPOCH', epoch, 'loss', round(rec['loss'],4), 'fp', fp, 'fn', fn, 'sum', fp+fn, flush=True)
    rank=(fp+fn, fp, rec['loss'])
    if best is None or rank<best['rank']:
        best={'rank':rank,'epoch':epoch,'tp':tp,'fp':fp,'fn':fn,'tn':tn}; stall=0
        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    else:
        stall+=1
    if stall>=5: break
log.close()
torch.save({'epoch':best['epoch'],'state':best_state}, OUT+'/best-state.pt')
model.load_state_dict(best_state)
import lidc3d.diagnostic_training as c
from lidc3d.data import load_array
parent=json.load(open(c.DATA['inputs']['parent_selection']['path']))
scans={e['scan']['scan_id']: e['scan'] for e in parent['entries']}
val_ids=[]; seen=set()
for line in open(ROOT+'/runs/resnet18-constlr-001/val-predictions.jsonl'):
    r=json.loads(line)
    if r['epoch']==2 and r['id'] not in seen:
        seen.add(r['id']); val_ids.append((r['id'], int(r['label'])))
byid={}
for entry in json.load(open(c.DATA['inputs']['validation']['path'])):
    for item in entry['selected']:
        row=dict(item['row']); row['label']=int(item['label']); byid[row['id']]=row
cache={}; tp=fp=fn=tn=0
model.eval()
with torch.no_grad():
    for id_, label in val_ids:
        item=byid[id_]
        assert int(item['label'])==label and scans[item['scan_id']]['split']=='val'
        path=scans[item['scan_id']]['image_path']
        if path not in cache: cache[path]=load_array(path)
        arr=np.asarray(crop(cache[path], [float(v) for v in item['zyx']], 18)[0]).astype(np.float32)
        arr=(arr-arr.mean())/(arr.std()+1e-6)
        p=bool(torch.softmax(model(torch.from_numpy(arr[None,None]).to(device)).float(),1)[0,1].ge(0.5).item())
        if label==1 and p: tp+=1
        elif label==1: fn+=1
        elif p: fp+=1
        else: tn+=1
n=tp+fp+fn+tn
acc=(tp+tn)/n
prec=tp/(tp+fp) if tp+fp else 0.0
recall=tp/(tp+fn) if tp+fn else 0.0
pos_f1=2*prec*recall/(prec+recall) if prec+recall else 0.0
np_ = tn/(tn+fn) if tn+fn else 0.0
nr = tn/(tn+fp) if tn+fp else 0.0
neg_f1=2*np_*nr/(np_+nr) if np_+nr else 0.0
macro=(pos_f1+neg_f1)/2
summary={'stop_epoch':best['epoch'],'pool':meta,'stop_fp':best['fp'],'stop_fn':best['fn'],'val':{'tp':tp,'fp':fp,'fn':fn,'tn':tn,'acc':acc,'pos_f1':pos_f1,'macro_f1':macro,'precision':prec,'recall':recall},'threshold':0.5}
json.dump(summary, open(OUT+'/summary.json','w'), indent=2)
print('DONE', 'epoch', best['epoch'], 'VAL_ACC', round(acc,4), 'VAL_F1', round(pos_f1,4), 'MACRO', round(macro,4), 'FP', fp, 'FN', fn, 'TP', tp, 'TN', tn, flush=True)
