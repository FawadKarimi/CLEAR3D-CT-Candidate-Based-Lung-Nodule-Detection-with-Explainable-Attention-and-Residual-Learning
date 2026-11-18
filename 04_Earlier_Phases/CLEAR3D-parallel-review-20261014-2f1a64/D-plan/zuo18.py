import json, os, sqlite3, sys
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0,'/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/work/src')
import lidc3d.diagnostic_training as c
from lidc3d.data import load_array
from lidc3d.geometry import crop
ROOT='/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64'
OUT=ROOT+'/runs/zuo18-001'
os.makedirs(OUT, exist_ok=True)
CLEAN={'scan_e2ef02265730531c459d:candidate:000268','scan_3ffbe7c95fa086f4a39e:candidate:000000','scan_c5973537991d725f03ab:candidate:000159','scan_aeeda85c3e8748887611:candidate:000227','scan_3386dd59b61402a85ab2:candidate:000000','scan_f86f7c1d51e9e7141921:candidate:000000','scan_f86f7c1d51e9e7141921:candidate:000019','scan_f86f7c1d51e9e7141921:candidate:000020','scan_62d1bce3a9980e7eb339:candidate:000131','scan_786987a82ce7e0172a06:candidate:001459'}
parent=json.load(open(c.DATA['inputs']['parent_selection']['path']))
scans={e['scan']['scan_id']: e['scan'] for e in parent['entries']}
db=sqlite3.connect(c.DATA['inputs']['sidecar']['path'])
def rows(label):
    out=[]
    for pid, payload in db.execute('SELECT id, payload FROM labels WHERE label=?', (label,)):
        item=json.loads(payload)
        item['label']=label
        item['patient']=scans[item['scan_id']]['patient_id']
        item['split']=scans[item['scan_id']]['split']
        out.append(item)
    return out
pos=[r for r in rows(1) if r['split']=='train' and r['id'] in CLEAN]
if len(pos)!=10: raise SystemExit('clean %s'%len(pos))
hold='LIDC-IDRI-0694'
hold_rows=[r for r in pos if r['patient']==hold]
if not hold_rows:
    raise SystemExit('hold patient has no clean positive')
if scans[[r for r in pos if r['patient']==hold][0]['scan_id']]['split']!='train':
    raise SystemExit('hold not train')
fit_pos=[r for r in pos if r['patient']!=hold]
neg=[r for r in rows(0) if r['split']=='train' and r['patient']!=hold]
hold_eval=[r for r in rows(0)+rows(1) if r['patient']==hold and r['split']=='train']
print('SPLIT', 'fit_pos', len(fit_pos), 'neg_pool', len(neg), 'hold_pos', len(hold_rows), 'hold_eval', len(hold_eval), 'hold', hold, flush=True)
cache={}
def volume(path):
    if path not in cache:
        cache[path]=load_array(path)
    return cache[path]
def crop18(item, offset, noise, rng):
    scan=scans[item['scan_id']]
    center=[float(item['zyx'][i])+float(offset[i]) for i in range(3)]
    arr,_=crop(volume(scan['image_path']), center, 18)
    arr=arr.astype(np.float32)
    arr=(arr-arr.mean())/(arr.std()+1e-6)
    if noise:
        arr=arr+rng.normal(0, 0.125, arr.shape).astype(np.float32)
    return arr
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ch=(1,32,64,128,128)
        layers=[]
        for a,b in zip(ch, ch[1:]):
            layers += [nn.Conv3d(a,b,3,padding=1), nn.ReLU(inplace=True), nn.MaxPool3d(2, ceil_mode=True)]
        self.features=nn.Sequential(*layers)
        self.fc=nn.Linear(128*2*2*2, 2)
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.fc(self.features(x).flatten(1))
device=torch.device('cuda')
torch.manual_seed(17)
model=Net().to(device)
opt=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
rng=np.random.default_rng(17)
def confusion(items, jitter):
    model.eval()
    tp=fp=fn=tn=0
    with torch.no_grad():
        for i in range(0, len(items), 64):
            batch=[]
            labels=[]
            for item in items[i:i+64]:
                off=rng.integers(-2,3,size=3) if jitter else (0,0,0)
                batch.append(crop18(item, off, False, rng))
                labels.append(int(item['label']))
            x=torch.from_numpy(np.stack(batch)[:,None]).to(device)
            pred=torch.softmax(model(x).float(),1)[:,1].ge(0.5).cpu().numpy()
            for p,y in zip(pred.tolist(), labels):
                if y==1 and p: tp+=1
                elif y==1 and not p: fn+=1
                elif y==0 and p: fp+=1
                else: tn+=1
    return tp, fp, fn, tn
best=None
best_state=None
stall=0
log=open(OUT+'/train-log.jsonl','w')
for epoch in range(1,16):
    model.train()
    samples=[]
    for item in fit_pos:
        for _ in range(8):
            samples.append((item, rng.integers(-2,3,size=3), True))
    npos=len(samples)
    pick=rng.choice(len(neg), size=7*npos, replace=False)
    for j in pick:
        samples.append((neg[int(j)], (0,0,0), True))
    rng.shuffle(samples)
    w=torch.tensor([0.25, 1.75], device=device)
    loss_fn=nn.CrossEntropyLoss(weight=w)
    total=0.0; steps=0
    for i in range(0, len(samples)-127, 128):
        chunk=samples[i:i+128]
        batch=np.stack([crop18(it, off, True, rng) for it,off,nz in chunk])
        y=torch.tensor([int(it['label']) for it,_,_ in chunk], device=device)
        opt.zero_grad(set_to_none=True)
        loss=loss_fn(model(torch.from_numpy(batch[:,None]).to(device)), y)
        if not torch.isfinite(loss):
            raise SystemExit('nonfinite')
        loss.backward()
        opt.step()
        total += float(loss.detach().cpu()); steps += 1
    tp,fp,fn,tn=confusion(hold_eval, False)
    rec={'epoch':epoch,'loss':total/max(steps,1),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'sum':fp+fn}
    log.write(json.dumps(rec)+chr(10)); log.flush()
    print('EPOCH', epoch, 'loss', round(rec['loss'],4), 'hold_fp', fp, 'hold_fn', fn, 'sum', fp+fn, flush=True)
    rank=(fp+fn, fp, rec['loss'])
    if best is None or rank<best['rank']:
        best={'rank':rank, 'epoch':epoch, 'tp':tp, 'fp':fp, 'fn':fn, 'tn':tn}
        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        stall=0
    else:
        stall+=1
    if stall>=5:
        break
log.close()
torch.save({'epoch':best['epoch'], 'state':best_state}, OUT+'/best-state.pt')
model.load_state_dict(best_state)
val_ids=[]; seen=set()
for line in open(ROOT+'/runs/resnet18-constlr-001/val-predictions.jsonl'):
    r=json.loads(line)
    if r['epoch']==2 and r['id'] not in seen:
        seen.add(r['id']); val_ids.append((r['id'], int(r['label'])))
byid={}
for entry in json.load(open(c.DATA['inputs']['validation']['path'])):
    for item in entry['selected']:
        row=dict(item['row']); row['label']=int(item['label']); byid[row['id']]=row
val_rows=[]
for id_, label in val_ids:
    item=byid[id_]
    assert int(item['label'])==label
    assert scans[item['scan_id']]['split']=='val'
    val_rows.append(item)
tp,fp,fn,tn=confusion(val_rows, False)
n=tp+fp+fn+tn
acc=(tp+tn)/n
prec=tp/(tp+fp) if tp+fp else 0.0
rec=tp/(tp+fn) if tp+fn else 0.0
f1=2*prec*rec/(prec+rec) if prec+rec else 0.0
summary={'stop_epoch':best['epoch'],'hold':hold,'hold_fp':best['fp'],'hold_fn':best['fn'],'val':{'tp':tp,'fp':fp,'fn':fn,'tn':tn,'acc':acc,'pos_f1':f1,'precision':prec,'recall':rec},'threshold':0.5}
json.dump(summary, open(OUT+'/summary.json','w'), indent=2)
print('DONE', 'epoch', best['epoch'], 'VAL_ACC', round(acc,4), 'VAL_F1', round(f1,4), 'FP', fp, 'FN', fn, 'TP', tp, 'TN', tn, flush=True)