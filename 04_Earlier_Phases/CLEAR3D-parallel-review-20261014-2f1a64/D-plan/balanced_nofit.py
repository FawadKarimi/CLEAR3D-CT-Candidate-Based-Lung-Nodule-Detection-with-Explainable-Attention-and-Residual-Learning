import json, os, sqlite3, sys
import numpy as np
import torch
sys.path.insert(0,'/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/work/src')
import lidc3d.diagnostic_training as c
from lidc3d.data import load_array
from lidc3d.geometry import crop
from lidc3d.medicalnet_finetuning import build
ROOT='/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64'
OUT=ROOT+'/runs/balanced-nofit-002'
os.makedirs(OUT, exist_ok=True)
cfg=json.load(open('/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/configuration-resnet18.json'))
parent=json.load(open(c.DATA['inputs']['parent_selection']['path']))
scans={e['scan']['scan_id']: e['scan'] for e in parent['entries']}
db=sqlite3.connect(c.DATA['inputs']['sidecar']['path'])
def payload(id_):
    row=db.execute('SELECT payload FROM labels WHERE id=?', (id_,)).fetchone()
    if row is None:
        raise SystemExit('missing label '+id_)
    return json.loads(row[0])
def patch_of(item):
    scan=scans[item['scan_id']]
    image,_=crop(load_array(scan['image_path']), item['zyx'], 48)
    valid,_=crop(load_array(scan['valid_path']), item['zyx'], 48, False)
    arr=np.where(valid, image, 0).astype(np.float32)
    return {'id': item['id'], 'label': int(item.get('label', item.get('_label'))), 'image': arr, 'patient': scan['patient_id'], 'split': scan['split']}
pos=[json.loads(r[0]) for r in db.execute('SELECT payload FROM labels WHERE label=1')]
for p in pos:
    p['_label']=1
    assert scans[p['scan_id']]['split']=='train'
patients=sorted({scans[p['scan_id']]['patient_id'] for p in pos})
hold_patient=patients[-1]
fit_pos=[p for p in pos if scans[p['scan_id']]['patient_id']!=hold_patient]
check_pos=[p for p in pos if scans[p['scan_id']]['patient_id']==hold_patient]
if len(fit_pos)<8 or len(check_pos)<1:
    ordered=sorted(pos, key=lambda p: p['id'])
    check_pos, fit_pos=ordered[:3], ordered[3:]
    hold_patient='id-holdout'
train_scans={s['scan_id'] for s in scans.values() if s['split']=='train'}
neg_rows=db.execute('SELECT id, scan FROM labels WHERE label=0').fetchall()
neg_rows=[(i,s) for i,s in neg_rows if s in train_scans]
rng=np.random.default_rng(17)
rng.shuffle(neg_rows)
fit_neg_ids=[]; check_neg_ids=[]
for i,s in neg_rows:
    patient=scans[s]['patient_id']
    if patient==hold_patient and len(check_neg_ids)<max(8, 4*len(check_pos)):
        check_neg_ids.append(i)
    elif patient!=hold_patient and len(fit_neg_ids)<max(32, 4*len(fit_pos)):
        fit_neg_ids.append(i)
    if len(fit_neg_ids)>=max(32, 4*len(fit_pos)) and len(check_neg_ids)>=max(8, 4*len(check_pos)):
        break
def materialize(items, label):
    out=[]
    for item in items:
        if isinstance(item, str):
            item=payload(item)
        item=dict(item); item['_label']=label
        row=patch_of(item); row['label']=label; out.append(row)
    return out
fit_pos_r=materialize(fit_pos, 1)
check_pos_r=materialize(check_pos, 1)
fit_neg_r=materialize(fit_neg_ids, 0)
check_neg_r=materialize(check_neg_ids, 0)
fit_rows=fit_pos_r+fit_neg_r
check_rows=check_pos_r+check_neg_r
if len(check_neg_ids)<max(8, 4*len(check_pos)) and hold_patient=='id-holdout':
    need=max(8, 4*len(check_pos))
    fit_scans={p['scan_id'] for p in fit_pos}
    used=set(check_neg_ids)|set(fit_neg_ids)
    for i,s in neg_rows:
        if s not in fit_scans and i not in used:
            check_neg_ids.append(i); used.add(i)
        if len(check_neg_ids)>=need:
            break
print('SPLIT_COUNTS', 'fit_pos', len(fit_pos), 'fit_neg', len(fit_neg_ids), 'check_pos', len(check_pos), 'check_neg', len(check_neg_ids), 'hold', hold_patient)
val_ids=[]
seen=set()
for line in open(ROOT+'/runs/resnet18-constlr-001/val-predictions.jsonl'):
    r=json.loads(line)
    if r['epoch']==2 and r['id'] not in seen:
        seen.add(r['id']); val_ids.append((r['id'], int(r['label'])))
byid={}
for entry in json.load(open(c.DATA['inputs']['validation']['path'])):
    for item in entry['selected']:
        row=dict(item['row'])
        row['_label']=int(item['label'])
        byid[row['id']]=row
val_rows=[]
for id_, label in val_ids:
    item=byid[id_]
    assert int(item['_label'])==label
    assert scans[item['scan_id']]['split']=='val'
    row=patch_of(item); row['label']=label; val_rows.append(row)
assert len(val_rows)==384
print('SPLIT', 'fit_pos', len(fit_pos_r), 'fit_neg', len(fit_neg_r), 'check_pos', len(check_pos_r), 'check_neg', len(check_neg_r), 'val', len(val_rows), 'hold', hold_patient)
device=torch.device('cuda')
torch.manual_seed(17)
model=build(cfg, 17).to(device)
head=[p for p in model.head.parameters()]
head_ids=set(map(id, head))
rest=[p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
opt=torch.optim.AdamW([{'params': head, 'lr': 1e-3, 'weight_decay': 1e-4}, {'params': rest, 'lr': 1e-4, 'weight_decay': 1e-4}])
loss_fn=torch.nn.CrossEntropyLoss()
def metrics(rows):
    model.eval()
    probs=[]; labels=[]
    with torch.no_grad():
        for i in range(0, len(rows), 4):
            batch=np.stack([r['image'] for r in rows[i:i+4]])[:, None]
            logits=model(torch.from_numpy(batch).to(device))
            prob=torch.softmax(logits.float(), dim=1)[:,1].detach().cpu().numpy()
            probs.extend(prob.tolist()); labels.extend(r['label'] for r in rows[i:i+4])
    acc=0; st={0:[0,0,0], 1:[0,0,0]}
    for p,y in zip(probs, labels):
        pred=1 if p>=0.5 else 0; acc += pred==y
        st[y][2]+=1; st[pred][1]+=1
        if pred==y: st[y][0]+=1
    f1s=[]
    for k in (0,1):
        tp,pn,tn=st[k]
        prec=tp/pn if pn else 0.0; rec=tp/tn if tn else 0.0
        f1s.append(2*prec*rec/(prec+rec) if prec+rec else 0.0)
    p1,n1,t1=st[1]
    return {'acc': acc/len(rows), 'macro': sum(f1s)/2, 'pos_f1': f1s[1], 'pos_precision': (p1/n1 if n1 else 0.0), 'pos_recall': (p1/t1 if t1 else 0.0), 'n': len(rows)}
log=open(OUT+'/train-log.jsonl','w')
stop='budget'; stall=0
fit_pos_idx=np.arange(len(fit_pos_r)); fit_neg_idx=np.arange(len(fit_neg_r))
for step in range(1, 241):
    model.train()
    pb=rng.choice(fit_pos_idx, size=2, replace=True)
    nb=rng.choice(fit_neg_idx, size=2, replace=True)
    chosen=[fit_pos_r[i] for i in pb]+[fit_neg_r[i] for i in nb]
    batch=np.stack([r['image'] for r in chosen])[:, None]
    y=torch.tensor([r['label'] for r in chosen], device=device)
    opt.zero_grad(set_to_none=True)
    logits=model(torch.from_numpy(batch).to(device))
    loss=loss_fn(logits, y)
    if not torch.isfinite(loss):
        stop='nonfinite'; break
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step%20==0:
        tr=metrics(fit_rows); ck=metrics(check_rows)
        rec={'step': step, 'loss': float(loss.detach().cpu()), 'train': tr, 'check': ck}
        log.write(json.dumps(rec)+chr(10)); log.flush()
        print('STEP', step, 'loss', round(rec['loss'],4), 'train_acc', round(tr['acc'],4), 'train_macro', round(tr['macro'],4), 'check_acc', round(ck['acc'],4), 'check_macro', round(ck['macro'],4), flush=True)
        if ck['acc']>=0.98 and ck['macro']>=0.98:
            stop='check_target'; break
        if tr['macro']>=0.95 and ck['macro']<=tr['macro']-0.20:
            stall+=1
        else:
            stall=0
        if stall>=2:
            stop='overfitting'; break
log.close()
val=metrics(val_rows)
summary={'stop': stop, 'hold_patient': hold_patient, 'fit_pos': len(fit_pos_r), 'check_pos': len(check_pos_r), 'val': val, 'threshold': 0.5}
json.dump(summary, open(OUT+'/summary.json','w'), indent=2)
print('DONE', stop, 'VAL_ACC', round(val['acc'],4), 'VAL_MACRO', round(val['macro'],4), 'VAL_POS_F1', round(val['pos_f1'],4))
torch.save({'head': model.head.state_dict()}, OUT+'/head-state.pt')
