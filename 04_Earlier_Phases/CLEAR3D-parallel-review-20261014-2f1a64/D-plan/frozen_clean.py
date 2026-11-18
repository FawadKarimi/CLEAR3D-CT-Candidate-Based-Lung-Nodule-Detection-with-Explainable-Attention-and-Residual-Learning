import json, os, sqlite3, sys
import numpy as np
import torch
sys.path.insert(0,'/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/work/src')
import lidc3d.diagnostic_training as c
from lidc3d.data import load_array
from lidc3d.geometry import crop
from lidc3d.medicalnet_adapter import build_transfer
ROOT='/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64'
DEPTH=int(os.environ['PRETRAIN_DEPTH'])
if DEPTH not in (10,18):
    raise SystemExit('unreviewed depth')
OUT=ROOT+'/runs/frozen-clean10-001'
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
CLEAN={'scan_e2ef02265730531c459d:candidate:000268','scan_3ffbe7c95fa086f4a39e:candidate:000000','scan_c5973537991d725f03ab:candidate:000159','scan_aeeda85c3e8748887611:candidate:000227','scan_3386dd59b61402a85ab2:candidate:000000','scan_f86f7c1d51e9e7141921:candidate:000000','scan_f86f7c1d51e9e7141921:candidate:000019','scan_f86f7c1d51e9e7141921:candidate:000020','scan_62d1bce3a9980e7eb339:candidate:000131','scan_786987a82ce7e0172a06:candidate:001459'}
pos=[p for p in pos if p['id'] in CLEAN]
if len(pos)!=10: raise SystemExit('clean count '+str(len(pos)))
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
print('SPLIT_COUNTS', 'fit_pos', len(fit_pos), 'fit_neg', len(fit_neg_ids), 'check_pos', len(check_pos), 'check_neg', len(check_neg_ids), 'hold', hold_patient, 'depth', DEPTH)
print('SPLIT', 'fit_pos', len(fit_pos_r), 'fit_neg', len(fit_neg_r), 'check_pos', len(check_pos_r), 'check_neg', len(check_neg_r), 'hold', hold_patient, 'depth', DEPTH)
device=torch.device('cuda')
torch.manual_seed(17)
wpath='/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-guarded-review-20261013-b17e4c/weights/resnet_%d.pth'%DEPTH
model=build_transfer(cfg['model'], DEPTH, wpath, 17).to(device)
print('INIT', DEPTH)
for n,p in model.named_parameters():
    p.requires_grad=n.startswith('head.')
head=[p for p in model.parameters() if p.requires_grad]
if not head: raise SystemExit('no head params')
opt=torch.optim.AdamW(head, lr=1e-3, weight_decay=1e-4)
print('FROZEN_HEAD_ONLY', len(head))
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
    fp=n1-p1; fn=t1-p1
    return {'acc': acc/len(rows), 'macro': sum(f1s)/2, 'pos_f1': f1s[1], 'pos_precision': (p1/n1 if n1 else 0.0), 'pos_recall': (p1/t1 if t1 else 0.0), 'tp': p1, 'fp': fp, 'fn': fn, 'n': len(rows)}
log=open(OUT+'/train-log.jsonl','w')
stop='budget'; stall=0
best=None
best_state=None
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
        print('STEP', step, 'loss', round(rec['loss'],4), 'check_fn', ck['fn'], 'check_fp', ck['fp'], 'check_macro', round(ck['macro'],4), flush=True)
        overfit_now=tr['macro']>=0.95 and ck['macro']<=tr['macro']-0.20
        rank=(-ck['fn'], -ck['fp'], ck['macro'], ck['acc'])
        if (best is None or rank>best['rank']) and not overfit_now:
            best={'rank': rank, 'step': step, 'train': tr, 'check': ck}
            best_state={k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if ck['acc']>=0.98 and ck['macro']>=0.98:
            stop='check_target'; break
        if overfit_now:
            stall+=1
        else:
            stall=0
        if stall>=2:
            stop='overfitting'; break
log.close()
if best_state is None:
    raise SystemExit('no check snapshot')
torch.save({'depth': DEPTH, 'step': best['step'], 'state': best_state}, OUT+'/best-state.pt')
summary={'stop': stop, 'depth': DEPTH, 'hold_patient': hold_patient, 'fit_pos': len(fit_pos_r), 'check_pos': len(check_pos_r), 'best_step': best['step'], 'best_check': best['check'], 'best_train': best['train'], 'threshold': 0.5}
json.dump(summary, open(OUT+'/summary.json','w'), indent=2)
print('DONE', stop, 'BEST_STEP', best['step'], 'CHECK_ACC', round(best['check']['acc'],4), 'CHECK_MACRO', round(best['check']['macro'],4))
