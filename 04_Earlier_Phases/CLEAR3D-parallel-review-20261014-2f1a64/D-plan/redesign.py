import csv, json, os, sys
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, "/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/work/src")
from lidc3d.geometry import crop
PREP="/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared"
ANN="/experiment-storage/chaidir/fawad-dsai/data/reference/luna16-zenodo-3723295/resolved-reference-v1/annotations.csv"
CAND="/experiment-storage/chaidir/fawad-dsai/data/reference/luna16-zenodo-3723295/resolved-reference-v1/candidates_V2.csv"
ROOT="/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64"
OUT=ROOT+"/runs/redesign-001"
os.makedirs(OUT, exist_ok=True)
OLD_VAL={"LIDC-IDRI-0020","LIDC-IDRI-0428","LIDC-IDRI-0634","LIDC-IDRI-0722","LIDC-IDRI-0865","LIDC-IDRI-0895"}
class SE(nn.Module):
    def __init__(self, ch):
        super().__init__()
        h=max(ch//4, 4)
        self.fc=nn.Sequential(nn.Linear(ch, h), nn.ReLU(inplace=True), nn.Linear(h, ch), nn.Sigmoid())
    def forward(self, x):
        w=self.fc(x.mean(dim=(2,3,4)))
        return x*w[:,:,None,None,None]
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ks=nn.ModuleList(); self.fuse=nn.ModuleList(); self.se=nn.ModuleList(); cin=1
        for cout in (32, 64, 128, 128):
            self.ks.append(nn.ModuleList([nn.Conv3d(cin, cout, 3, padding=1), nn.Conv3d(cin, cout, 5, padding=2)]))
            self.fuse.append(nn.Conv3d(cout*2, cout, 1)); self.se.append(SE(cout)); cin=cout
        self.pool=nn.MaxPool3d(2, ceil_mode=True); self.act=nn.ReLU(inplace=True); self.fc=nn.Linear(128*8, 2)
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):
        for k, f, se in zip(self.ks, self.fuse, self.se):
            h=torch.cat([k[0](x), k[1](x)], 1)
            x=self.pool(se(self.act(f(h))))
        return self.fc(x.flatten(1))
def bright(arr):
    if arr.shape!=(48,48,48): return False
    inner=arr[16:32,16:32,16:32]
    shell=np.concatenate([arr[:16].ravel(), arr[32:].ravel(), arr[16:32,:16].ravel(), arr[16:32,32:].ravel(), arr[16:32,16:32,:16].ravel(), arr[16:32,16:32,32:].ravel()])
    thr=float(shell.mean()+shell.std()); m=inner>=thr
    if not m[8,8,8]: return False
    seen=np.zeros(m.shape, bool); stack=[(8,8,8)]; seen[8,8,8]=True; n=0
    while stack:
        z,y,x=stack.pop(); n+=1
        for dz,dy,dx in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            a,b,c=z+dz,y+dy,x+dx
            if 0<=a<16 and 0<=b<16 and 0<=c<16 and m[a,b,c] and not seen[a,b,c]:
                seen[a,b,c]=True; stack.append((a,b,c))
    return n>=10
by={}
for name in os.listdir(PREP):
    d=os.path.join(PREP, name)
    if not os.path.isdir(d): continue
    rec=json.load(open(os.path.join(d, "record.json")))
    if rec.get("split") in ("val", "test"): continue
    by[rec["series_uid"]]=rec
nod=defaultdict(list)
for r in csv.DictReader(open(ANN)):
    rec=by.get(r["seriesuid"])
    if rec is None: continue
    geo=json.load(open(os.path.join(PREP, rec["scan_id"], "geometry.json")))
    ijk=np.linalg.inv(np.array(geo["affine"], float)) @ np.array([float(r["coordX"]), float(r["coordY"]), float(r["coordZ"]), 1.0])
    nod[rec["series_uid"]].append({"patient":rec["patient_id"], "scan":rec["scan_id"], "series":rec["series_uid"], "zyx":[float(ijk[0]), float(ijk[1]), float(ijk[2])]})
passed=[]
for i,(uid, items) in enumerate(nod.items(), 1):
    image=np.load(os.path.join(PREP, items[0]["scan"], "image.npy"), mmap_mode="r")
    valid=np.load(os.path.join(PREP, items[0]["scan"], "valid.npy"), mmap_mode="r")
    for it in items:
        z,y,x=[int(round(v)) for v in it["zyx"]]
        if not (0<=z<image.shape[0] and 0<=y<image.shape[1] and 0<=x<image.shape[2]): continue
        if valid[z,y,x]==0: continue
        arr=np.asarray(crop(np.asarray(image), it["zyx"], 48)[0])
        if not bright(arr): continue
        pad=np.asarray(crop(np.asarray(image), it["zyx"], 24)[0])
        if pad.shape!=(24,24,24): continue
        it["pad"]=pad.astype(np.float32); passed.append(it)
    if i%150==0: print("AUDITED", i, "PASS", len(passed), flush=True)
print("AUDIT", len(passed), flush=True)
by_pat=defaultdict(list)
for it in passed: by_pat[it["patient"]].append(it)
rng=np.random.default_rng(17)
forced=[p for p in sorted(by_pat) if p in OLD_VAL]
rest=[p for p in sorted(by_pat) if p not in OLD_VAL]
rng.shuffle(rest)
n_val=max(1, int(round(0.15*len(by_pat))))
n_test=max(len(forced), int(round(0.15*len(by_pat))))
val_p=set(rest[:n_val])
test_p=set(rest[n_val:n_val+(n_test-len(forced))])|set(forced)
train_p=set(rest[n_val+(n_test-len(forced)):])
assert not (train_p & val_p or train_p & test_p or val_p & test_p)
groups={"train":train_p, "val":val_p, "test":test_p}
items={k:[it for p in v for it in by_pat[p]] for k,v in groups.items()}
print("SPLIT", {k:(len(v), len(items[k])) for k,v in groups.items()}, flush=True)
series_of={k:{it["series"] for it in items[k]} for k in items}
buckets={k:[] for k in items}
for r in csv.DictReader(open(CAND)):
    if r["class"]!="0": continue
    for k, ss in series_of.items():
        if r["seriesuid"] in ss:
            buckets[k].append((r["seriesuid"], float(r["coordX"]), float(r["coordY"]), float(r["coordZ"])))
            break
def take_negs(key, need):
    rows=buckets[key]
    pick=rng.choice(len(rows), size=min(need, len(rows)), replace=False)
    grouped=defaultdict(list)
    for j in pick:
        uid,x,y,z=rows[int(j)]; grouped[uid].append((x,y,z))
    out=[]
    for uid, xyzs in grouped.items():
        rec=by[uid]; image=np.load(os.path.join(PREP, rec["scan_id"], "image.npy"), mmap_mode="r")
        geo=json.load(open(os.path.join(PREP, rec["scan_id"], "geometry.json")))
        Ainv=np.linalg.inv(np.array(geo["affine"], float))
        for x,y,z in xyzs:
            ijk=Ainv @ np.array([x,y,z,1.0])
            arr=np.asarray(crop(np.asarray(image), [float(ijk[0]), float(ijk[1]), float(ijk[2])], 18)[0])
            if arr.shape==(18,18,18): out.append(arr.astype(np.float32))
    return np.stack(out)
pos={k:np.stack([it["pad"] for it in items[k]]) for k in items}
neg={k:take_negs(k, 7*len(items[k])) for k in items}
print("CROPS", {k:(pos[k].shape[0], neg[k].shape[0]) for k in pos}, flush=True)
def zbatch(batch, noise, rng):
    m=batch.mean(axis=(1,2,3), keepdims=True); s=batch.std(axis=(1,2,3), keepdims=True)
    out=(batch-m)/(s+1e-6)
    if noise: out=out+rng.normal(0, 0.125, out.shape).astype(np.float32)
    return out
def probs(model, batch, device):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0, len(batch), 64):
            x=zbatch(batch[i:i+64].astype(np.float32), False, None)
            t=torch.from_numpy(np.ascontiguousarray(x[:,None])).to(device)
            out.append(torch.softmax(model(t).float(), 1)[:,1].cpu().numpy())
    return np.concatenate(out)
def center18(pad):
    return pad[:, 3:21, 3:21, 3:21]
def metrics_at(sp, sn, cut):
    tp=int((sp>=cut).sum()); fn=int((sp<cut).sum()); fp=int((sn>=cut).sum()); tn=int((sn<cut).sum())
    prec=tp/(tp+fp) if tp+fp else 0.0; rec=tp/(tp+fn) if tp+fn else 0.0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0.0; spec=tn/(tn+fp) if tn+fp else 0.0
    acc=(tp+tn)/max(tp+tn+fp+fn, 1)
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,"acc":acc,"precision":prec,"recall":rec,"specificity":spec,"f1":f1}
def choose_cut(sp, sn):
    pool=np.concatenate([sp, sn]); cuts=np.unique(pool)
    if len(cuts)>400: cuts=np.quantile(pool, np.linspace(0, 1, 400))
    best=None
    for cut in cuts:
        m=metrics_at(sp, sn, float(cut))
        wang=int(m["recall"]>=0.932 and m["precision"]>=0.942 and m["f1"]>=0.937)
        zuo=int(m["acc"]>=0.9783 and m["recall"]>=0.8771 and m["precision"]>=0.9426 and m["specificity"]>=0.9925 and m["f1"]>=0.909)
        rank=(wang, zuo, m["f1"], -m["fp"], -m["fn"])
        if best is None or rank>best[0]: best=(rank, float(cut), m, wang, zuo)
    return best
def crop_pos(pad, rng):
    off=rng.integers(-2, 3, size=3); c0,c1,c2=(12+int(v) for v in off)
    return pad[c0-9:c0+9, c1-9:c1+9, c2-9:c2+9]
free=os.popen("nvidia-smi --query-gpu=memory.free,uuid --format=csv,noheader,nounits").read(); okmem=False
for line in free.splitlines():
    p=[x.strip() for x in line.split(",")]
    if len(p)==2 and p[1]=="GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90" and int(p[0])>=10240: okmem=True
if not okmem:
    print("LOWMEM", free, flush=True); raise SystemExit(0)
device=torch.device("cuda"); torch.manual_seed(17); model=Net().to(device)
opt=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([0.25, 1.75], device=device))
tr_pos=pos["train"]; tr_neg=neg["train"]; best=None; stall=0
log=open(OUT+"/train-log.jsonl", "w")
for epoch in range(1, 13):
    model.train()
    if epoch==1: hard=rng.choice(len(tr_neg), size=min(3000, len(tr_neg)), replace=False)
    else:
        sample=rng.choice(len(tr_neg), size=min(3000, len(tr_neg)), replace=False)
        pr=probs(model, tr_neg[sample], device); hard=sample[np.argsort(pr)[-1200:]]
    total=0.0
    for step in range(250):
        pb=np.stack([crop_pos(tr_pos[int(i)], rng) for i in rng.integers(0, len(tr_pos), 4)])
        ix=np.concatenate([hard[rng.integers(0, len(hard), 14)], rng.integers(0, len(tr_neg), 14)])
        batch=zbatch(np.concatenate([pb, tr_neg[ix]]).astype(np.float32), True, rng)
        y=torch.tensor([1]*4+[0]*28, device=device)
        opt.zero_grad(set_to_none=True)
        loss=loss_fn(model(torch.from_numpy(np.ascontiguousarray(batch[:,None])).to(device)), y)
        if not torch.isfinite(loss): raise SystemExit("nonfinite")
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); total+=float(loss.detach())
    rank, cut, m, wang, zuo=choose_cut(probs(model, center18(pos["val"]), device), probs(model, neg["val"], device))
    rec=dict(m); rec.update({"epoch":epoch,"cut":cut,"loss":total/250,"wang":wang,"zuo":zuo})
    log.write(json.dumps(rec)+chr(10)); log.flush()
    print("EPOCH", epoch, "loss", round(rec["loss"],4), "acc", round(m["acc"],4), "sens", round(m["recall"],4), "prec", round(m["precision"],4), "f1", round(m["f1"],4), "wang", wang, "zuo", zuo, flush=True)
    if best is None or rank>best[0]:
        best=(rank, cut, m, wang, zuo, epoch); stall=0
        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    else: stall+=1
    if wang and stall>=2: break
log.close()
model.load_state_dict(best_state); cut=best[1]
sp=probs(model, center18(pos["test"]), device); sn=probs(model, neg["test"], device)
test=metrics_at(sp, sn, cut)
zuo_beat=int(test["acc"]>0.9783 and test["recall"]>0.8771 and test["precision"]>0.9426 and test["specificity"]>0.9925 and test["f1"]>0.909)
wang_beat=int(test["recall"]>0.932 and test["precision"]>0.942 and test["f1"]>0.937)
summary={"epoch":best[5],"cut":cut,"val":best[2],"test":test,"test_pos":int(len(pos["test"])),"test_neg":int(len(neg["test"])),"zuo_beat":zuo_beat,"wang_beat":wang_beat}
json.dump(summary, open(OUT+"/summary.json","w"), indent=2)
torch.save({"epoch":best[5],"cut":cut,"state":best_state}, OUT+"/best-state.pt")
print("DONE", "epoch", best[5], "TEST_ACC", round(test["acc"],4), "SENS", round(test["recall"],4), "PREC", round(test["precision"],4), "SPEC", round(test["specificity"],4), "F1", round(test["f1"],4), "TP", test["tp"], "FP", test["fp"], "FN", test["fn"], "TN", test["tn"], "ZUO", zuo_beat, "WANG", wang_beat, flush=True)
