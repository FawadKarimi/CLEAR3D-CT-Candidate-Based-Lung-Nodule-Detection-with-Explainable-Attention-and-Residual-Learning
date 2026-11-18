import json, os, sys
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, "/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/work/src")
from lidc3d.geometry import crop
from lidc3d.data import load_array
import lidc3d.diagnostic_training as c
ROOT="/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64"
OUT=ROOT+"/runs/multibranch-001"
os.makedirs(OUT, exist_ok=True)
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
def operating_point(sp, sn):
    pool=np.concatenate([sp, sn])
    cuts=np.unique(pool)
    if len(cuts)>400: cuts=np.quantile(pool, np.linspace(0, 1, 400))
    best=None
    for cut in cuts:
        tp=int((sp>=cut).sum()); fn=int((sp<cut).sum()); fp=int((sn>=cut).sum()); tn=int((sn<cut).sum())
        prec=tp/(tp+fp) if tp+fp else 0.0; rec=tp/(tp+fn) if tp+fn else 0.0
        f1=2*prec*rec/(prec+rec) if prec+rec else 0.0
        ok=int(rec>=0.933 and prec>=0.9426)
        rank=(ok, f1, -fp, -fn)
        row={"cut":float(cut),"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":prec,"recall":rec,"f1":f1,"ok":ok,"rank":rank}
        if best is None or rank>best["rank"]: best=row
    return best
def crop_pos(pad, rng):
    off=rng.integers(-2, 3, size=3)
    c0,c1,c2=(12+int(v) for v in off)
    return pad[c0-9:c0+9, c1-9:c1+9, c2-9:c2+9]
free=os.popen("nvidia-smi --query-gpu=memory.free,uuid --format=csv,noheader,nounits").read()
okmem=False
for line in free.splitlines():
    p=[x.strip() for x in line.split(",")]
    if len(p)==2 and p[1]=="GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90" and int(p[0])>=10240: okmem=True
if not okmem:
    print("LOWMEM", free, flush=True); raise SystemExit(0)
device=torch.device("cuda"); torch.manual_seed(17); rng=np.random.default_rng(17)
model=Net().to(device)
opt=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([0.25, 1.75], device=device))
banks=np.load(ROOT+"/runs/luna-expand-001/banks.npz")
pos=banks["pos"]; neg=banks["neg"]; stop_pos=banks["stop_pos"]; stop_neg=banks["stop_neg"]
print("DATA", tuple(pos.shape), tuple(neg.shape), tuple(stop_pos.shape), tuple(stop_neg.shape), flush=True)
best=None; stall=0; log=open(OUT+"/train-log.jsonl", "w")
for epoch in range(1, 13):
    model.train()
    if epoch==1:
        hard=rng.choice(len(neg), size=min(4000, len(neg)), replace=False)
    else:
        sample=rng.choice(len(neg), size=min(4000, len(neg)), replace=False)
        pr=probs(model, neg[sample], device)
        hard=sample[np.argsort(pr)[-1500:]]
    total=0.0
    for step in range(300):
        pb=np.stack([crop_pos(pos[int(i)], rng) for i in rng.integers(0, len(pos), 4)])
        hn=hard[rng.integers(0, len(hard), 14)]; rn=rng.integers(0, len(neg), 14)
        nb=np.concatenate([neg[hn], neg[rn]], 0)
        batch=zbatch(np.concatenate([pb, nb]).astype(np.float32), True, rng)
        y=torch.tensor([1]*4+[0]*28, device=device)
        opt.zero_grad(set_to_none=True)
        loss=loss_fn(model(torch.from_numpy(np.ascontiguousarray(batch[:,None])).to(device)), y)
        if not torch.isfinite(loss): raise SystemExit("nonfinite")
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); total+=float(loss.detach())
    sp=probs(model, stop_pos, device); sn=probs(model, stop_neg, device); op=operating_point(sp, sn)
    rec={k:op[k] for k in ("cut","tp","fp","fn","tn","precision","recall","f1","ok")}; rec["epoch"]=epoch; rec["loss"]=total/300
    log.write(json.dumps(rec)+chr(10)); log.flush()
    print("EPOCH", epoch, "loss", round(rec["loss"],4), "cut", round(op["cut"],4), "fp", op["fp"], "fn", op["fn"], "f1", round(op["f1"],4), "ok", op["ok"], flush=True)
    if best is None or op["rank"]>best["rank"]:
        best=dict(op); best["epoch"]=epoch; stall=0
        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    else:
        stall+=1
    if op["ok"] and stall>=3: break
log.close()
torch.save({"epoch":best["epoch"],"cut":best["cut"],"state":best_state}, OUT+"/best-state.pt")
print("BEST", best["epoch"], "cut", best["cut"], "stop_fp", best["fp"], "stop_fn", best["fn"], "f1", best["f1"], flush=True)
model.load_state_dict(best_state)
cut=float(best["cut"])
parent=json.load(open(c.DATA["inputs"]["parent_selection"]["path"]))
scans={e["scan"]["scan_id"]: e["scan"] for e in parent["entries"]}
val_ids=[]; seen=set()
for line in open(ROOT+"/runs/resnet18-constlr-001/val-predictions.jsonl"):
    r=json.loads(line)
    if r["epoch"]==2 and r["id"] not in seen:
        seen.add(r["id"]); val_ids.append((r["id"], int(r["label"])))
byid={}
for entry in json.load(open(c.DATA["inputs"]["validation"]["path"])):
    for item in entry["selected"]:
        row=dict(item["row"]); row["label"]=int(item["label"]); byid[row["id"]]=row
cache={}; vp=[]; vy=[]
for id_, label in val_ids:
    item=byid[id_]
    assert int(item["label"])==label and scans[item["scan_id"]]["split"]=="val"
    path=scans[item["scan_id"]]["image_path"]
    if path not in cache: cache[path]=load_array(path)
    arr=np.asarray(crop(cache[path], [float(v) for v in item["zyx"]], 18)[0]).astype(np.float32)
    vp.append(arr); vy.append(label)
vp=np.stack(vp); vy=np.array(vy); pr=probs(model, vp, device)
def confuse(cutv):
    pred=pr>=cutv
    tp=int(((vy==1)&pred).sum()); fn=int(((vy==1)&~pred).sum()); fp=int(((vy==0)&pred).sum()); tn=int(((vy==0)&~pred).sum())
    n=tp+fp+fn+tn; acc=(tp+tn)/n
    prec=tp/(tp+fp) if tp+fp else 0.0; rec=tp/(tp+fn) if tp+fn else 0.0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0.0
    spec=tn/(tn+fp) if tn+fp else 0.0
    np_=tn/(tn+fn) if tn+fn else 0.0; nr=tn/(tn+fp) if tn+fp else 0.0
    neg_f1=2*np_*nr/(np_+nr) if np_+nr else 0.0
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,"acc":acc,"precision":prec,"recall":rec,"f1":f1,"specificity":spec,"macro_f1":(f1+neg_f1)/2}
val=confuse(cut); val05=confuse(0.5)
summary={"epoch":best["epoch"],"cut":cut,"stop":{k:best[k] for k in ("tp","fp","fn","tn","precision","recall","f1","ok")},"val":val,"val_at_0.5":val05}
json.dump(summary, open(OUT+"/summary.json","w"), indent=2)
print("DONE", "epoch", best["epoch"], "VAL_ACC", round(val["acc"],4), "SENS", round(val["recall"],4), "PREC", round(val["precision"],4), "SPEC", round(val["specificity"],4), "F1", round(val["f1"],4), "MACRO", round(val["macro_f1"],4), "TP", val["tp"], "FP", val["fp"], "FN", val["fn"], "TN", val["tn"], flush=True)
