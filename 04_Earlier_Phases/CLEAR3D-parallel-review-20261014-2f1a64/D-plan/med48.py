import csv, json, os, sys
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, "/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/work/src")
from lidc3d.geometry import crop
from lidc3d.data import load_array
from lidc3d.medicalnet_adapter import build_transfer
import lidc3d.diagnostic_training as c
PREP="/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared"
ANN="/experiment-storage/chaidir/fawad-dsai/data/reference/luna16-zenodo-3723295/resolved-reference-v1/annotations.csv"
CAND="/experiment-storage/chaidir/fawad-dsai/data/reference/luna16-zenodo-3723295/resolved-reference-v1/candidates_V2.csv"
ROOT="/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64"
W="/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-guarded-review-20261013-b17e4c/weights/resnet_18.pth"
OUT=ROOT+"/runs/med48-001"
os.makedirs(OUT, exist_ok=True)
VAL={"LIDC-IDRI-0020","LIDC-IDRI-0428","LIDC-IDRI-0634","LIDC-IDRI-0722","LIDC-IDRI-0865","LIDC-IDRI-0895"}
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
    if rec["patient_id"] in VAL or rec.get("split") in ("val", "test"): continue
    by[rec["series_uid"]]=rec
nod=defaultdict(list)
for r in csv.DictReader(open(ANN)):
    rec=by.get(r["seriesuid"])
    if rec is None: continue
    geo=json.load(open(os.path.join(PREP, rec["scan_id"], "geometry.json")))
    ijk=np.linalg.inv(np.array(geo["affine"], float)) @ np.array([float(r["coordX"]), float(r["coordY"]), float(r["coordZ"]), 1.0])
    nod[rec["series_uid"]].append({"patient":rec["patient_id"], "scan":rec["scan_id"], "zyx":[float(ijk[0]), float(ijk[1]), float(ijk[2])]})
passed=[]
for i,(uid, items) in enumerate(nod.items(), 1):
    d=os.path.join(PREP, items[0]["scan"])
    image=np.load(os.path.join(d, "image.npy"), mmap_mode="r")
    valid=np.load(os.path.join(d, "valid.npy"), mmap_mode="r")
    for it in items:
        z,y,x=[int(round(v)) for v in it["zyx"]]
        if not (0<=z<image.shape[0] and 0<=y<image.shape[1] and 0<=x<image.shape[2]): continue
        if valid[z,y,x]==0: continue
        arr=np.asarray(crop(np.asarray(image), it["zyx"], 48)[0])
        if not bright(arr): continue
        pad=np.asarray(crop(np.asarray(image), it["zyx"], 52)[0])
        if pad.shape!=(52,52,52): continue
        it["pad"]=pad.astype(np.float32); passed.append(it)
    if i%100==0: print("AUDITED", i, "PASS", len(passed), flush=True)
print("AUDIT_DONE", len(passed), flush=True)
by_pat=defaultdict(list)
for it in passed: by_pat[it["patient"]].append(it)
hold="LIDC-IDRI-0694"; patients=sorted(by_pat, key=lambda p: (len(by_pat[p]), p)); stop=[]; nstop=0
if hold in by_pat:
    stop.append(hold); nstop=len(by_pat[hold])
for p in patients:
    if p==hold: continue
    if nstop>=30: break
    stop.append(p); nstop+=len(by_pat[p])
stop=set(stop); grad=[p for p in by_pat if p not in stop]
grad_items=[it for p in grad for it in by_pat[p]]; stop_items=[it for p in stop for it in by_pat[p]]
print("GATE", len(grad_items), len(stop_items), flush=True)
if len(grad_items)<200 or len(stop_items)<30: raise SystemExit("gate")
scan_to_series={rec["scan_id"]: uid for uid, rec in by.items()}
series_grad={scan_to_series[it["scan"]] for it in grad_items}
series_stop={scan_to_series[it["scan"]] for it in stop_items}
rng=np.random.default_rng(17)
neg_xyz=[]; stop_xyz=[]
for r in csv.DictReader(open(CAND)):
    if r["class"]!="0": continue
    row=(r["seriesuid"], float(r["coordX"]), float(r["coordY"]), float(r["coordZ"]))
    if r["seriesuid"] in series_grad: neg_xyz.append(row)
    elif r["seriesuid"] in series_stop: stop_xyz.append(row)
neg_pick=rng.choice(len(neg_xyz), size=min(2400, len(neg_xyz)), replace=False)
stop_pick=rng.choice(len(stop_xyz), size=min(600, len(stop_xyz)), replace=False)
def crops_at(picks, rows):
    grouped=defaultdict(list)
    for j in picks:
        uid,x,y,z=rows[int(j)]; grouped[uid].append((x,y,z))
    out=[]
    for uid, xyzs in grouped.items():
        rec=by[uid]; image=np.load(os.path.join(PREP, rec["scan_id"], "image.npy"), mmap_mode="r")
        geo=json.load(open(os.path.join(PREP, rec["scan_id"], "geometry.json")))
        Ainv=np.linalg.inv(np.array(geo["affine"], float))
        for x,y,z in xyzs:
            ijk=Ainv @ np.array([x,y,z,1.0])
            arr=np.asarray(crop(np.asarray(image), [float(ijk[0]), float(ijk[1]), float(ijk[2])], 48)[0])
            if arr.shape==(48,48,48): out.append(arr.astype(np.float32))
    return np.stack(out)
neg=crops_at(neg_pick, neg_xyz); stop_neg=crops_at(stop_pick, stop_xyz)
pos=np.stack([it["pad"] for it in grad_items]); stop_pos=np.stack([np.asarray(it["pad"])[2:50,2:50,2:50] for it in stop_items])
print("BANKS", pos.shape, neg.shape, stop_pos.shape, stop_neg.shape, flush=True)
np.savez(OUT+"/banks.npz", pos=pos, neg=neg, stop_pos=stop_pos, stop_neg=stop_neg)
def zbatch(batch, noise, rng):
    m=batch.mean(axis=(1,2,3), keepdims=True); s=batch.std(axis=(1,2,3), keepdims=True)
    out=(batch-m)/(s+1e-6)
    if noise: out=out+rng.normal(0, 0.125, out.shape).astype(np.float32)
    return out
def probs(model, batch, device):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0, len(batch), 4):
            x=zbatch(batch[i:i+4].astype(np.float32), False, None)
            t=torch.from_numpy(np.ascontiguousarray(x[:,None])).to(device)
            out.append(torch.softmax(model(t).float(), 1)[:,1].cpu().numpy())
    model.train(); return np.concatenate(out)
def operating_point(sp, sn):
    pool=np.concatenate([sp, sn]); cuts=np.unique(pool)
    if len(cuts)>300: cuts=np.quantile(pool, np.linspace(0, 1, 300))
    best=None
    for cut in cuts:
        tp=int((sp>=cut).sum()); fn=int((sp<cut).sum()); fp=int((sn>=cut).sum())
        prec=tp/(tp+fp) if tp+fp else 0.0; rec=tp/len(sp)
        f1=2*prec*rec/(prec+rec) if prec+rec else 0.0
        ok=int(rec>=0.933 and prec>=0.9426); rank=(ok, f1, -fp, -fn)
        row={"cut":float(cut),"tp":tp,"fp":fp,"fn":fn,"precision":prec,"recall":rec,"f1":f1,"ok":ok,"rank":rank}
        if best is None or rank>best["rank"]: best=row
    return best
def crop48(pad, rng):
    off=rng.integers(-2, 3, size=3); c0,c1,c2=(26+int(v) for v in off)
    return pad[c0-24:c0+24, c1-24:c1+24, c2-24:c2+24]
free=os.popen("nvidia-smi --query-gpu=memory.free,uuid --format=csv,noheader,nounits").read(); okmem=False
for line in free.splitlines():
    p=[x.strip() for x in line.split(",")]
    if len(p)==2 and p[1]=="GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90" and int(p[0])>=10240: okmem=True
if not okmem:
    print("LOWMEM", free, flush=True); raise SystemExit(0)
cfg=json.load(open("/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/configuration-resnet18.json"))
device=torch.device("cuda"); torch.manual_seed(17)
model=build_transfer(cfg["model"], 18, W, 17).to(device)
head=[p for n,p in model.named_parameters() if n.startswith("head.")]
rest=[p for n,p in model.named_parameters() if not n.startswith("head.") and p.requires_grad]
opt=torch.optim.SGD([{"params":rest, "lr":1e-4}, {"params":head, "lr":1e-3}], momentum=0.9)
loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([0.25, 1.75], device=device))
best=None; stall=0; log=open(OUT+"/train-log.jsonl","w")
for epoch in range(1, 9):
    model.train(); total=0.0
    if epoch==1: hard=rng.choice(len(neg), size=min(800, len(neg)), replace=False)
    else:
        sample=rng.choice(len(neg), size=min(800, len(neg)), replace=False)
        pr=probs(model, neg[sample], device); hard=sample[np.argsort(pr)[-300:]]
    for step in range(80):
        pb=np.stack([crop48(pos[int(i)], rng) for i in rng.integers(0, len(pos), 2)])
        nb=neg[np.concatenate([hard[rng.integers(0, len(hard), 1)], rng.integers(0, len(neg), 1)])]
        batch=zbatch(np.concatenate([pb, nb]).astype(np.float32), True, rng)
        y=torch.tensor([1,1,0,0], device=device)
        opt.zero_grad(set_to_none=True)
        loss=loss_fn(model(torch.from_numpy(np.ascontiguousarray(batch[:,None])).to(device)), y)
        if not torch.isfinite(loss): raise SystemExit("nonfinite")
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); total+=float(loss.detach())
    op=operating_point(probs(model, stop_pos, device), probs(model, stop_neg, device))
    rec={k:op[k] for k in ("cut","tp","fp","fn","precision","recall","f1","ok")}; rec["epoch"]=epoch; rec["loss"]=total/80
    log.write(json.dumps(rec)+chr(10)); log.flush()
    print("EPOCH", epoch, "loss", round(rec["loss"],4), "fp", op["fp"], "fn", op["fn"], "f1", round(op["f1"],4), "ok", op["ok"], flush=True)
    if best is None or op["rank"]>best["rank"]:
        best=dict(op); best["epoch"]=epoch; stall=0
        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    else: stall+=1
    if op["ok"] and stall>=2: break
log.close(); torch.save({"epoch":best["epoch"],"cut":best["cut"],"state":best_state}, OUT+"/best-state.pt")
model.load_state_dict(best_state); cut=float(best["cut"])
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
    item=byid[id_]; path=scans[item["scan_id"]]["image_path"]
    if path not in cache: cache[path]=load_array(path)
    vp.append(np.asarray(crop(cache[path], [float(v) for v in item["zyx"]], 48)[0]).astype(np.float32)); vy.append(label)
pr=probs(model, np.stack(vp), device); vy=np.array(vy)
pred=pr>=cut
tp=int(((vy==1)&pred).sum()); fn=int(((vy==1)&~pred).sum()); fp=int(((vy==0)&pred).sum()); tn=int(((vy==0)&~pred).sum())
n=tp+fp+fn+tn; acc=(tp+tn)/n; prec=tp/(tp+fp) if tp+fp else 0.0; reca=tp/(tp+fn) if tp+fn else 0.0
f1=2*prec*reca/(prec+reca) if prec+reca else 0.0; spec=tn/(tn+fp) if tn+fp else 0.0
summary={"epoch":best["epoch"],"cut":cut,"stop_fp":best["fp"],"stop_fn":best["fn"],"val":{"tp":tp,"fp":fp,"fn":fn,"tn":tn,"acc":acc,"precision":prec,"recall":reca,"f1":f1,"specificity":spec}}
json.dump(summary, open(OUT+"/summary.json","w"), indent=2)
print("DONE", "epoch", best["epoch"], "VAL_ACC", round(acc,4), "SENS", round(reca,4), "PREC", round(prec,4), "SPEC", round(spec,4), "F1", round(f1,4), "TP", tp, "FP", fp, "FN", fn, "TN", tn, flush=True)
