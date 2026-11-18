import json, math, sys
from pathlib import Path
sys.path[:0]=['/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/work/src','/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/work/scripts']
import lidc3d.diagnostic_training as c
import lidc3d.scoped_epochs as engine
import lidc3d.diagnostics as diagnostics
c.RESEARCH_LIMITS={**c.RESEARCH_LIMITS,'total_seconds':1800,'run_seconds':1800}
METRIC=Path('/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64/runs/resnet18-constlr-001/val-predictions.jsonl')
def epoch_lr(base, epoch, cfg):
    return float(base)*10.0
engine.epoch_lr=epoch_lr
_validation=diagnostics.Diagnostics.validation
def validation(self,predictions,labels,weights,epoch):
    rows=[]
    for r in predictions:
        label=int(labels[r['id']])
        logits=[float(x) for x in r['logits']]
        m=max(logits)
        exps=[math.exp(x-m) for x in logits]
        z=sum(exps)
        rows.append({'epoch':int(epoch),'id':r['id'],'label':label,'prob':exps[1]/z})
    METRIC.parent.mkdir(parents=True, exist_ok=True)
    with METRIC.open('a') as handle:
        for row in rows:
            handle.write(json.dumps(row)+chr(10))
    return _validation(self,predictions,labels,weights,epoch)
diagnostics.Diagnostics.validation=validation
from diagnostic_training_worker import main
if __name__=='__main__':
    raise SystemExit(main())
