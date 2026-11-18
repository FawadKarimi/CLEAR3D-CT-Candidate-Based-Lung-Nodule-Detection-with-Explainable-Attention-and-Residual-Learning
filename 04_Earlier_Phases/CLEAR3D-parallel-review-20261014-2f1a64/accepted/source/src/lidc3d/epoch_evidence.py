"""Durable phase transitions for production epoch orchestration."""
import json,os,time
from pathlib import Path
class Observer:
    def __init__(self,out):self.out=Path(out);self.last=None;self.started=time.monotonic();self.validation_start=None
    def __call__(self,name,**fields):
        now=time.monotonic()
        record={'name':name,'monotonic':now,**fields}
        if name=='validation':self.validation_start=now
        if self.validation_start is not None:
            record['validation_started']=self.validation_start
            if name not in ('validation','metrics','final_metrics'):
                record['completed_validation_seconds']=now-self.validation_start;self.validation_start=None
        if self.last is not None:record['previous_seconds']=now-self.last['monotonic'];record['previous']=self.last
        with (self.out/'phases.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+chr(10));f.flush();os.fsync(f.fileno())
        for filename,value in [('phase-detail.json',record),('phase.json',{'name':'complete' if name=='complete' else 'setup' if name in ('setup','indexing','model_setup') else 'training'})]:
            tmp=self.out/(filename+'.tmp');tmp.write_text(json.dumps(value));os.replace(tmp,self.out/filename)
        self.last=record
