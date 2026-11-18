"""Isolated metadata-only label sidecar. No production approval or pixel access."""
import hashlib, heapq, json, math, sqlite3, time, os
from pathlib import Path
from collections.abc import Sequence
from .references import candidate_label

SCOPE='cap4000-fixed12-reference-selection-v1'
LIMITS={'seconds':1800,'rss_bytes':2*1024**3,'minimum_available_ram':8*1024**3,'minimum_disk':4*1024**3,'json_bytes':16*1024**2,'payload_bytes':16384,'rows':48000,'per_scan':4000}
HOLDS=['reference origin','excluded-reference outliers','geometry and source scope','protocol/evaluation disclosure','prior test access','independent reference review']

def require(ok,message):
    if not ok:raise ValueError(message)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def chash(x):return hashlib.sha256(canonical(x)).hexdigest()
def save(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def safe(path):
    p=Path(path).absolute()
    require(not any(x.is_symlink() for x in [p,*p.parents]),'Symlink boundary')
    return p

def hash_file(path,tick=lambda:None):
    p=safe(path);require(p.is_file(),'Missing regular file');before=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        while True:
            tick();b=f.read(1024**2)
            if not b:break
            h.update(b)
    after=p.stat();require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'Input changed')
    return h.hexdigest()

def bounded_json(path):
    p=safe(path);require(p.stat().st_size<=LIMITS['json_bytes'],'JSON admission');return json.loads(p.read_bytes())

def gate(disposition,binding_hash,bindings):
    require(disposition.get('status')=='ADOPTED' and disposition.get('scope')==SCOPE,'Explicit scoped adoption required')
    require(disposition.get('binding_sha256')==binding_hash and disposition.get('inputs')==bindings,'Disposition identity changed')
    require(disposition.get('limits')==LIMITS and disposition.get('reference_release')=='HOLD' and disposition.get('cap4000_policy')=='NOT_ADOPTED','Scope/limits changed')
    require(disposition.get('acknowledged_holds')==HOLDS and disposition.get('exposure_uncertainty') is True and disposition.get('sampled_monitoring') is True,'Missing limitations')
    require(all(isinstance(disposition.get(k),str) and disposition[k].strip() for k in ['supplied_by','instruction','date']),'Missing actual adoption')

def identities(entries,pool):
    require(len(entries)==len(pool)==18,'Fixed 18 identity admission')
    expected=[(p['scan_id'],p['patient_id'],p['role']) for p in pool]
    actual=[(e['scan']['scan_id'],e['scan']['patient_id'],e['scan']['split']) for e in entries]
    require(actual==expected and len({x[0] for x in actual})==18,'Fixed identities/order changed')
    patients={}
    for _,patient,role in actual:
        require(role in ('train','val'),'Development roles only');require(patients.setdefault(patient,role)==role,'Patient separation')
    require(sum(x[2]=='train' for x in actual)==12,'Fixed role counts')

def references(scan):
    positive=scan['positive'];ignored=scan.get('ignored',[])
    require(isinstance(positive,list) and isinstance(ignored,list),'Reference lists required')
    for region in positive+ignored:
        require(not region.get('mask_path'),'Mask-backed references require a different scope')
        require(len(region['center_lps'])==3 and all(math.isfinite(float(x)) for x in region['center_lps']),'Nonfinite reference')
        require(math.isfinite(float(region['radius_mm'])) and region['radius_mm']>=0,'Invalid radius')
    return positive,ignored

def readonly(path):
    db=sqlite3.connect(safe(path).as_uri()+'?mode=ro',uri=True)
    db.execute('PRAGMA query_only=ON');db.execute('PRAGMA cache_size=-4096');return db

def build(index,selection,pool,out,tick=lambda:None):
    """Caller verifies byte bindings and adoption first. Synthetic-test seam."""
    entries=selection['entries'];identities(entries,pool)
    val=[e for e in entries if e['scan']['split']=='val']
    require(len(val)==6 and all(len(e['selected'])==64 for e in val),'Unchanged validation count')
    out=safe(out);require(out.is_dir(),'Fresh output created by controller required')
    target=out/'labels.sqlite';require(not target.exists(),'Existing sidecar')
    source=readonly(index);db=sqlite3.connect(target);total=0;usable=0;receipts=[]
    try:
        db.execute('PRAGMA cache_size=-4096');db.execute('PRAGMA temp_store=FILE')
        db.execute('CREATE TABLE labels(id TEXT PRIMARY KEY, scan TEXT, source_ordinal INTEGER UNIQUE, label INTEGER CHECK(label IN (-1,0,1)), usable_index INTEGER UNIQUE, payload TEXT, payload_sha256 TEXT)')
        for entry in entries:
            s=entry['scan']
            if s['split']!='train':continue
            pos,ign=references(s);refhash=chash({'positive':pos,'ignored':ign});counts={-1:0,0:0,1:0};n=0;last=-1;stream=hashlib.sha256()
            for cid,sid,role,ordinal,payload in source.execute('SELECT id,scan,role,ordinal,payload FROM candidates WHERE scan=? ORDER BY ordinal',(s['scan_id'],)):
                tick();raw=payload.encode();require(n<LIMITS['per_scan'] and total<LIMITS['rows'] and len(raw)<=LIMITS['payload_bytes'],'Row/byte admission')
                require(role=='train' and sid==s['scan_id'] and ordinal>last,'Role/order mismatch');last=ordinal
                row=json.loads(payload);require(row['id']==cid and row['scan_id']==sid and 'label' not in row,'Candidate identity/label drift')
                require(len(row['center_lps'])==3 and all(math.isfinite(float(x)) for x in row['center_lps']),'Invalid center')
                label=candidate_label(row['center_lps'],pos,ign);ph=hashlib.sha256(raw).hexdigest();stream.update(raw)
                db.execute('INSERT INTO labels VALUES(?,?,?,?,?,?,?)',(cid,sid,ordinal,label,usable if label!=-1 else None,payload,ph))
                if label!=-1:usable+=1
                counts[label]+=1;n+=1;total+=1
            require(n==4000,'Fixed cap4000 count mismatch')
            receipts.append({'scan_id':s['scan_id'],'rows':n,'labels':{str(k):v for k,v in counts.items()},'reference_object_sha256':refhash,'reference_rows':{'positive':len(pos),'ignored':len(ign)},'candidate_payload_stream_sha256':stream.hexdigest()})
            db.commit();save(out/('scan-'+s['scan_id']+'.json'),receipts[-1])
        require(total==48000,'Incomplete training population')
        db.execute('CREATE INDEX class_order ON labels(label,usable_index)');db.commit()
        save(out/'validation-unchanged.json',val)
        result={'status':'COMPLETE_EXPLORATORY','rows':total,'usable':usable,'per_scan':receipts,'validation_rows':384,'validation_objects_sha256':chash(val),'reference_release':'HOLD','cap4000_policy':'NOT_ADOPTED','training_ready':False}
        save(out/'selection.json',result);return result
    except BaseException as ex:
        db.rollback();save(out/'selection-failure.json',{'status':'INCOMPLETE','rows_processed':total,'receipts':receipts,'error':repr(ex),'identities_accepted_only_after_input_verification':True});raise
    finally:source.close();db.close()

class LabelRows(Sequence):
    validated_index=True;role='train'
    def __init__(self,path,expected_hash,tick=lambda:None):
        require(hash_file(path,tick)==expected_hash,'Sidecar hash mismatch');self.db=readonly(path)
        n,lo,hi=self.db.execute('SELECT count(*),min(usable_index),max(usable_index) FROM labels WHERE usable_index IS NOT NULL').fetchone()
        require(0<n<=48000 and lo==0 and hi==n-1,'Usable index admission');self.n=n
    def __len__(self):return self.n
    def __getitem__(self,i):
        if not isinstance(i,int) or not 0<=i<self.n:raise IndexError('Index outside admitted population')
        payload,label,ph=self.db.execute('SELECT payload,label,payload_sha256 FROM labels WHERE usable_index=?',(i,)).fetchone()
        require(hashlib.sha256(payload.encode()).hexdigest()==ph and label in (0,1),'Sidecar payload mismatch')
        return {**json.loads(payload),'label':label}
    def close(self):self.db.close()

class DiskTrainingSample(Sequence):
    """Exact positive-cycle-negative-hash-v1; at most 240+720 indices retained."""
    validated_index=True
    def __init__(self,rows,spec,seed,tick=lambda:None):
        require(rows.role=='train' and seed==17,'Training-only sampling/seed')
        require(spec=={'policy':'positive-cycle-negative-hash-v1','positive_draws':240,'negative_draws':720,'max_source_rows':283200,'max_epoch_rows':65536},'Sampling policy changed')
        require(len(rows)<=48000,'Source admission');self.rows=rows;self.spec=dict(spec);self.seed=seed;self.tick=tick;self.set_epoch(0)
    def set_epoch(self,epoch):
        require(isinstance(epoch,int) and 0<=epoch<3,'Three-epoch scope');heaps={0:[],1:[]};counts={0:0,1:0}
        for i,label in self.rows.db.execute('SELECT usable_index,label FROM labels WHERE usable_index IS NOT NULL ORDER BY usable_index'):
            self.tick();require(label in (0,1),'Ignored sampling refusal');counts[label]+=1
            rank=int.from_bytes(hashlib.sha256(f'{self.seed}:{epoch}:{i}'.encode()).digest(),'big');item=(-rank,-i);h=heaps[label];cap=240 if label else 720
            if len(h)<cap:heapq.heappush(h,item)
            elif item>h[0]:heapq.heapreplace(h,item)
        require(counts[1]>0 and counts[0]>=720,'Insufficient class coverage; no substitution')
        p=[-i for _,i in sorted(heaps[1],reverse=True)];n=[-i for _,i in sorted(heaps[0],reverse=True)]
        self.indices=[p[i%len(p)] for i in range(240)]+n
        self.receipt={'epoch':epoch,'policy':self.spec,'seed':self.seed,'positive_draws':240,'negative_draws':720,'unique_positive_sources':min(counts[1],240),'source_index_sha256':chash(self.indices)}
        if hasattr(self.rows,'draw_plans'):
            plan=self.rows.draw_plans[epoch]
            require(plan=={'indices':self.indices,'receipt':self.receipt},'Accepted draw plan mismatch')
    def __len__(self):return 960
    def __getitem__(self,i):return self.rows[self.indices[i]]

def checkpoint_identity(parent,selection_hash,sidecar_hash,validation_hash,manifest_hash,index_hash,source_hash):
    return {**parent,'candidate_label_adapter':SCOPE,'label_selection_sha256':selection_hash,'label_sidecar_sha256':sidecar_hash,'unchanged_validation_sha256':validation_hash,'candidate_manifest_sha256':manifest_hash,'metadata_index_sha256':index_hash,'source_binding_sha256':source_hash}
