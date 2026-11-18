"""Bounded disk index for immutable capped candidates; no proposal-pool materialization."""
import hashlib,json,math,sqlite3
from pathlib import Path
from collections.abc import Sequence
from .references import candidate_label
from .provenance import sha256,canonical_hash,write_json
MAX_LINE=16384
LIMITS={'train':708,'val':89}

def lines(path,limit=318800):
    with Path(path).open('rb') as f:
        count=0
        while True:
            raw=f.readline(MAX_LINE+1)
            if not raw:break
            if len(raw)>MAX_LINE or not raw.strip() or count>=limit:raise ValueError('Candidate admission limit')
            yield count,json.loads(raw)
            count+=1

class Rows(Sequence):
    validated_index=True
    def __init__(self,connection,role,scan=None):
        self.db,self.role,self.scan=connection,role,scan
        self.where='role=?'+(' AND scan=?' if scan is not None else '')
        self.args=(role,scan) if scan is not None else (role,)
        self.count=self.db.execute('SELECT count(*) FROM rows WHERE '+self.where,self.args).fetchone()[0]
    def __len__(self):return self.count
    def __getitem__(self,i):
        if not isinstance(i,int):raise TypeError('Integer index required')
        if i<0:i+=len(self)
        if i<0 or i>=len(self):raise IndexError(i)
        if self.scan is None:
            raw=self.db.execute('SELECT payload FROM rows WHERE role=? AND role_index=?',(self.role,i)).fetchone()
        else:
            raw=self.db.execute('SELECT payload FROM rows WHERE role=? AND scan=? ORDER BY role_index LIMIT 1 OFFSET ?',(*self.args,i)).fetchone()
        if raw is None:raise IndexError(i)
        return json.loads(raw[0])
    def __iter__(self):
        for (raw,) in self.db.execute('SELECT payload FROM rows WHERE '+self.where+' ORDER BY role_index',self.args):yield json.loads(raw)
    def for_scan(self,scan):return Rows(self.db,self.role,scan)


class TrackedLoader:
    """Hash ordered row indices and labels without retaining the epoch in RAM."""
    def __init__(self,loader,path,expected):self.loader,self.path,self.expected=loader,Path(path),expected
    def __len__(self):return len(self.loader)
    def __iter__(self):
        digest=hashlib.sha256();count=0;classes={'0':0,'1':0};seen=bytearray(self.expected)
        try:
            for batch in self.loader:
                for i,label in zip(batch['index'].tolist(),batch['label'].tolist()):
                    if not 0<=i<self.expected or seen[i]:raise ValueError('Repeated/outside epoch row')
                    if label not in (0,1):raise ValueError('Ignored/invalid epoch label')
                    seen[i]=1;count+=1;classes[str(label)]+=1;digest.update(f'{i}:{label};'.encode())
                yield batch
            if count!=self.expected:raise ValueError('Incomplete epoch consumption')
            write_json(self.path,{'status':'COMPLETE','rows':count,'classes':classes,'ordered_index_label_sha256':digest.hexdigest()})
        except BaseException:
            write_json(self.path.with_name(self.path.stem+'-failure.json'),{'status':'INCOMPLETE','rows_yielded':count,'classes':classes,'ordered_index_label_sha256':digest.hexdigest()});raise

class CandidateIndex:
    PER_SCAN_CAP = 400
    TOTAL_CAP = 318800
    def __init__(self,path,scans,output,tick=lambda **kw:None):
        self.db=None
        ids={s['scan_id']:s for s in scans}
        if len(ids)!=len(scans) or not ids:raise ValueError('Invalid cohort')
        if any(s['split'] not in LIMITS for s in scans):raise ValueError('Development only')
        if any(sum(s['split']==role for s in scans)>cap for role,cap in LIMITS.items()):raise ValueError('Scan admission limit')
        output=Path(output);self.path=output/'candidate-index.sqlite'
        if self.path.exists():raise ValueError('Index destination exists')
        tick(force=True)
        if Path(path).stat().st_size>self.TOTAL_CAP*MAX_LINE:raise ValueError('Candidate byte admission')
        source=sha256(path)
        db=sqlite3.connect(self.path);self.db=db
        db.execute('PRAGMA cache_size=-4096');db.execute('PRAGMA temp_store=FILE')
        db.execute('CREATE TABLE rows(role TEXT,role_index INTEGER,scan TEXT,id TEXT UNIQUE,payload TEXT,PRIMARY KEY(role,role_index))')
        db.execute('CREATE INDEX scan_rows ON rows(role,scan,role_index)')
        counts={s:0 for s in ids};roles={'train':0,'val':0};classes={'0':0,'1':0,'-1':0}
        try:
            for ordinal,row in lines(path, self.TOTAL_CAP):
                tick()
                sid=row.get('scan_id')
                if sid not in ids:raise ValueError('Candidate outside cohort')
                counts[sid]+=1
                if counts[sid]>self.PER_SCAN_CAP:raise ValueError('Per-scan candidate cap exceeded')
                if not isinstance(row.get('id'),str) or not row['id']:raise ValueError('Missing candidate identity')
                for key in ('zyx','center_lps'):
                    if not isinstance(row.get(key),list) or len(row[key])!=3 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in row[key]):raise ValueError('Invalid candidate coordinates')
                s=ids[sid];role=s['split']
                label=candidate_label(row['center_lps'],s['positive'],s.get('ignored',[])) if role=='train' else None
                if role=='train':classes[str(label)]+=1
                # IDs of ignored rows must still be unique, so retain with a separate role.
                store_role='ignored' if label==-1 else role
                ix=roles.get(store_role,0);roles[store_role]=ix+1
                if label is not None:row={**row,'label':label}
                db.execute('INSERT INTO rows VALUES(?,?,?,?,?)',(store_role,ix,sid,row['id'],json.dumps(row,sort_keys=True,allow_nan=False)))
            tick(force=True)
            if source!=sha256(path):raise ValueError('Candidate file changed during indexing')
            db.commit();db.execute('PRAGMA query_only=ON')
            self.receipt={'schema':'capped-index-v1','source_sha256':source,'cohort_sha256':canonical_hash(scans),'per_scan':counts,'roles':roles,'training_labels':classes,'limits':{'train_scans':708,'val_scans':89,'per_scan':self.PER_SCAN_CAP,'total':self.TOTAL_CAP,'line_bytes':MAX_LINE}}
            write_json(output/'candidate-index.json',self.receipt)
            self.train=Rows(db,'train');self.val=Rows(db,'val')
        except BaseException:
            db.rollback();db.close();self.db=None;raise
    def close(self):
        if self.db is not None:self.db.close();self.db=None
