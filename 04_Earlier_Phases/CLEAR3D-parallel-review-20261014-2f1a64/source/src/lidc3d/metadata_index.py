"""Label-free disk index. Deliberately not a training CandidateIndex."""
import hashlib,json,sqlite3
from pathlib import Path
from .accounting import require,save,regular,signature

def digest(path,tick):
    h=hashlib.sha256()
    with regular(path).open('rb') as f:
        while True:
            tick();raw=f.read(1024**2)
            if not raw:break
            h.update(raw)
    return h.hexdigest()

def build_index(path,expected,identities,out,tick):
    path=regular(path);out=Path(out);before=signature(path)
    require(path.stat().st_size<=12*1024**3,'Index source byte admission')
    require(digest(path,tick)==expected,'Index source binding')
    ids={x['scan_id']:x for x in identities};require(len(ids)==len(identities)<=797,'Index identity inventory')
    require(all(x['split'] in ('train','val') for x in identities),'Development only')
    patients={}
    for x in identities:require(patients.setdefault(x['patient_id'],x['split'])==x['split'],'Patient separation')
    target=out/'metadata-index.sqlite';require(not target.exists(),'Existing index')
    db=sqlite3.connect(target);counts={sid:0 for sid in ids};roles={'train':0,'val':0};n=0;h=hashlib.sha256()
    try:
        db.execute('PRAGMA cache_size=-4096');db.execute('PRAGMA temp_store=FILE')
        db.execute('CREATE TABLE candidates(id TEXT PRIMARY KEY,scan TEXT,role TEXT,ordinal INTEGER,payload TEXT)')
        with path.open('rb') as f:
            while True:
                tick();raw=f.readline(16385)
                if not raw:break
                require(len(raw)<=16384 and raw.strip() and n<3188000,'Index line/count admission');h.update(raw);r=json.loads(raw);sid=r['scan_id']
                require(sid in ids and 'label' not in r,'Identity/label refusal');counts[sid]+=1;require(counts[sid]<=4000,'Per-scan cap')
                role=ids[sid]['split'];roles[role]+=1
                db.execute('INSERT INTO candidates VALUES(?,?,?,?,?)',(r['id'],sid,role,n,raw.decode()));n+=1
        require(h.hexdigest()==expected and signature(path)==before,'Index source changed')
        db.execute('CREATE INDEX scan_order ON candidates(scan,ordinal)');db.commit();tick(force=True)
        result={'status':'COMPLETE_UNFROZEN','source_sha256':expected,'rows':n,'roles':roles,'per_scan':counts,'labels_evaluated':False,'training_ready':False,'reference_release':'HOLD'}
        save(out/'metadata-index.json',result);return result
    except BaseException as ex:
        db.rollback();save(out/'index-failure.json',{'status':'INCOMPLETE','rows':n,'error':repr(ex),'reference_release':'HOLD'});raise
    finally:db.close()
