"""Separate cap4000 review path. Does not authorize a policy or reference use."""
import hashlib,json,math,re
from pathlib import Path
from .indexed import CandidateIndex
from .accounting import require,regular,read_bound,raw_json,save
from .cap4000 import POLICY,canonical

class CandidateIndex4000(CandidateIndex):
    PER_SCAN_CAP=4000
    TOTAL_CAP=3188000


def flatten_manifest(root,expected_hash,destination,tick=lambda **kw:None):
    """Verify one bounded scan file at a time; never open arrays or references.

    Caller must separately authorize real construction and label-dependent indexing.
    Output is provisional until receipt exists; failures retain partial bytes.
    """
    root=Path(root);destination=Path(destination)
    require(not destination.exists(),'Destination exists')
    require(not any(p.is_symlink() for p in [destination.parent,*destination.parent.parents]),'Unsafe output')
    require(root.resolve()!=destination.parent.resolve() and root.resolve() not in destination.parents,'Output overlap')
    raw,_=read_bound(root/'candidate-manifest.json',expected_hash,16*1024**2);m=raw_json(raw)
    require(m['policy']==POLICY and m['status']=='COMPLETE_UNFROZEN' and m['reference_release']=='HOLD','Manifest policy/state')
    ids=m['identities'];require(len(ids)<=797 and len({s['scan_id'] for s in ids})==len(ids),'Identity admission')
    expected_files={'scans/'+s['scan_id']+'.json' for s in ids}
    require(set(m['files'])==expected_files,'File inventory')
    counts={'train':0,'val':0};total=0;size=0;digest=hashlib.sha256()
    try:
        with destination.open('xb') as out:
            for identity in ids:
                tick(force=True);sid=identity['scan_id'];role=identity['split']
                require(role in counts and sid and all(c.isalnum() or c in '_-' for c in sid),'Development identity')
                name='scans/'+sid+'.json';raw,_=read_bound(root/name,m['files'][name],128*1024**2);scan=raw_json(raw)
                require(scan['identity']==identity and scan['receipt']['status']=='VERIFIED','Identity/receipt mismatch');records=scan['records']
                require(len(records)<=4000,'Per-scan cap');last=None;ordinals=set()
                for rank,item in enumerate(records):
                    tick();row=item['row'];ordinal=item['source_ordinal']
                    require(type(ordinal) is int and ordinal>=0 and ordinal not in ordinals,'Duplicate/source ordinal')
                    ordinals.add(ordinal)
                    require(item['rank']==rank and row['scan_id']==sid and row['id']==f'{sid}:candidate:{ordinal:06d}','Rank/ordinal identity')
                    require(hashlib.sha256(canonical(row)).hexdigest()==item['payload_sha256'],'Payload hash')
                    key=(-row['log_response'],row['id'],ordinal)
                    require(type(row['log_response']) in (int,float) and math.isfinite(row['log_response']) and row['log_response']>=.005 and (last is None or last<key),'Ordering/threshold');last=key
                    require(isinstance(item['raw_line_sha256'],str) and re.fullmatch(r'[0-9a-f]{64}',item['raw_line_sha256']) is not None,'Raw hash format')
                    line=canonical(row)+b'\n';require(len(line)<=16384,'Candidate line admission')
                    total+=1;counts[role]+=1;require(total<=3188000,'Total cap')
                    size+=len(line);require(size<=12*1024**3,'Total byte ceiling');out.write(line);digest.update(line)
                del raw,scan,records
            require(counts==m['candidate_counts'],'Role count mismatch')
            out.flush()
            import os
            os.fsync(out.fileno())
        receipt={'status':'VERIFIED_UNFROZEN','manifest_sha256':expected_hash,'rows':total,'roles':counts,'sha256':digest.hexdigest(),'policy':POLICY,'reference_release':'HOLD','labels_evaluated':False}
        save(destination.with_suffix('.receipt.json'),receipt);return receipt
    except BaseException as ex:
        save(destination.with_suffix('.failure.json'),{'status':'UNVERIFIED','rows_written':total,'error':repr(ex),'reference_release':'HOLD'});raise
