"""Unfrozen canonical metadata manifests; no image or training operations."""
import hashlib
import heapq
import json
from pathlib import Path
from .accounting import (CAP as PARENT_CAP, THRESHOLDS, CONTRACT, MANIFEST_BYTES, PREPARATION_SHA256,
    Budget, LIMITS, require, raw_json, read_bound, read_manifest, signature, regular, append, save)
from .streaming_selection import Worst
CAP = 4000
POLICY = {'threshold': .005, 'cap': 4000, 'ranking': 'response-descending, ID-ascending, stable-source-ordinal',
          'purpose': 'training-reference-informed proposal; NOT_ADOPTED; not validation-selected', 'frozen': False}

class Cap4000Budget(Budget):
    def __init__(self,out):
        super().__init__(out,{**LIMITS,'evidence_bytes':12*1024**3})
    def check(self,force=False):
        old=self.samples
        super().check(force)
        if self.samples!=old:
            import shutil
            require(shutil.disk_usage(self.out).free>=8*1024**3,'DISK_HEADROOM')
            total=0
            for p in self.out.rglob('*'):
                require(not p.is_symlink(),'Output symlink')
                if p.is_file():total+=p.stat().st_size
            require(total<=12*1024**3,'OUTPUT_SIZE_LIMIT')

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def equivalence(receipts, identities):
    require(len(receipts) == len(identities), 'Receipt count')
    seen = set()
    for rec, identity in zip(receipts, identities):
        sid = rec['scan_id']; require(sid == identity['scan_id'] and sid not in seen, 'Receipt order/identity')
        seen.add(sid)
        require(rec['status'] == 'VERIFIED' and rec['split'] == identity['split'] and rec['patient_id'] == identity['patient_id'], 'Receipt ancestry')
        ts = rec['thresholds']; require(set(ts) == {str(t) for t in THRESHOLDS}, 'Threshold inventory')
        require(len({x['ranked_keys_sha256'] for x in ts.values()}) == 1, 'Ranked digests differ')
        require(len({x['candidate_count'] for x in ts.values()}) == 1 and 0 <= ts['0.005']['candidate_count'] <= PARENT_CAP, 'Cap/count mismatch')
    return {'verified_scans': len(seen), 'all_five_ranked_digests_equal': True, 'policy': POLICY}

def select_stream(path, sid, expected, count, accepted, tick=lambda **kw: None, aggregate=None):
    path = Path(path); regular(path); before = signature(path)
    require(path.stat().st_size == expected['bytes'], 'Stream byte admission')
    h = hashlib.sha256(); heap = []; n = size = qualifying = 0
    try:
        with path.open('rb') as f:
            while True:
                if n % 256 == 0: tick()
                raw = f.readline(16385)
                if not raw: break
                require(len(raw) <= 16384 and raw.strip() and n < count, 'Stream line/count admission')
                h.update(raw); size += len(raw)
                if aggregate is not None: aggregate.update(raw)
                row = raw_json(raw)
                require(row.get('scan_id') == sid and row.get('id') == f'{sid}:candidate:{n:06d}', 'Source ordinal identity')
                response = row.get('log_response')
                import math
                require(type(response) in (int, float) and math.isfinite(response), 'Response invalid')
                if response >= POLICY['threshold']:
                    qualifying += 1; key = (-response, row['id'], n)
                    item = Worst(key, {'source_ordinal': n, 'raw_line_sha256': hashlib.sha256(raw).hexdigest(),
                                      'payload_sha256': hashlib.sha256(canonical(row)).hexdigest(), 'row': row})
                    if len(heap) < CAP: heapq.heappush(heap, item)
                    elif key < heap[0].key: heapq.heapreplace(heap, item)
                n += 1
        tick(force=True)
        require(n == count and size == expected['bytes'] and h.hexdigest() == expected['sha256'] and signature(path) == before, 'Incomplete/changed stream')
        ordered = sorted(heap, key=lambda x: x.key); keys = hashlib.sha256()
        for item in ordered[:PARENT_CAP]: keys.update((json.dumps(item.key, separators=(',', ':'), allow_nan=False)+'\n').encode())
        actual = {'qualifying_before_cap': qualifying, 'candidate_count': min(len(ordered), PARENT_CAP), 'ranked_keys_sha256': keys.hexdigest()}
        require(actual == accepted['thresholds']['0.005'], 'Accepted canonical receipt mismatch')
        require(accepted['sha256'] == h.hexdigest() and accepted['proposal_count'] == n and accepted['bytes'] == size, 'Accepted stream receipt mismatch')
        selected = [dict(x.row, rank=i) for i,x in enumerate(ordered)]
        return selected, {'scan_id': sid, 'status': 'VERIFIED', 'proposal_count': n, 'bytes': size,
                         'sha256': h.hexdigest(), 'accepted_cap400_prefix': actual, 'selected_count': len(ordered), 'input_stat': before}
    except BaseException as ex:
        ex.candidate_partial = {'scan_id': sid, 'status': 'UNVERIFIED', 'rows_read': n, 'bytes_read': size, 'prefix_sha256': h.hexdigest()}
        raise

def build(root, out, receipts, accepted_identities, budget=None, contract=CONTRACT, manifest_bytes=MANIFEST_BYTES, preparation_hash=PREPARATION_SHA256):
    root, out = Path(root), Path(out); budget = budget or Cap4000Budget(out)
    require(root.absolute() not in out.absolute().parents and out.absolute() not in root.absolute().parents and root.absolute()!=out.absolute(), 'Output overlap')
    stats = {}; files = {}; completed = 0
    def metadata(name, expected, limit):
        raw, st = read_bound(root/name, expected, limit); stats[name] = st; return raw_json(raw)
    try:
        budget.check(force=True)
        prep = metadata('preparation.json', preparation_hash, 16*1024**2)
        require(prep['status']=='PROPOSALS_NOT_YET_FROZEN' and prep['config']['candidates']['threshold_search']==list(THRESHOLDS) and prep['config']['candidates']['cap']==PARENT_CAP and prep['minimum_pool_threshold']==.005, 'Policy/stage mismatch')
        require(all(prep['derivation']['counts'][k]==v for k,v in contract.items()), 'Derivative count mismatch')
        identities, manifest_receipt, st = read_manifest(root/'scans.jsonl', prep['files']['scans.jsonl'], manifest_bytes, contract, budget.check)
        stats['scans.jsonl']=st
        require(identities==accepted_identities, 'Accepted identities changed'); equivalence(receipts, identities)
        save(out/'identities.json', identities); (out/'scans').mkdir()
        aggregate=hashlib.sha256(); total=0; candidates={'train':0,'val':0}
        for identity, accepted in zip(identities, receipts):
            sid=identity['scan_id']; cp=metadata(sid+'/checkpoint.json', prep['artifacts'][sid+'/checkpoint.json'], 2*1024**2)
            expected=prep['artifacts'][sid+'/proposals.jsonl']
            require(cp['scan_id']==sid and cp['preparation_key']==prep['preparation_key'] and cp['files']['proposals.jsonl']==expected, 'Checkpoint ancestry')
            require(type(cp['proposal_count']) is int and 0<=cp['proposal_count']<=contract['proposals']-total, 'Proposal admission')
            rows, receipt=select_stream(root/sid/'proposals.jsonl',sid,expected,cp['proposal_count'],accepted,budget.check,aggregate)
            stats[sid+'/proposals.jsonl']=receipt['input_stat']; total+=receipt['proposal_count']; candidates[identity['split']]+=len(rows)
            name='scans/'+sid+'.json'; save(out/name, {'identity':identity,'records':rows,'receipt':receipt})
            raw=(out/name).read_bytes(); files[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
            append(out/'streams.jsonl',dict(receipt,split=identity['split'])); completed+=1
        require(total==contract['proposals'] and aggregate.hexdigest()==prep['files']['proposal_pool.jsonl'], 'Cohort stream aggregate')
        for name,st in stats.items(): budget.check(); require(signature(root/name)==st,'Input mutation: '+name)
        budget.check(force=True)
        result={'status':'COMPLETE_UNFROZEN','reference_release':'HOLD','policy':POLICY,'preparation_sha256':preparation_hash,
                'scans_manifest':manifest_receipt,'identities':identities,'files':files,'candidate_counts':candidates,
                'proposal_count':total,'resources':budget.summary(),'reference_labels_evaluated':False,'policy_status':'NOT_ADOPTED'}
        save(out/'candidate-manifest.json',result); return result
    except BaseException as ex:
        save(out/'failure.json',{'status':'INCOMPLETE','verified_scans':completed,'error':repr(ex),'partial':getattr(ex,'candidate_partial',None),'manifest_partial':getattr(ex,'manifest_partial',None),'reference_release':'HOLD'})
        raise

