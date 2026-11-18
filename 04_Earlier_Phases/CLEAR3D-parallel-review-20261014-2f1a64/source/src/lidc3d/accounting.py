"""Label-free, bounded five-threshold accounting; never freeze or select a policy."""
import hashlib
import heapq
import json
import math
import os
import re
import time
from pathlib import Path
from .streaming_selection import Worst

THRESHOLDS = (0.005, 0.01, 0.02, 0.04, 0.08)
CAP = 400
MANIFEST_BYTES = 131674595
MANIFEST_TOTAL_LIMIT = 160*1024**2
MANIFEST_LINE_LIMIT = 1024**2
PREPARATION_SHA256 = '47f5a6a740b405d758431e56f241993e48c93b330fa7859d93a7133378fa120a'
CONTRACT = {'scans': 797, 'patients': 796, 'train': 708, 'val': 89, 'proposals': 36183150}
LIMITS = {'total_seconds': 14400, 'rss_bytes': 2*1024**3, 'sample_seconds': 1.0,
          'min_ram_bytes': 8*1024**3, 'reserve_ram_bytes': 4*1024**3,
          'min_disk_bytes': 1024**3, 'evidence_bytes': 256*1024**2}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def regular(path):
    p = Path(path).absolute()
    require(not any(q.is_symlink() for q in (p, *p.parents)) and p.is_file(), 'Unsafe input path: '+str(p))
    return p


def signature(path):
    s = regular(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def raw_json(raw):
    def unique(pairs):
        d = {}
        for key, value in pairs:
            require(key not in d, 'Duplicate JSON key')
            d[key] = value
        return d
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))


def read_bound(path, expected, limit):
    p = regular(path)
    before = signature(p)
    require(before[2] <= limit, 'Metadata admission limit')
    with p.open('rb') as f:
        raw = f.read(limit+1)
    require(len(raw) <= limit and signature(p) == before, 'Metadata changed/admission limit')
    digest = expected if isinstance(expected, str) else expected['sha256']
    require(hashlib.sha256(raw).hexdigest() == digest, 'Metadata hash mismatch: '+str(p))
    if isinstance(expected, dict):
        require(len(raw) == expected['bytes'], 'Metadata byte mismatch')
    return raw, before


def append(path, obj):
    with Path(path).open('a', encoding='utf8') as f:
        f.write(json.dumps(obj, sort_keys=True, allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())


def save(path, obj):
    p = Path(path)
    with p.open('x', encoding='utf8') as f:
        json.dump(obj, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def resources():
    status = Path('/proc/self/status').read_text()
    rss = int(next(x for x in status.splitlines() if x.startswith('VmRSS:')).split()[1])*1024
    ram = int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
    return rss, ram


class Budget:
    """Cooperative sampled checks, NOT hard containment or a separate watchdog."""
    def __init__(self, out, limits=LIMITS, clock=time.monotonic, measure=resources):
        self.out, self.limits, self.clock, self.measure = Path(out), limits, clock, measure
        self.start = clock(); self.last = None; self.samples = 0; self.peak = 0
    def check(self, force=False):
        now = self.clock()
        require(now-self.start <= self.limits['total_seconds'], 'TOTAL_TIME_LIMIT')
        if force or self.last is None or now-self.last >= self.limits['sample_seconds']:
            rss, ram = self.measure(); self.last = now; self.samples += 1; self.peak = max(self.peak, rss)
            append(self.out/'resources.jsonl', {'elapsed_seconds': now-self.start, 'pid': os.getpid(),
                   'rss_bytes': rss, 'available_ram_bytes': ram, 'sample': self.samples})
            require(rss <= self.limits['rss_bytes'], 'SAMPLED_RSS_LIMIT')
            require(ram >= self.limits['reserve_ram_bytes'], 'HOST_RAM_HEADROOM')
            require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= self.limits['evidence_bytes'], 'EVIDENCE_SIZE_LIMIT')
    def summary(self):
        return {'runtime_seconds': self.clock()-self.start, 'samples': self.samples,
                'peak_sampled_rss_bytes': self.peak, 'enforcement': 'cooperative sampling; not containment'}


class FiveHeaps:
    """Store only ranking keys, never proposal payloads or reference labels."""
    def __init__(self):
        self.heaps = [[] for _ in THRESHOLDS]
        self.qualified = [0]*len(THRESHOLDS)
        self.peak_items = 0
    def add(self, response, identifier, ordinal):
        require(type(response) in (float, int) and math.isfinite(response), 'Invalid response')
        require(isinstance(identifier, str), 'Invalid candidate ID')
        key = (-response, identifier, ordinal)
        for i, threshold in enumerate(THRESHOLDS):
            if response < threshold:
                continue
            self.qualified[i] += 1
            heap = self.heaps[i]
            item = Worst(key, None)
            if len(heap) < CAP:
                heapq.heappush(heap, item)
            elif key < heap[0].key:
                heapq.heapreplace(heap, item)
        self.peak_items = max(self.peak_items, sum(map(len, self.heaps)))
    def ordered_keys(self):
        return [[x.key for x in sorted(heap, key=lambda x: x.key)] for heap in self.heaps]
    def report(self):
        result = {}
        for threshold, count, keys in zip(THRESHOLDS, self.qualified, self.ordered_keys()):
            h = hashlib.sha256()
            for key in keys:
                h.update((json.dumps(key, separators=(',', ':'), allow_nan=False)+'\n').encode())
            result[str(threshold)] = {'qualifying_before_cap': count, 'candidate_count': len(keys),
                                     'ranked_keys_sha256': h.hexdigest()}
        return result


def account_stream(path, sid, expected, count, tick=lambda **kw: None, aggregate=None):
    require(type(count) is int and count >= 0, 'Invalid proposal count')
    require(type(expected['bytes']) is int and expected['bytes'] >= 0, 'Invalid stream size')
    p = regular(path); before = signature(p)
    require(before[2] == expected['bytes'], 'Proposal byte admission mismatch')
    h = hashlib.sha256(); size = 0; n = 0; heaps = FiveHeaps(); started = time.monotonic()
    try:
        with p.open('rb') as f:
            while True:
                if n % 256 == 0:
                    tick()
                raw = f.readline(16385)
                if not raw:
                    break
                require(len(raw) <= 16384 and raw.strip() and n < count, 'Proposal admission limit')
                h.update(raw); size += len(raw)
                if aggregate is not None:
                    aggregate.update(raw)
                row = raw_json(raw)
                require(row.get('scan_id') == sid and row.get('id') == f'{sid}:candidate:{n:06d}', 'Proposal ordinal/identity mismatch')
                heaps.add(row['log_response'], row['id'], n)
                n += 1
        require(n == count and size == expected['bytes'] and h.hexdigest() == expected['sha256'], 'Incomplete/changed proposal stream')
        require(signature(p) == before, 'Proposal changed during read')
        tick(force=True)
        return {'scan_id': sid, 'status': 'VERIFIED', 'proposal_count': n, 'bytes': size,
                'sha256': h.hexdigest(), 'thresholds': heaps.report(), 'peak_heap_items': heaps.peak_items,
                'runtime_seconds': time.monotonic()-started, 'input_stat': before}
    except BaseException as ex:
        ex.accounting_partial = {'scan_id': sid, 'status': 'UNVERIFIED', 'rows_parsed': n,
                                'bytes_read': size, 'prefix_sha256': h.hexdigest(), 'error': repr(ex)}
        raise


def read_manifest(path, expected_hash, expected_bytes, contract, tick=lambda **kw: None):
    """Verify all binary lines before returning provisional identity-only records."""
    scans = []; ids = set(); series = set(); fingerprints = set(); patient_roles = {}
    digest = hashlib.sha256(); consumed = 0; parsed = 0; before = None
    try:
        require(type(expected_bytes) is int and 0 <= expected_bytes <= MANIFEST_TOTAL_LIMIT,
                'Manifest expected-byte admission limit')
        p = regular(path); before = signature(p)
        require(before[2] == expected_bytes and before[2] <= MANIFEST_TOTAL_LIMIT,
                'Manifest byte-count admission mismatch')
        tick(force=True)
        with p.open('rb') as handle:
            while True:
                tick()
                raw = handle.readline(MANIFEST_LINE_LIMIT+1)
                if not raw:
                    break
                digest.update(raw); consumed += len(raw)
                require(consumed <= expected_bytes and consumed <= MANIFEST_TOTAL_LIMIT,
                        'Manifest total-byte limit')
                require(len(raw) <= MANIFEST_LINE_LIMIT and raw.strip(), 'Scan metadata line admission')
                require(parsed < contract['scans'], 'Scan metadata record admission')
                tick()
                s = raw_json(raw)
                sid = s['scan_id']; role = s['split']
                require(isinstance(sid, str) and re.fullmatch(r'[A-Za-z0-9_-]+', sid) is not None, 'Unsafe scan ID')
                require(sid not in ids and role in ('train', 'val') and s.get('fold') == 0 and s.get('status') == 'ok', 'Development identity/role/status mismatch')
                for k in ('patient_id', 'series_uid', 'pixel_fingerprint'):
                    require(isinstance(s.get(k), str) and bool(s[k]), 'Missing identity: '+k)
                require(s['series_uid'] not in series and s['pixel_fingerprint'] not in fingerprints, 'Duplicate series/image identity')
                require(patient_roles.setdefault(s['patient_id'], role) == role, 'Patient crosses roles')
                ids.add(sid); series.add(s['series_uid']); fingerprints.add(s['pixel_fingerprint'])
                scans.append({k: s[k] for k in ('scan_id', 'patient_id', 'series_uid', 'pixel_fingerprint', 'split', 'fold')})
                parsed += 1
                del s, raw
                tick()
        require(consumed == expected_bytes and digest.hexdigest() == expected_hash,
                'Incomplete/changed scans manifest')
        require(signature(p) == before, 'Manifest changed during read')
        actual = {'scans': len(scans), 'patients': len(patient_roles),
                  'train': sum(s['split']=='train' for s in scans), 'val': sum(s['split']=='val' for s in scans)}
        require(actual == {k: contract[k] for k in actual}, 'Cohort counts changed')
        tick(force=True)
        return scans, {'path': 'scans.jsonl', 'status': 'VERIFIED', 'bytes': consumed,
                       'sha256': digest.hexdigest(), 'records': parsed,
                       'retained_fields': ['scan_id','patient_id','series_uid','pixel_fingerprint','split','fold']}, before
    except BaseException as ex:
        ex.manifest_partial = {'status': 'UNVERIFIED', 'identities_verified': False,
                               'bytes_read': consumed, 'records_parsed_provisionally': parsed,
                               'prefix_sha256': digest.hexdigest(), 'expected_bytes': expected_bytes,
                               'total_byte_limit': MANIFEST_TOTAL_LIMIT, 'error': repr(ex)}
        raise


def account_cohort(root, preparation_hash, out, contract=CONTRACT, budget=None, manifest_bytes=MANIFEST_BYTES):
    """out already created exclusively by launcher; generated fixtures may supply contract."""
    root, out = Path(root).absolute(), Path(out).absolute()
    require(root != out and root not in out.parents and out not in root.parents, 'Output overlaps preparation')
    budget = budget or Budget(out)
    completed = 0; active = None; stats = {}; metadata_receipts = []
    def metadata(relative, expected, limit):
        raw, st = read_bound(root/relative, expected, limit)
        stats[relative] = st
        metadata_receipts.append({'path': relative, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
        return raw
    try:
        budget.check(force=True)
        prep = raw_json(metadata('preparation.json', preparation_hash, 16*1024**2))
        require(prep['status'] == 'PROPOSALS_NOT_YET_FROZEN', 'Preparation stage changed')
        require(prep['config']['candidates']['threshold_search'] == list(THRESHOLDS) and prep['config']['candidates']['cap'] == CAP and prep['minimum_pool_threshold'] == THRESHOLDS[0], 'Candidate policy changed')
        require(all(prep['derivation']['counts'][k] == v for k,v in contract.items()), 'Pinned derivative counts changed')
        scans, manifest_receipt, manifest_stat = read_manifest(root/'scans.jsonl', prep['files']['scans.jsonl'], manifest_bytes, contract, budget.check)
        stats['scans.jsonl'] = manifest_stat
        metadata_receipts.append(manifest_receipt)
        save(out/'cohort-identities.json', {'identities': scans, 'patient_separation': 'PASS', 'reference_release': 'HOLD'})
        totals = {role: {str(t): {'qualifying_before_cap': 0, 'candidate_count': 0, 'zero_candidate_scans': 0} for t in THRESHOLDS} for role in ('train','val')}
        aggregate = hashlib.sha256(); total_rows = total_bytes = 0
        for s in scans:
            budget.check(); sid = s['scan_id']; active = sid
            cpname = sid+'/checkpoint.json'
            cp = raw_json(metadata(cpname, prep['artifacts'][cpname], 2*1024**2))
            require(cp['scan_id'] == sid and cp['preparation_key'] == prep['preparation_key'], 'Checkpoint ancestry changed')
            name = sid+'/proposals.jsonl'; expected = prep['artifacts'][name]
            require(cp['files']['proposals.jsonl'] == expected, 'Checkpoint proposal binding changed')
            require(type(cp['proposal_count']) is int and 0 <= cp['proposal_count'] <= contract['proposals']-total_rows, 'Total proposal count admission')
            receipt = account_stream(root/name, sid, expected, cp['proposal_count'], budget.check, aggregate)
            stats[name] = tuple(receipt['input_stat']); receipt.update(split=s['split'], patient_id=s['patient_id'], path=name)
            append(out/'streams.jsonl', receipt)
            for t, counts in receipt['thresholds'].items():
                target = totals[s['split']][t]
                for k in ('qualifying_before_cap','candidate_count'):
                    target[k] += counts[k]
                target['zero_candidate_scans'] += int(counts['candidate_count'] == 0)
            total_rows += receipt['proposal_count']; total_bytes += receipt['bytes']; completed += 1
        require(total_rows == contract['proposals'], 'Incomplete total proposal count')
        require(aggregate.hexdigest() == prep['files']['proposal_pool.jsonl'], 'Concatenated stream aggregate binding mismatch')
        # Stat checks avoid a second proposal read. This detects ordinary mutation,
        # not malicious same-metadata rewrites; stream hashes attest bytes when read.
        for name, st in stats.items():
            budget.check(); require(signature(root/name) == st, 'Input changed before finalization: '+name)
        budget.check(force=True)
        save(out/'metadata-receipts.json', metadata_receipts)
        result = {'status': 'PASS', 'scope': 'label-free metadata accounting, not frozen candidates',
                  'reference_release': 'HOLD', 'preparation_sha256': preparation_hash,
                  'contract': contract, 'thresholds': THRESHOLDS, 'cap': CAP, 'role_counts': totals,
                  'verified_streams': completed, 'proposal_rows': total_rows, 'proposal_bytes': total_bytes,
                  'concatenated_stream_sha256': aggregate.hexdigest(), 'aggregate_file_read': False,
                  'input_stats_unchanged': True, 'ranking': 'response descending, ID ascending, stable source ordinal',
                  'resources': budget.summary(), 'policy_selection_performed': False}
        save(out/'accounting-result.json', result)
        return result
    except BaseException as ex:
        save(out/'accounting-failure.json', {'status': 'INCOMPLETE', 'error': repr(ex), 'active_scan': active,
             'verified_streams': completed, 'partial': getattr(ex, 'accounting_partial', None),
             'manifest_partial': getattr(ex, 'manifest_partial', None),
             'metadata_receipts': metadata_receipts, 'resources': budget.summary(), 'reference_release': 'HOLD'})
        raise
