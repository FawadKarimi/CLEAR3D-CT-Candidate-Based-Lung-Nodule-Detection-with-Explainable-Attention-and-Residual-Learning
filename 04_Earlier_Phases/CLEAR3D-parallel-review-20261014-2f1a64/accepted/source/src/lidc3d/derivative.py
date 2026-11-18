"""Explicit, append-only image-equivalence exclusion derivatives.

No preprocessing, reference editing, identity merging, selection or release.
Only a separately authorized caller may pass execute=True.
"""
from __future__ import annotations
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
from .provenance import (canonical_hash, read_json, read_jsonl, sha256,
                         write_json, write_jsonl, verify_scan_identities)

POLICY = 'exclude_all_members_of_confirmed_image_equivalence_pair'
ARRAYS = ('image.npy', 'valid.npy', 'lung_mask.npy', 'base_mask.npy')
REUSE = (*ARRAYS, 'geometry.json', 'proposals.jsonl')
HOLD_IDS = ('reference_origin', 'excluded_reference_outliers',
            'geometry_and_source_scope', 'protocol_evaluation_disclosure',
            'prior_test_access', 'independent_reference_review')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def regular(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)),
            'Symlink input or output is not permitted')
    require(path.is_file(), 'Missing regular input: ' + str(path))
    return path


def checked(path, digest):
    path = regular(path)
    require(sha256(path) == digest, 'Input hash mismatch: ' + str(path))
    return path


def evidence(path):
    return {'bytes': Path(path).stat().st_size, 'sha256': sha256(path)}


def assignments(path):
    with regular(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == ['seriesuid', 'patient_id', 'official_subset',
            'patient_grouped_subset', 'assignment_changed'], 'Assignment columns changed')
        rows = list(reader)
    require(len({r['seriesuid'] for r in rows}) == len(rows), 'Duplicate assigned series')
    groups = {}
    for row in rows:
        official, grouped = int(row['official_subset']), int(row['patient_grouped_subset'])
        require(official in range(10) and grouped in range(10), 'Invalid subset')
        require(row['assignment_changed'] == str(official != grouped), 'Assignment flag mismatch')
        require(groups.setdefault(row['patient_id'], grouped) == grouped,
                'Patient crosses grouped subsets')
    return rows


def derive_protocol(rows, parent, exclusions, parent_hash):
    """Recalculate identities/counts from assignment metadata; never open test labels."""
    require({f['fold'] for f in parent['folds']} == set(range(10)) and
            len(parent['folds']) == 10, 'Protocol must contain ten unique folds')
    removed = {e['series_uid'] for e in exclusions}
    kept = [r for r in rows if r['seriesuid'] not in removed]
    require(len(rows) - len(kept) == len(exclusions), 'Exclusion series missing from protocol')
    folds = []
    for fold in sorted(parent['folds'], key=lambda x: x['fold']):
        role_map = {}
        require(set(fold['roles']) == {'train', 'validation', 'test'}, 'Invalid fold roles')
        for role, entry in fold['roles'].items():
            for subset in entry['subsets']:
                require(subset in range(10) and subset not in role_map, 'Overlapping fold subsets')
                role_map[subset] = role
        require(set(role_map) == set(range(10)), 'Incomplete fold subsets')
        result = {'fold': fold['fold'], 'roles': {}}
        for role, entry in fold['roles'].items():
            prior = [r for r in rows if role_map[int(r['patient_grouped_subset'])] == role]
            members = [r for r in kept if role_map[int(r['patient_grouped_subset'])] == role]
            require(len(prior) == entry['series'], 'Parent protocol series count mismatch')
            if 'patients' in entry:
                require(len({r['patient_id'] for r in prior}) == entry['patients'],
                        'Parent protocol patient count mismatch')
            result['roles'][role] = {'subsets': entry['subsets'], 'series': len(members),
                'patients': len({r['patient_id'] for r in members}),
                'series_uids': sorted(r['seriesuid'] for r in members),
                'patient_ids': sorted({r['patient_id'] for r in members}),
                'membership_sha256': canonical_hash(members),
                'excluded_reference_row_delta': -sum(e['ignored_count'] for e in exclusions
                    if role_map[e['patient_grouped_subset']] == role),
                'included_reference_row_delta': -sum(e['positive_count'] for e in exclusions
                    if role_map[e['patient_grouped_subset']] == role)}
        folds.append(result)
    return {'schema_version': 3, 'kind': 'image_equivalence_exclusion_protocol',
        'status': 'DERIVED_NOT_REFERENCE_APPROVED', 'parent_protocol_sha256': parent_hash,
        'series_count': len(kept), 'patient_count': len({r['patient_id'] for r in kept}),
        'folds': folds, 'reference_files_inherited': parent.get('reference_files', {}),
        'reference_count_scope': 'Deltas from excluded development rows; no withheld labels read',
        'evaluation': 'Modified population; not official LUNA16 benchmark equivalence'}, kept


def verify_derivative_metadata(output, scans, preparation):
    """Extra checks supplement, never replace, the existing preparation verifier."""
    output = Path(output)
    binding = preparation['derivation']
    expected = {'exclusion-ledger.json', 'cohort_development.jsonl', 'assignments.csv',
                'protocol.json', 'release-holds.json', 'preparation_plan.json', 'reuse-map.json'}
    require(set(binding.get('files', {})) == expected, 'Derivative metadata inventory changed')
    for name, digest in binding['files'].items():
        checked(output/name, digest)
    ledger = read_json(output/'exclusion-ledger.json')
    require(ledger['policy'] == POLICY and ledger['biological_identity_adjudication'] is False,
            'Unsupported exclusion policy')
    plan = read_json(output/'preparation_plan.json')
    plan_body = {k: v for k, v in plan.items() if k != 'preparation_key'}
    require(canonical_hash(plan_body) == plan['preparation_key'] == preparation['preparation_key'],
            'Derivative preparation key changed')
    require(plan['ledger_sha256'] == sha256(output/'exclusion-ledger.json') and
            plan['protocol_sha256'] == sha256(output/'protocol.json'), 'Plan metadata binding changed')
    require(plan['minimum_pool_threshold'] == preparation['minimum_pool_threshold'], 'Threshold binding changed')
    cohort = read_jsonl(output/'cohort_development.jsonl')
    require(canonical_hash(cohort) == preparation['cohort_hash'] == plan['cohort_hash'],
            'Derivative cohort binding changed')
    require(plan['config_hash'] == canonical_hash(preparation['config']), 'Derivative config changed')
    require([r['scan_id'] for r in cohort] == [r['scan_id'] for r in scans] == plan['scan_ids'],
            'Derivative scan order or membership changed')
    excluded = {r['scan_id'] for r in ledger['exclusions']}
    require(not excluded.intersection(plan['scan_ids']), 'Excluded scan reintroduced')
    verify_scan_identities(scans)
    reuse = read_json(output/'reuse-map.json')
    require(set(reuse) == set(plan['scan_ids']), 'Reuse inventory changed')
    for original, scan in zip(cohort, scans):
        for key, value in original.items():
            if key not in ('status', 'source_hashes'):
                require(scan.get(key) == value, 'Retained cohort field changed: ' + key)
        sid = scan['scan_id']
        checkpoint = read_json(output/sid/'checkpoint.json')
        require(checkpoint['scan_hash'] == canonical_hash(original), 'Checkpoint cohort hash changed')
        require(checkpoint['derivation'] == reuse[sid]['checkpoint_binding'], 'Checkpoint ancestry changed')
        for name in REUSE:
            require(checkpoint['files'][name] == reuse[sid]['reused_files'][name], 'Reused artifact changed')
        for name in ARRAYS:
            require(Path(scan[name[:-4]+'_path']).absolute() == output/sid/name,
                    'Derivative array path changed')
    rows = assignments(output/'assignments.csv')
    protocol = read_json(output/'protocol.json')
    require(protocol['assignment_sha256'] == sha256(output/'assignments.csv'), 'Protocol assignment binding changed')
    excluded_series = {e['series_uid'] for e in ledger['exclusions']}
    require(not excluded_series.intersection(r['seriesuid'] for r in rows), 'Excluded series reintroduced')
    mapping = {r['seriesuid']: r for r in rows}
    require({f['fold'] for f in protocol['folds']} == set(range(10)) and len(protocol['folds']) == 10,
            'Derivative fold inventory changed')
    for fold in protocol['folds']:
        seen = []
        for role, detail in fold['roles'].items():
            members = [r for r in rows if int(r['patient_grouped_subset']) in detail['subsets']]
            require(detail['series_uids'] == sorted(r['seriesuid'] for r in members) and
                detail['patient_ids'] == sorted({r['patient_id'] for r in members}) and
                detail['series'] == len(members) and detail['patients'] == len({r['patient_id'] for r in members}) and
                detail['membership_sha256'] == canonical_hash(members), 'Fold membership changed')
            seen.extend(r['seriesuid'] for r in members)
        require(len(seen) == len(set(seen)) == len(rows), 'Fold roles fail to partition population')
    for scan in scans:
        a = mapping.get(scan['series_uid'], {})
        require(a.get('patient_id') == scan['patient_id'] and
                int(a['patient_grouped_subset']) == scan['patient_grouped_subset'] and
                int(a['official_subset']) == scan['official_subset'], 'Retained assignment changed')
    counts = {'scans':len(scans), 'patients':len({s['patient_id'] for s in scans}),
        'train':sum(s['split']=='train' for s in scans), 'val':sum(s['split']=='val' for s in scans),
        'positive':sum(len(s['positive']) for s in scans), 'ignored':sum(len(s['ignored']) for s in scans),
        'protocol_series':len(rows),'protocol_patients':len({r['patient_id'] for r in rows}),
        'proposals':sum(read_json(output/s['scan_id']/'checkpoint.json')['proposal_count'] for s in scans)}
    require(counts == binding['counts'] == ledger['expected_counts'], 'Derivative counts changed')
    require(len(rows) == protocol['series_count'] and
            len({r['patient_id'] for r in rows}) == protocol['patient_count'],
            'Derivative protocol counts changed')
    require(set(HOLD_IDS).issubset(read_json(output/'release-holds.json')['unresolved']),
            'Required release findings disappeared')


def require_derivative_review(output, preparation, review):
    if 'derivation' not in preparation:
        return
    require(review.get('preparation_sha256') == sha256(Path(output)/'preparation.json'),
            'Reference review must bind exact derivative preparation')
    require(review.get('derivative_binding_sha256') == canonical_hash(preparation['derivation']),
            'Reference review must bind exact derivative ledger and protocol')
    resolutions = review.get('hold_resolutions', {})
    holds = read_json(Path(output)/'release-holds.json')['unresolved']
    require(all(isinstance(resolutions.get(k), str) and resolutions[k].strip() for k in holds),
            'Unresolved derivative reference-review and prior-test-access findings')


def _copy_verified(source, target, expected, mode):
    checked(source, expected['sha256'])
    require(not target.exists(), 'Refuse existing output')
    if mode == 'reflink-auto':
        subprocess.run(['cp', '--reflink=auto', '--no-clobber', '--', str(source), str(target)], check=True)
    else:
        with regular(source).open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
    require(evidence(target) == expected, 'Reused copy failed verification')
    require(not os.path.samefile(source, target), 'Hardlink reuse is not permitted')


def migrate_exclusion(parent, ledger_path, output, *, execute=False, reuse_mode='copy', ledger_sha256=None):
    """Future explicit execution entry point; never called on real data in this review."""
    require(execute is True, 'Migration requires separate authorization and execute=True')
    require(reuse_mode in ('copy', 'reflink-auto'), 'Unknown reuse mode')
    from . import pipeline
    parent, output = Path(parent).absolute(), Path(output).absolute()
    require(not output.exists() and parent != output and parent not in output.parents and
            output not in parent.parents, 'Output must be new and disjoint from accepted preparation')
    require(not any(p.is_symlink() for p in (output, *output.parents)), 'Symlink output')
    require(isinstance(ledger_sha256,str) and len(ledger_sha256)==64, 'Pinned ledger SHA-256 required')
    ledger_path = checked(ledger_path, ledger_sha256)
    ledger = read_json(ledger_path)
    require(ledger['policy'] == POLICY and ledger['biological_identity_adjudication'] is False,
            'Unsupported exclusion policy')
    require(len(ledger['exclusions']) == 2 and
            len({e['scan_id'] for e in ledger['exclusions']}) == 2, 'Exactly two exclusions required')
    require(set(HOLD_IDS).issubset(ledger['unresolved_release_findings']), 'Missing release findings')
    for name, digest in ledger['parent_prepared_files'].items():
        require(name in ('scans.jsonl', 'preparation.json', 'preparation_plan.json'), 'Unexpected parent file')
        checked(parent/name, digest)
    require(set(ledger['parent_prepared_files']) == {'scans.jsonl','preparation.json','preparation_plan.json'},
            'Missing original preparation bindings')
    for item in ledger['evidence'].values():
        checked(item['path'], item['sha256'])
    files = ledger['inputs']
    require(set(files) == {'cohort', 'assignments', 'protocol'}, 'Input inventory changed')
    for item in files.values():
        checked(item['path'], item['sha256'])
    cohort = read_jsonl(files['cohort']['path'])
    scans = read_jsonl(parent/'scans.jsonl')
    prep = read_json(parent/'preparation.json')
    source_plan = read_json(parent/'preparation_plan.json')
    require([r['scan_id'] for r in cohort] == [r['scan_id'] for r in scans], 'Original membership mismatch')
    require(len({r['scan_id'] for r in scans}) == len(scans), 'Duplicate scan ID')
    require(canonical_hash(cohort) == prep['cohort_hash'] == source_plan['cohort_hash'], 'Parent cohort mismatch')
    require(prep['preparation_key'] == source_plan['preparation_key'], 'Parent plan mismatch')
    require(canonical_hash({k:v for k,v in source_plan.items() if k!='preparation_key'}) == prep['preparation_key'],
            'Parent plan key mismatch')
    require(all(s['split'] in ('train','val') and s['status']=='ok' and s.get('fold')==0 for s in scans),
            'Only successful fold-0 development scans may be migrated')
    for scan in scans:
        require(output != Path(scan['source_root']).absolute() and
                Path(scan['source_root']).absolute() not in output.parents, 'Output overlaps raw sources')
    # Verify parent fully without invoking duplicate guard (the known parent pair is intentional).
    pool = pipeline.PreparedProposalPool(parent, scans)
    pipeline._verify_preparation_artifacts(parent, scans, pool, prep, prep['config'])
    byid = {s['scan_id']: s for s in scans}
    for entry in ledger['exclusions']:
        sid = entry['scan_id']; require(sid in byid, 'Exclusion absent from development')
        scan = byid[sid]
        for key in ('patient_id','study_uid','series_uid','patient_grouped_subset','pixel_fingerprint'):
            require(scan[key] == entry[key], 'Exclusion identity mismatch: '+key)
        require(canonical_hash(scan['positive']) == entry['positive_sha256'] and
                canonical_hash(scan['ignored']) == entry['ignored_sha256'], 'Exclusion references changed')
        require(len(scan['positive'])==entry['positive_count'] and len(scan['ignored'])==entry['ignored_count'],
                'Exclusion reference count mismatch')
        checked(parent/sid/'record.json', entry['record_sha256'])
    require(len({e['pixel_fingerprint'] for e in ledger['exclusions']})==1, 'Exclusions are not an image pair')
    removed = {e['scan_id'] for e in ledger['exclusions']}
    kept = [s for s in scans if s['scan_id'] not in removed]
    manifest = [s for s in cohort if s['scan_id'] not in removed]
    require(kept, 'Empty derivative')
    verify_scan_identities(kept)
    for original, scan in zip(cohort, scans):
        for key,value in original.items():
            if key not in ('status','source_hashes'):
                require(scan.get(key)==value, 'Parent manifest/prepared disagreement: '+key)
        require({str(Path(k).resolve()):v for k,v in original['source_hashes'].items()} ==
                scan['source_hashes'], 'Parent source hashes changed')
    rows = assignments(files['assignments']['path'])
    protocol_parent = read_json(files['protocol']['path'])
    require(protocol_parent['assignment_sha256']==files['assignments']['sha256'], 'Protocol assignment hash mismatch')
    protocol, new_rows = derive_protocol(rows, protocol_parent, ledger['exclusions'], files['protocol']['sha256'])
    mapping = {r['seriesuid']:r for r in rows}
    fold0 = next(f for f in protocol_parent['folds'] if f['fold']==0)
    non_test = {r['seriesuid'] for r in rows if int(r['patient_grouped_subset']) not in fold0['roles']['test']['subsets']}
    require(non_test=={s['series_uid'] for s in scans}, 'Parent development population not complete')
    for s in scans:
        a = mapping[s['series_uid']]
        require(a['patient_id']==s['patient_id'] and int(a['patient_grouped_subset'])==s['patient_grouped_subset']
            and int(a['official_subset'])==s['official_subset'], 'Assignment identity changed')
        role = 'validation' if s['split']=='val' else 'train'
        require(s['patient_grouped_subset'] in fold0['roles'][role]['subsets'], 'Fold-0 role mismatch')
    counts={'scans':len(kept),'patients':len({s['patient_id'] for s in kept}),
        'train':sum(s['split']=='train' for s in kept),'val':sum(s['split']=='val' for s in kept),
        'positive':sum(len(s['positive']) for s in kept),'ignored':sum(len(s['ignored']) for s in kept),
        'protocol_series':len(new_rows),'protocol_patients':len({r['patient_id'] for r in new_rows}),
        'proposals':sum(read_json(parent/s['scan_id']/'checkpoint.json')['proposal_count'] for s in kept)}
    require(counts==ledger['expected_counts'], 'Derivative expected counts mismatch')
    output.mkdir(parents=True, exist_ok=False)
    # Failure leaves an incomplete new directory, never a usable preparation marker.
    checked(ledger_path, ledger_sha256)
    shutil.copyfile(ledger_path, output/'exclusion-ledger.json')
    checked(output/'exclusion-ledger.json', ledger_sha256)
    write_jsonl(output/'cohort_development.jsonl', manifest)
    with (output/'assignments.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(new_rows)
    protocol['assignment_sha256']=sha256(output/'assignments.csv')
    write_json(output/'protocol.json',protocol)
    write_json(output/'release-holds.json',{'status':'HOLD_REFERENCE_RELEASE',
        'unresolved':ledger['unresolved_release_findings'], 'parent_evidence':ledger['evidence']})
    plan={'schema_version':1,'kind':'lidc3d_preparation_plan','implementation':'verified_exclusion_derivative_v1',
        'implementation_hash':canonical_hash({n:sha256(Path(__file__).parent/n) for n in
            ('derivative.py','pipeline.py','provenance.py')}),'cohort_hash':canonical_hash(manifest),
        'config_hash':canonical_hash(prep['config']),'minimum_pool_threshold':prep['minimum_pool_threshold'],
        'scan_ids':[s['scan_id'] for s in kept],'parent_preparation_key':prep['preparation_key'],
        'ledger_sha256':sha256(output/'exclusion-ledger.json'),'protocol_sha256':sha256(output/'protocol.json')}
    plan['preparation_key']=canonical_hash(plan);write_json(output/'preparation_plan.json',plan)
    derived=[];reuse={}
    for original, scan in zip(manifest, kept):
        sid=scan['scan_id'];old=parent/sid;dest=output/sid;dest.mkdir()
        checkpoint=read_json(old/'checkpoint.json')
        require(checkpoint['scan_hash']==canonical_hash(original), 'Original checkpoint manifest binding changed')
        for name in REUSE:_copy_verified(old/name,dest/name,checkpoint['files'][name],reuse_mode)
        record=copy.deepcopy(scan)
        for name in ARRAYS:record[name[:-4]+'_path']=str(dest/name)
        write_json(dest/'record.json',record)
        ancestry={'parent_checkpoint_sha256':sha256(old/'checkpoint.json'),
                  'parent_preparation_key':prep['preparation_key'],'ledger_sha256':plan['ledger_sha256']}
        new_checkpoint={**checkpoint,'preparation_key':plan['preparation_key'],
            'scan_hash':canonical_hash(original),'derivation':ancestry,'preparation_seconds':0,
            'reused_preparation_seconds':checkpoint['preparation_seconds'],
            'files':{n:evidence(dest/n) for n in (*REUSE,'record.json')}}
        write_json(dest/'checkpoint.json',new_checkpoint)
        reuse[sid]={'checkpoint_binding':ancestry,'reused_files':{n:checkpoint['files'][n] for n in REUSE}}
        derived.append(record)
    write_json(output/'reuse-map.json',reuse)
    write_jsonl(output/'scans.jsonl',derived)
    with (output/'proposal_pool.jsonl').open('xb') as dst:
        for s in derived:
            with (output/s['scan_id']/'proposals.jsonl').open('rb') as src:
                shutil.copyfileobj(src,dst,1024*1024)
    metadata=['exclusion-ledger.json','cohort_development.jsonl','assignments.csv','protocol.json',
              'release-holds.json','preparation_plan.json','reuse-map.json']
    result={'schema_version':1,'status':'PROPOSALS_NOT_YET_FROZEN','preparation_key':plan['preparation_key'],
        'cohort_hash':plan['cohort_hash'],'config':prep['config'],'minimum_pool_threshold':prep['minimum_pool_threshold'],
        'files':{n:sha256(output/n) for n in ('scans.jsonl','proposal_pool.jsonl')},
        'artifacts':pipeline._collect_preparation_artifacts(output,derived),
        'derivation':{'kind':'image_equivalence_exclusion_v1','counts':counts,
            'parent_prepared_files':ledger['parent_prepared_files'],
            'files':{n:sha256(output/n) for n in metadata}}}
    pipeline._verify_preparation_artifacts(output,derived,pipeline.PreparedProposalPool(output,derived),result,prep['config'])
    for n,digest in ledger['parent_prepared_files'].items():checked(parent/n,digest)
    checked(ledger_path, ledger_sha256)
    pipeline._write_aggregate(output,'preparation.json',write_json,result)
    return result
