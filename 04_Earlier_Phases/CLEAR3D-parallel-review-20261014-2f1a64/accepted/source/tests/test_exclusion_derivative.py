"""Synthetic-only correction tests: no real manifests, CT, labels or selection."""
import copy
import csv
from pathlib import Path
import numpy as np
import pytest
from lidc3d import pipeline, derivative
from lidc3d.provenance import (write_json, write_jsonl, read_json, read_jsonl,
    canonical_hash, sha256, verify_bundle, verify_scan_identities)


def snapshot(root):
    return {str(p.relative_to(root)):(sha256(p),p.stat().st_mtime_ns,p.stat().st_mode)
            for p in root.rglob('*') if p.is_file()}


@pytest.fixture
def case(tmp_path, monkeypatch):
    raw=tmp_path/'raw';raw.mkdir();audit=tmp_path/'synthetic-audit.json';write_json(audit,{'synthetic':True})
    cohort=[]
    for name,subset in [('a',4),('b',3),('c',2),('d',1)]:
        source=raw/(name+'.dcm');source.write_bytes(('SYNTHETIC_'+name).encode())
        ignored=[{'id':name+':i0','center_lps':[1.,2.,3.],'diameter_mm':-1.,
                  'radius_mm':5.,'strict_radius':True,'point_only':True,'mask_path':None,'reference_state':'excluded'}]
        if name=='c':ignored.append({**ignored[0],'id':name+':i1'})
        positive=[] if name in ('a','b') else [{'id':name+':p','center_lps':[0.,1.,2.],
            'diameter_mm':4.,'radius_mm':2.,'point_only':True,'strict_radius':True,'mask_path':None,'reference_state':'included'}]
        cohort.append({'scan_id':'scan_'+name,'patient_id':'patient_'+name,'study_uid':'study_'+name,
            'series_uid':'series_'+name,'source_root':str(raw),'dicom_paths':[str(source)],
            'source_hashes':{str(source):sha256(source)},'positive':positive,'ignored':ignored,
            'reference_audit_path':str(audit),'unresolved_associations':0,'fold':0,
            'official_subset':subset,'patient_grouped_subset':subset,'split':'val' if subset==1 else 'train',
            'status':'manifest_only_not_prepared'})
    config={'geometry':{},'preprocess':{},'candidates':{'threshold':.01,'threshold_search':[.01], 'cap':5}}
    def fake_prepare(scan,config,scan_dir,final_dir,threshold):
        r=copy.deepcopy(scan);r.update(status='ok',failure_reason=None,
            pixel_fingerprint='duplicate' if scan['scan_id'] in ('scan_a','scan_b') else scan['scan_id'],
            affine=np.eye(4).tolist(),native_affine=np.eye(4).tolist(),coordinate_aware_resampling=False)
        for n in derivative.ARRAYS:
            a=np.zeros((2,3,4),dtype=np.float32) if n=='image.npy' else np.ones((2,3,4),dtype=bool)
            np.save(scan_dir/n,a,allow_pickle=False);r[n[:-4]+'_path']=str(final_dir/n)
        write_json(scan_dir/'geometry.json',{'shape':[2,3,4],'affine':r['affine'],'metadata':{},'array_order':'zyx','patient_coordinates':'LPS_mm'})
        return r,[{'id':scan['scan_id']+f':candidate:{i:06d}','scan_id':scan['scan_id'],'zyx':[0,0,i],
                   'center_lps':[i,0,0],'log_response':.5} for i in range(3)]
    monkeypatch.setattr(pipeline,'_prepare_scan',fake_prepare)
    parent=tmp_path/'parent';scans,_=pipeline.prepare_candidates(cohort,config,parent,materialize_pool=False)
    def forbidden(*a,**k):raise AssertionError('Preprocessing/selection must not run')
    for n in ['_prepare_scan','_load_configured_series','resample_lps','lung_mask','generate','select_candidates']:
        monkeypatch.setattr(pipeline,n,forbidden)
    manifest=tmp_path/'cohort.jsonl';write_jsonl(manifest,cohort)
    rows=[{'seriesuid':'series_'+n,'patient_id':'patient_'+n,'official_subset':str(s),
        'patient_grouped_subset':str(s),'assignment_changed':'False'} for n,s in [('withheld_metadata_only',0),('d',1),('c',2),('b',3),('a',4)]]
    assign=tmp_path/'assignments.csv'
    with assign.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    folds=[]
    for k in range(10):
        roles={'test':[k],'validation':[(k+1)%10],'train':[s for s in range(10) if s not in [k,(k+1)%10]]}
        folds.append({'fold':k,'roles':{role:{'subsets':ss,'series':sum(int(r['patient_grouped_subset']) in ss for r in rows),
            'patients':sum(int(r['patient_grouped_subset']) in ss for r in rows)} for role,ss in roles.items()}})
    prot=tmp_path/'protocol.json';write_json(prot,{'assignment_sha256':sha256(assign),'folds':folds,'reference_files':{}})
    exclusions=[]
    for r in scans[:2]:
        exclusions.append({**{k:r[k] for k in ['scan_id','patient_id','study_uid','series_uid','patient_grouped_subset','pixel_fingerprint']},
            'record_sha256':sha256(parent/r['scan_id']/'record.json'),'positive_sha256':canonical_hash(r['positive']),
            'ignored_sha256':canonical_hash(r['ignored']),'positive_count':len(r['positive']),'ignored_count':len(r['ignored'])})
    ledger={'policy':derivative.POLICY,'biological_identity_adjudication':False,'exclusions':exclusions,
        'parent_prepared_files':{n:sha256(parent/n) for n in ['scans.jsonl','preparation.json','preparation_plan.json']},
        'inputs':{n:{'path':str(q),'sha256':sha256(q)} for n,q in [('cohort',manifest),('assignments',assign),('protocol',prot)]},
        'evidence':{'synthetic_only':{'path':str(audit),'sha256':sha256(audit)}},
        'unresolved_release_findings':{k:'SYNTHETIC unresolved finding' for k in derivative.HOLD_IDS},
        'expected_counts':{'scans':2,'patients':2,'train':1,'val':1,'positive':2,'ignored':3,
                           'protocol_series':3,'protocol_patients':3,'proposals':6}}
    lp=tmp_path/'ledger.json';write_json(lp,ledger)
    return dict(root=tmp_path,parent=parent,output=tmp_path/'new',cohort=cohort,scans=scans,ledger=ledger,lp=lp,config=config)


def run(case,**kwargs):
    return derivative.migrate_exclusion(case['parent'],case['lp'],case['output'],execute=True,ledger_sha256=sha256(case['lp']),**kwargs)


def rewrite_ledger(case):
    import json
    case['lp'].write_text(json.dumps(case['ledger']))


def test_coherent_derivative_reuses_bytes_preserves_refs_and_parent(case,monkeypatch):
    before=snapshot(case['parent'])
    real=derivative.read_jsonl
    def no_proposal_materialization(path):
        assert Path(path).name not in ('proposal_pool.jsonl','proposals.jsonl')
        return real(path)
    monkeypatch.setattr(derivative,'read_jsonl',no_proposal_materialization)
    monkeypatch.setattr(pipeline,'read_jsonl',no_proposal_materialization)
    prep=run(case);out=case['output'];scans=read_jsonl(out/'scans.jsonl');manifest=read_jsonl(out/'cohort_development.jsonl')
    assert snapshot(case['parent'])==before
    assert [s['scan_id'] for s in scans]==['scan_c','scan_d']
    assert manifest==case['cohort'][2:]
    # Migration has finished under the no-materialization guard.
    # The legacy per-scan loader intentionally reads three synthetic rows.
    monkeypatch.setattr(pipeline,'read_jsonl',real)
    for source,scan in zip(case['scans'][2:],scans):
        for k in source:
            if not k.endswith('_path') or k=='reference_audit_path':assert scan[k]==source[k]
        for n in derivative.REUSE:
            a=case['parent']/scan['scan_id']/n;b=out/scan['scan_id']/n
            assert a.read_bytes()==b.read_bytes() and not a.samefile(b)
        record,proposals=pipeline._load_scan_checkpoint(manifest[scans.index(scan)],prep['preparation_key'],out/scan['scan_id'])
        assert record==scan and len(proposals)==3
    assert (out/'proposal_pool.jsonl').read_bytes()==b''.join((out/s['scan_id']/'proposals.jsonl').read_bytes() for s in scans)
    monkeypatch.setattr(pipeline,'read_jsonl',no_proposal_materialization)
    pipeline._verify_preparation_artifacts(out,scans,pipeline.PreparedProposalPool(out,scans),prep,case['config'])
    assert len(verify_bundle({'files':{},'scans':str(out/'scans.jsonl')}))==2
    assert read_json(out/'release-holds.json')['status']=='HOLD_REFERENCE_RELEASE'
    assert not (out/'bundle.json').exists()


def test_all_ten_fold_memberships_recomputed_without_reassignment(case):
    run(case);out=case['output'];protocol=read_json(out/'protocol.json');rows=derivative.assignments(out/'assignments.csv')
    assert {r['seriesuid'] for r in rows}=={'series_c','series_d','series_withheld_metadata_only'}
    for f in protocol['folds']:
        all_ids=[]
        for role,r in f['roles'].items():
            expected=sorted(x['seriesuid'] for x in rows if int(x['patient_grouped_subset']) in r['subsets'])
            assert r['series_uids']==expected and r['series']==len(expected)
            assert r['patients']==len(r['patient_ids']);all_ids.extend(expected)
        assert len(all_ids)==len(set(all_ids))==3
    assert not list(out.rglob('*WITHHOLD*'))


@pytest.mark.parametrize('mutation',['hash','reference','identity','count','missing_hold','unknown_exclusion'])
def test_ledger_drift_fails_before_output(case,mutation):
    l=case['ledger']
    if mutation=='hash':l['parent_prepared_files']['scans.jsonl']='0'*64
    if mutation=='reference':l['exclusions'][0]['ignored_sha256']='0'*64
    if mutation=='identity':l['exclusions'][0]['patient_id']='wrong'
    if mutation=='count':l['expected_counts']['scans']=3
    if mutation=='missing_hold':l['unresolved_release_findings'].pop('prior_test_access')
    if mutation=='unknown_exclusion':l['exclusions'][0]['scan_id']='scan_missing'
    rewrite_ledger(case);before=snapshot(case['parent'])
    with pytest.raises(ValueError):run(case)
    assert not case['output'].exists() and snapshot(case['parent'])==before


def test_execute_opt_in_and_existing_destination_rejected(case):
    with pytest.raises(ValueError,match='separate authorization'):
        derivative.migrate_exclusion(case['parent'],case['lp'],case['output'])
    assert not case['output'].exists()
    case['output'].mkdir()
    with pytest.raises(ValueError,match='new and disjoint'):run(case)


@pytest.mark.parametrize('kind',['array','raw','proposals'])
def test_source_or_checkpoint_tampering_rejected(case,kind):
    p={'array':case['parent']/'scan_c/image.npy','raw':Path(case['cohort'][2]['dicom_paths'][0]),
       'proposals':case['parent']/'scan_c/proposals.jsonl'}[kind]
    with p.open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError):run(case)
    assert not case['output'].exists()


def test_copy_interruption_has_no_commit_marker_and_preserves_parent(case,monkeypatch):
    before=snapshot(case['parent']);real=derivative._copy_verified;count=0
    def fail(*args):
        nonlocal count
        real(*args);count+=1
        if count==2:raise OSError('synthetic disk failure')
    monkeypatch.setattr(derivative,'_copy_verified',fail)
    with pytest.raises(OSError,match='synthetic disk'):run(case)
    assert not (case['output']/'preparation.json').exists()
    assert snapshot(case['parent'])==before


def test_symlink_destination_rejected(case):
    link=case['root']/'linked';link.symlink_to(case['root'],target_is_directory=True)
    case['output']=link/'escape'
    with pytest.raises(ValueError,match='Symlink'):run(case)


def test_duplicate_guard_before_freeze_threshold_selection(case):
    with pytest.raises(ValueError,match='Repeated pixel volume'):
        pipeline.freeze_bundle(case['parent']/'scans.jsonl',case['parent']/'proposal_pool.jsonl',case['config'],{},case['root']/'frozen')
    assert not (case['root']/'frozen').exists()


def test_duplicate_guard_still_enabled_for_bundle(case):
    with pytest.raises(ValueError,match='Repeated pixel volume'):
        verify_bundle({'files':{},'scans':str(case['parent']/'scans.jsonl')})


def test_derivative_missing_review_fails_before_selection(case):
    run(case)
    with pytest.raises(ValueError,match='exact derivative preparation'):
        pipeline.freeze_bundle(case['output']/'scans.jsonl',case['output']/'proposal_pool.jsonl',case['config'],{},case['root']/'frozen')
    assert not (case['root']/'frozen').exists()


@pytest.mark.parametrize('kind',['protocol','holds','ledger','reuse'])
def test_derivative_metadata_tamper_rejected(case,kind):
    run(case);out=case['output'];names={'protocol':'protocol.json','holds':'release-holds.json','ledger':'exclusion-ledger.json','reuse':'reuse-map.json'}
    with (out/names[kind]).open('a') as f:f.write(' ')
    prep=read_json(out/'preparation.json');scans=read_jsonl(out/'scans.jsonl')
    with pytest.raises(ValueError,match='hash mismatch'):
        pipeline._verify_preparation_artifacts(out,scans,pipeline.PreparedProposalPool(out,scans),prep,case['config'])


def test_reflink_auto_reuses_without_shared_inode(case):
    run(case,reuse_mode='reflink-auto')
    for sid in ('scan_c','scan_d'):
        for name in derivative.REUSE:
            old=case['parent']/sid/name;new=case['output']/sid/name
            assert sha256(old)==sha256(new) and not old.samefile(new)


def test_ancestry_cannot_be_silently_stripped(case):
    run(case);out=case['output'];prep=read_json(out/'preparation.json');prep.pop('derivation')
    scans=read_jsonl(out/'scans.jsonl')
    with pytest.raises(ValueError,match='ancestry missing'):
        pipeline._verify_preparation_artifacts(out,scans,pipeline.PreparedProposalPool(out,scans),prep,case['config'])


def test_prior_test_access_hold_blocks_even_with_hash_bindings(case):
    prep=run(case);out=case['output']
    review={'preparation_sha256':sha256(out/'preparation.json'),
            'derivative_binding_sha256':canonical_hash(prep['derivation']),
            'hold_resolutions':{k:'SYNTHETIC statement' for k in derivative.HOLD_IDS if k!='prior_test_access'}}
    with pytest.raises(ValueError,match='prior-test-access'):
        pipeline.freeze_bundle(out/'scans.jsonl',out/'proposal_pool.jsonl',case['config'],review,case['root']/'frozen')
    assert not (case['root']/'frozen').exists()


@pytest.mark.parametrize('splits',[('train','train'),('train','val'),('val','test')])
def test_same_pixels_rejected_in_same_or_cross_role(splits):
    scans=[{'scan_id':'scan_'+str(i),'patient_id':'p'+str(i),'split':role,'pixel_fingerprint':'same'} for i,role in enumerate(splits)]
    with pytest.raises(ValueError,match='Repeated pixel volume'):verify_scan_identities(scans)


def test_rehashed_malformed_parent_aggregation_rejected(case):
    import json
    pool=case['parent']/'proposal_pool.jsonl';lines=pool.read_text().splitlines(True);pool.write_text(''.join(reversed(lines)))
    prep_path=case['parent']/'preparation.json';prep=read_json(prep_path);prep['files']['proposal_pool.jsonl']=sha256(pool)
    prep_path.write_text(json.dumps(prep));case['ledger']['parent_prepared_files']['preparation.json']=sha256(prep_path);rewrite_ledger(case)
    with pytest.raises(ValueError,match='aggregation changed'):run(case)
    assert not case['output'].exists()


def test_wrong_pinned_ledger_hash_rejected(case):
    with pytest.raises(ValueError,match="hash mismatch"):
        derivative.migrate_exclusion(case["parent"],case["lp"],case["output"],execute=True,ledger_sha256="0"*64)
    assert not case["output"].exists()
