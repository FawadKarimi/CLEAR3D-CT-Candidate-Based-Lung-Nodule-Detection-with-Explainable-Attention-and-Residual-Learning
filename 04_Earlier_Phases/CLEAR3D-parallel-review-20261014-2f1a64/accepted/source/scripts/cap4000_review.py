"""Plan-only metadata entry; adoption is an explicit future operator input."""
import argparse,hashlib,json,os,signal,sys,shutil
from pathlib import Path
WORK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK/'src'));sys.path.insert(0,str(WORK/'scripts'))
from lidc3d.accounting import require,read_bound,raw_json,save,resources,PREPARATION_SHA256
from lidc3d.cap4000 import POLICY,Cap4000Budget,build
import user_watchdog as guards
LIMITS={'total_seconds':14400,'rss_bytes':2*1024**3,'sample_seconds':1.0,'min_ram_bytes':8*1024**3,'reserve_ram_bytes':4*1024**3,'min_disk_bytes':32*1024**3,'reserve_disk_bytes':8*1024**3,'output_bytes':12*1024**3}

def disposition(d,binding_hash,action,manifest_hash=None):
    require(d.get('status')=='ADOPTED' and bool(d.get('supplied_by')) and bool(d.get('instruction')),'Actual adoption required')
    require(d.get('scope')=='cap4000-metadata-only-v1' and d.get('binding_sha256')==binding_hash and d.get('preparation_sha256')==PREPARATION_SHA256,'Disposition binding')
    require(d.get('policy')==POLICY and d.get('limits')==LIMITS and d.get('reference_release')=='HOLD' and d.get('policy_status')=='NOT_ADOPTED','Policy/HOLD disposition')
    require(d.get('labels_authorized') is False and action in d.get('actions',[]),'Metadata-only action scope')
    if action=='index':require(d.get('manifest_sha256')==manifest_hash,'Index manifest disposition')

def execute(a):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('PYTHONNOUSERSITE')=='1','CPU isolation required')
    require(not any(os.environ.get(k) for k in ('PYTHONHOME','PYTHONPATH','LD_LIBRARY_PATH','LD_PRELOAD')),'Inherited overrides refused')
    b=guards.bindings(a.binding,a.binding_sha256);base=WORK.parent
    require(b['preparation_sha256']==PREPARATION_SHA256 and b['policy']==POLICY and b['limits']==LIMITS,'Bound policy/limits')
    require(b['source_root']==str(WORK) and b['reference_release']=='HOLD','Source/HOLD binding')
    require(a.acknowledge_sampled_risk and a.disposition and a.disposition_sha256,'Missing acknowledgement/disposition')
    d=raw_json(read_bound(a.disposition,a.disposition_sha256,1024**2)[0]);disposition(d,a.binding_sha256,a.execute,a.manifest_sha256)
    out=Path(b['outputs'][a.execute]);parent=out.parent
    require(parent.is_dir() and not any(x.is_symlink() for x in (out,*out.parents)) and not out.exists(),'Fresh real destination')
    require(Path(b['prepared']).is_dir() and not any(x.is_symlink() for x in (Path(b['prepared']),*Path(b['prepared']).parents)),'Safe preparation parent')
    require(out==base/'runs'/('cap4000-'+a.execute+'-001'),'Prescribed destination')
    out.mkdir();handlers={};budget=Cap4000Budget(out)
    def stop(sig,frame):raise InterruptedError('External signal '+str(sig))
    try:
        save(out/'invocation.json',{'argv':sys.argv,'binding_sha256':a.binding_sha256,'disposition_sha256':a.disposition_sha256,'limits':LIMITS,'imports':{'accounting':str(sys.modules['lidc3d.accounting'].__file__),'guards':str(guards.__file__)}})
        require(resources()[1]>=LIMITS['min_ram_bytes'] and shutil.disk_usage(out).free>=LIMITS['min_disk_bytes'],'Admission headroom')
        for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,stop)
        budget.check(force=True)
        read_bound(Path(b['prepared'])/'preparation.json',PREPARATION_SHA256,16*1024**2)
        identities=raw_json(read_bound(b['identities']['path'],b['identities'],2*1024**2)[0])['identities']
        require(len(identities)==797 and sum(x['split']=='train' for x in identities)==708 and sum(x['split']=='val' for x in identities)==89,'Exact cohort')
        if a.execute=='build':
            raw=read_bound(b['receipts']['path'],b['receipts'],4*1024**2)[0];receipts=[raw_json(line) for line in raw.splitlines()]
            build(b['prepared'],out,receipts,identities,budget=budget)
        else:
            require(a.manifest_sha256,'Exact build manifest hash required')
            os.environ['SQLITE_TMPDIR']=str(out)
            from lidc3d.cap4000_index import flatten_manifest
            from lidc3d.metadata_index import build_index
            root=Path(b['outputs']['build']);manifest=raw_json(read_bound(root/'candidate-manifest.json',a.manifest_sha256,16*1024**2)[0])
            require(manifest['identities']==identities and manifest['preparation_sha256']==PREPARATION_SHA256,'Index manifest ancestry')
            receipt=flatten_manifest(root,a.manifest_sha256,out/'candidates.jsonl',budget.check)
            result=build_index(out/'candidates.jsonl',receipt['sha256'],identities,out,budget.check)
            require(result['roles']==receipt['roles'] and result['rows']==receipt['rows'],'Index reconciliation')
        guards.bindings(a.binding,a.binding_sha256);budget.check(force=True)
        save(out/'completion.json',{'status':'PASS','worker_exit':0,'resources':budget.summary(),'reference_release':'HOLD','policy_status':'NOT_ADOPTED','labels_evaluated':False});return 0
    except BaseException as ex:
        save(out/'launcher-failure.json',{'status':'INCOMPLETE','worker_exit':1,'error':repr(ex),'resources':budget.summary(),'reference_release':'HOLD'});return 1
    finally:
        for sig,old in handlers.items():signal.signal(sig,old)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--execute',choices=['build','index','label-index']);p.add_argument('--binding');p.add_argument('--binding-sha256');p.add_argument('--disposition');p.add_argument('--disposition-sha256');p.add_argument('--manifest-sha256');p.add_argument('--acknowledge-sampled-risk',action='store_true');a=p.parse_args(argv)
    if not a.execute:
        print(json.dumps({'status':'PLAN_ONLY','limits':LIMITS,'policy':POLICY,'reference_release':'HOLD','torch_imported':'torch' in sys.modules,'inputs_opened':False}));return 0
    if a.execute=='label-index':p.error('Label-dependent indexing requires a separately reviewed reference-use entry; not authorized by this metadata launcher')
    try:return execute(a)
    except BaseException as ex:
        print(json.dumps({'status':'PREFLIGHT_REFUSAL','error':repr(ex),'worker_exit':1,'reference_release':'HOLD'}),file=sys.stderr);return 1
if __name__=='__main__':raise SystemExit(main())
