"""Plan-only by default. Linux cgroup-v2 pilot supervision; no torch import here."""
from __future__ import annotations
import argparse, copy, csv, datetime, hashlib, importlib.metadata, io, json, os
from pathlib import Path
import signal, subprocess, sys, time, uuid

GIB = 1024**3
GPU = 'GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90'
LIMITS = dict(setup_seconds=3600, run_seconds=28800, validation_seconds=7200,
              rss_bytes=16*GIB, allocator_bytes=8*GIB, gpu_bytes=9*GIB,
              free_gpu_bytes=10*GIB, free_ram_bytes=32*GIB, free_disk_bytes=20*GIB,
              sample_seconds=1.0)
PHASES = {'preflight','bundle_verification','candidate_loading','labeling','model_setup',
          'training','epoch_validation','epoch_metrics','checkpoint_publication',
          'epoch_record','best_checkpoint_load','final_validation','final_selection','complete','failed'}
RUN_PHASES = {'training','epoch_validation','epoch_metrics','checkpoint_publication',
              'epoch_record','best_checkpoint_load','final_validation','final_selection','complete'}

def require(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def read(path): return json.loads(Path(path).read_text())

def append(path, obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(obj,sort_keys=True,allow_nan=False)+chr(10)); f.flush(); os.fsync(f.fileno())

def save(path,obj):
    with Path(path).open('x') as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def inside(p,root):
    p,root=Path(p).resolve(),Path(root).resolve()
    return p==root or root in p.parents

def clean_environment(work, gpu=True):
    env=os.environ.copy()
    for k in ('PYTHONHOME','PYTHONPATH','LD_LIBRARY_PATH','LD_PRELOAD'): env.pop(k,None)
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(Path(work)/'src'),
               CUDA_VISIBLE_DEVICES=GPU if gpu else '',CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    return env

def prerequisites(request,contract,work):
    """Metadata gates before any child/CUDA. Hashes establish binding, not authorship."""
    require(request.get('scope')=='one-epoch-development-pilot','Pilot scope absent')
    require(request.get('seed')==17 and request.get('resume') is None,'Only fresh seed17 pilot supported')
    for field in ('scientific_decision','execution_authorization','exposure_supplement'):
        item=request.get(field,{})
        require(bool(item.get('path')) and bool(item.get('sha256')),field+' absent')
        require(sha(item['path'])==item['sha256'],field+' hash mismatch')
    require(request['exposure_supplement']['sha256']==contract['exposure_supplement_sha256'],
            'Corrected exposure supplement binding mismatch')
    require(str(Path(sys.executable).absolute())==contract['interpreter'],'Wrong interpreter path')
    require(sha(sys.executable)==contract['interpreter_sha256'],'Wrong interpreter bytes')
    require(len(contract['pins'])==44,'Incomplete environment pins')
    for name,expected in contract['pins'].items():
        require(importlib.metadata.version(name)==expected,'Environment pin mismatch: '+name)
    require({str(p.relative_to(work)) for p in Path(work).rglob('*') if p.is_file()}==set(contract['sources']),
            'Candidate file inventory differs from reviewed contract')
    for name,digest in contract['sources'].items():
        require(sha(Path(work)/name)==digest,'Source binding mismatch: '+name)
    package={str(p.relative_to(Path(work)/'src/lidc3d')):sha(p) for p in (Path(work)/'src/lidc3d').rglob('*.py')}
    require(canonical(package)==contract['code_identity'],'Unexpected package source inventory')
    prep=Path(contract['preparation'])
    require(sha(prep)==contract['preparation_sha256'],'Derivative binding changed')
    require(sha(prep.parent/'release-holds.json')==contract['holds_sha256'],'HOLD evidence changed')
    bp=Path(request['bundle']['path']).resolve()
    require(sha(bp)==request['bundle']['sha256'],'Bundle hash mismatch')
    bundle=read(bp)
    require(bundle.get('split_scope')=='development' and bundle.get('reference_audit_approved') is True
            and bundle.get('development_config_frozen') is True,'Approved frozen development bundle absent')
    # Never let arbitrary inventory paths make this wrapper hash withheld/raw files.
    for name in bundle['files']:
        require(inside(name,bp.parent) or inside(name,prep.parent),'Bundle inventory escapes development roots')
    for key in ('scans','candidates','resolved_config'):
        require(inside(bundle[key],bp.parent),'Bundle member escapes bundle root')
        require(str(Path(bundle[key]).resolve()) in bundle['files'],'Unbound bundle member')
    require(bundle['files'][str(Path(bundle['scans']).resolve())]==contract['scans_sha256'],
            'Frozen cohort differs from exact accepted derivative')
    rr=bp.parent/'reference_review.json'; trace=bp.parent/'candidate_selection.json'
    for p in (rr,trace,prep,prep.parent/'release-holds.json'):
        require(bundle['files'].get(str(p.resolve()))==sha(p),'Required binding absent or changed: '+p.name)
    review=read(rr)
    require(review.get('status')=='approved' and review.get('scans_sha256')==contract['scans_sha256'],
            'Scoped reference review absent')
    for k in ('reviewer','geometry_audit','annotation_audit','source_exclusions_audit','prior_test_access'):
        require(bool(review.get(k)),'Missing reference review field: '+k)
    require(review.get('preparation_sha256')==contract['preparation_sha256'],'Review derivative mismatch')
    holds=read(prep.parent/'release-holds.json')['unresolved']
    require(all(isinstance(review.get('hold_resolutions',{}).get(k),str) and
                review['hold_resolutions'][k].strip() for k in holds),'Unresolved reference holds')
    cfg_path=Path(request['config']['path'])
    require(sha(cfg_path)==request['config']['sha256'],'Pilot config file hash mismatch')
    import yaml
    cfg=yaml.safe_load(cfg_path.read_text()); frozen=read(bundle['resolved_config'])
    expected=copy.deepcopy(contract['pilot_template'])
    threshold=frozen['candidates']['threshold']
    require(threshold in expected['candidates']['threshold_search'],'Unreviewed candidate threshold')
    require(read(trace)['threshold']==threshold,'Frozen threshold trace mismatch')
    expected['candidates']['threshold']=threshold
    require(cfg==expected,'Pilot differs from exact one-epoch config contract')
    base=copy.deepcopy(expected);base['training']['max_epochs']=100
    require(frozen==base,'Frozen configuration differs from preparation configuration')
    require(canonical(cfg)==request['config'].get('canonical_sha256'),'Exact resolved config binding absent')
    return bundle,cfg

def resource_gate(gpu_free,ram_free,disk_free):
    require(gpu_free>=LIMITS['free_gpu_bytes'],'Less than10GiB free GPU memory')
    require(ram_free>=LIMITS['free_ram_bytes'],'Insufficient host RAM headroom')
    require(disk_free>=LIMITS['free_disk_bytes'],'Insufficient disk space')

class Budget:
    def __init__(self,start,limits=None):
        self.limits=dict(LIMITS if limits is None else limits);self.start=start;self.run_start=None
        self.phase='preflight';self.phase_start=start
    def check(self,now,phase,rss,gpu,phase_started=None):
        require(phase in PHASES,'Unknown measurement phase')
        if phase!=self.phase:
            self.phase=phase;self.phase_start=now if phase_started is None else phase_started
        if phase in RUN_PHASES and self.run_start is None:self.run_start=self.phase_start
        l=self.limits
        if now-self.start>l['setup_seconds']+l['run_seconds']:return 'TOTAL_TIME_LIMIT'
        if self.run_start is None and now-self.start>l['setup_seconds']:return 'SETUP_TIME_LIMIT'
        if self.run_start is not None and now-self.run_start>l['run_seconds']:return 'RUN_TIME_LIMIT'
        if phase in ('epoch_validation','final_validation') and now-self.phase_start>l['validation_seconds']:
            return 'VALIDATION_TIME_LIMIT'
        if rss>l['rss_bytes']:return 'PROCESS_TREE_RSS_LIMIT'
        if gpu>l['gpu_bytes']:return 'OWNED_GPU_LIMIT'
        return None

class Cgroup:
    """Fresh exclusive delegated group. No discovery/movement of resident processes."""
    def __init__(self,parent,limit):
        parent=Path(parent)
        require(parent.is_absolute() and not any(p.is_symlink() for p in (parent,*parent.parents)),
                'Cgroup delegation must be an absolute nonsymlink directory')
        require(inside(parent,'/sys/fs/cgroup') and parent!=Path('/sys/fs/cgroup'),
                'A delegated cgroup-v2 directory is required')
        require((parent/'cgroup.controllers').is_file(),'Missing cgroup-v2 delegation')
        self.path=parent/('lidc3d-pilot-'+uuid.uuid4().hex)
        self.path.mkdir(mode=0o700)
        self.attached=False
        for name,value in [('memory.max',str(limit)),('memory.swap.max','0'),('memory.oom.group','1')]:
            (self.path/name).write_text(value)
        require((self.path/'cgroup.kill').exists(),'Atomic owned-tree kill unavailable; no fallback')
        require(not self.pids(),'New cgroup unexpectedly populated')
    def pids(self):
        return sorted({int(x) for p in self.path.rglob('cgroup.procs') for x in p.read_text().split()})
    def attach(self,pid):
        require(not self.attached and not self.pids(),'Cgroup already used')
        (self.path/'cgroup.procs').write_text(str(pid));self.attached=True
        require(pid in self.pids(),'Child did not enter dedicated cgroup')
    def sample(self):
        rows=[]
        for pid in self.pids():
            try:
                fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
                rows.append({'pid':pid,'start_ticks':int(fields[19]),
                             'rss_bytes':int(fields[21])*os.sysconf('SC_PAGE_SIZE')})
            except (FileNotFoundError,ProcessLookupError):continue
        return {'processes':rows,'rss_bytes':sum(x['rss_bytes'] for x in rows),
                'cgroup_memory_current':int((self.path/'memory.current').read_text()),
                'cgroup_memory_peak':int((self.path/'memory.peak').read_text()) if (self.path/'memory.peak').exists() else None,
                'memory_events':(self.path/'memory.events').read_text()}
    def kill(self):
        if self.attached:(self.path/'cgroup.kill').write_text('1')
    def empty(self):return not self.pids()

class OwnedChild:
    """A child cannot proceed past its pipe gate before cgroup attachment."""
    def __init__(self,argv,env,cgroup,log):
        self.cg=cgroup;self.proc=None;self.fd=None
        rd,wr=os.pipe()
        try:
            self.proc=subprocess.Popen(argv+['--gate-fd',str(rd)],env=env,pass_fds=(rd,),
                                       start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
            self.fd=os.pidfd_open(self.proc.pid)
            cgroup.attach(self.proc.pid)
            os.write(wr,b'1')
        except BaseException:
            if self.fd is not None:
                try:signal.pidfd_send_signal(self.fd,signal.SIGKILL)
                except ProcessLookupError:pass
            elif self.proc is not None:self.proc.kill() # unreaped direct Popen child only
            if self.proc is not None:self.proc.wait(timeout=5)
            raise
        finally:os.close(rd);os.close(wr)
    def cleanup(self):
        # Dedicated cgroup contains only this launched tree, including detached descendants.
        self.cg.kill()
        try:self.proc.wait(timeout=10)
        finally:
            if self.fd is not None:os.close(self.fd);self.fd=None
        end=time.monotonic()+5
        while not self.cg.empty() and time.monotonic()<end:time.sleep(.05)
        require(self.cg.empty(),'Owned child cleanup incomplete')

def gpu_snapshot():
    def query(fields,kind):
        command=['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits']
        p=subprocess.run(command,text=True,capture_output=True,timeout=3,check=True)
        return list(csv.reader(io.StringIO(p.stdout),skipinitialspace=True))
    cards=query('uuid,name,driver_version,memory.free,memory.total','gpu')
    card=[r for r in cards if r[0]==GPU]
    require(len(card)==1,'Selected GPU missing or ambiguous')
    r=card[0];require(len(r)==5,'Malformed GPU observation')
    require(r[1]=='NVIDIA GeForce RTX 2080 Ti' and r[2]=='535.230.02','GPU model/driver changed')
    processes=[]
    for row in query('gpu_uuid,pid,used_gpu_memory','compute-apps'):
        require(len(row)==3,'Malformed compute-process observation')
        if row[0]==GPU:processes.append({'pid':int(row[1]),'gpu_bytes':int(row[2])*1024**2})
    return {'uuid':r[0],'model':r[1],'driver':r[2],'free_bytes':int(r[3])*1024**2,
            'total_bytes':int(r[4])*1024**2,'processes':processes}

class Tail:
    def __init__(self,path):self.path=Path(path);self.offset=0;self.events=0;self.last=None
    def poll(self):
        if not self.path.exists():return
        with self.path.open('rb') as f:
            f.seek(self.offset)
            while True:
                line=f.readline()
                if not line or not line.endswith(bytes([10])):break
                event=json.loads(line);self.offset=f.tell();self.events+=1
                if 'counters' in event:self.last=event['counters']

def monitor(child,out,observe=gpu_snapshot,limits=None,stop=None):
    """Always preserve outputs; sampled limits fail closed, never repair checkpoints."""
    start=time.monotonic();budget=Budget(start,limits);result={'status':'FAIL','reason':'NOT_FINISHED'}
    tail=Tail(out/'training/optimization_boundaries.jsonl');peak_rss=peak_gpu=0;complete_samples=0
    try:
        while True:
            now=time.monotonic();phase=read(out/'phase.json') if (out/'phase.json').exists() else {'name':'preflight','started':start}
            require(start-10<=phase['started']<=now,'Invalid phase clock')
            resources=child.cg.sample();gpu=observe();owned={r['pid'] for r in resources['processes']}
            own_gpu=sum(r['gpu_bytes'] for r in gpu['processes'] if r['pid'] in owned)
            peak_rss=max(peak_rss,resources['rss_bytes']);peak_gpu=max(peak_gpu,own_gpu);complete_samples+=1
            tail.poll()
            append(out/'resources.jsonl',{'utc':utc(),'monotonic':now,'phase':phase,'tree':resources,
                   'gpu':gpu,'owned_gpu_bytes':own_gpu,'update_counters':tail.last})
            if phase.get('run_started') is not None:
                require(start-10<=phase['run_started']<=now,'Invalid run clock')
                budget.run_start=phase['run_started']
            reason=budget.check(now,phase['name'],resources['rss_bytes'],own_gpu,phase['started'])
            if stop and stop():reason='EXTERNAL_TERMINATION'
            if reason:result['reason']=reason;break
            code=child.proc.poll()
            if code is not None:
                result['child_exit_status']=code
                require(code==0,'Worker failed with exit '+str(code))
                require(phase['name']=='complete','Worker exited without complete measurements')
                summary=read(out/'worker-result.json')
                require(summary['status']=='PASS','Worker acceptance absent')
                result.update(status='PASS',reason='COMPLETE');break
            time.sleep(budget.limits['sample_seconds'])
    except BaseException as error:result['reason']=type(error).__name__+': '+str(error)
    finally:
        try:child.cleanup()
        except BaseException as error:result.update(status='FAIL',cleanup_error=str(error))
        result['child_exit_status']=child.proc.poll()
        try:tail.poll()
        except BaseException as error:result.update(status='FAIL',diagnostic_error=str(error))
        result.update(wall_seconds=time.monotonic()-start,peak_sampled_rss_bytes=peak_rss if complete_samples else None,complete_resource_samples=complete_samples,
                                  peak_sampled_owned_gpu_bytes=peak_gpu if complete_samples else None,update_counters=tail.last,
                                  diagnostic_events=tail.events,throughput_scope='shared GPU',finished_utc=utc())
        try:append(out/'gpu-after.jsonl',{'utc':utc(),'gpu':observe()})
        except BaseException as error:result.update(status='FAIL',after_observation_error=str(error))
        save(out/'supervisor-result.json',result)
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    for k in ('contract','contract-sha256','request','request-sha256','out','cgroup-parent'):p.add_argument('--'+k)
    for k in ('reference-decision','pilot-execution','shared-gpu-permission'):p.add_argument('--acknowledge-'+k,action='store_true')
    args=p.parse_args(argv)
    if not args.execute:
        print(json.dumps({'status':'PLAN_ONLY','reference_release':'HOLD','limits':LIMITS,
                          'requires':'Reviewed request/config/bundle/scientific decision/execution permission and exclusive cgroup-v2 delegation',
                          'cuda_imported':False},indent=2));return 0
    out=None;owned_out=False;child=None;old={};stop=[False]
    try:
        require(all((args.acknowledge_reference_decision,args.acknowledge_pilot_execution,args.acknowledge_shared_gpu_permission)),
                'Explicit scoped acknowledgements required; technical acceptance is insufficient')
        require(all((args.contract,args.contract_sha256,args.request,args.request_sha256,args.out,args.cgroup_parent)),
                'Pinned prerequisites absent')
        require(sha(args.contract)==args.contract_sha256 and sha(args.request)==args.request_sha256,'Input binding mismatch')
        contract,request=read(args.contract),read(args.request);work=Path(__file__).resolve().parents[1]
        require(contract['limits']==LIMITS,'Resource policy changed')
        for k in ('SIGINT','SIGTERM'):
            s=getattr(signal,k);old[s]=signal.signal(s,lambda signum,frame:stop.__setitem__(0,True))
        out=Path(args.out).absolute()
        protected=[Path(contract['preparation']).parent,work,Path(contract['live_checkout'])]
        require(not any(inside(out,x) or inside(x,out) for x in protected),'Output overlaps protected paths')
        require(not any(p.is_symlink() for p in (out,*out.parents)),'Symlink output')
        out.mkdir(exist_ok=False);owned_out=True
        save(out/'request-snapshot.json',request);save(out/'contract-snapshot.json',contract)
        save(out/'invocation.json',{'argv':sys.argv if argv is None else argv,'utc':utc(),
             'request_sha256':args.request_sha256,'contract_sha256':args.contract_sha256,'limits':LIMITS})
        prerequisites(request,contract,work)
        require(not stop[0],'Termination requested during gates')
        gpu=gpu_snapshot()
        mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        ram_free=int(mem['MemAvailable'].split()[0])*1024
        import shutil
        resource_gate(gpu['free_bytes'],ram_free,shutil.disk_usage(out).free)
        save(out/'gpu-before.json',gpu)
        cg=Cgroup(args.cgroup_parent,LIMITS['rss_bytes'])
        save(out/'cgroup-binding.json',{'path':str(cg.path),'exclusive_created':True,
               'memory_max':LIMITS['rss_bytes'],'swap_max':0,'assumption':'No external actor may migrate processes into this exclusive group'})
        require(not stop[0],'Termination requested before launch')
        command=[sys.executable,'-B',str(work/'scripts/pilot_worker.py'),'--contract',str(Path(args.contract).resolve()),
                 '--request',str(Path(args.request).resolve()),'--out',str(out),'--parent-pid',str(os.getpid()),
                 '--contract-sha256',args.contract_sha256,'--request-sha256',args.request_sha256]
        with (out/'worker.log').open('x') as log:
            child=OwnedChild(command,clean_environment(work),cg,log)
            result=monitor(child,out,stop=lambda:stop[0]);child=None
        return 0 if result['status']=='PASS' else 1
    except BaseException as error:
        if child is not None:
            try:child.cleanup()
            except BaseException:pass
        if owned_out and not (out/'refusal.json').exists():
            save(out/'refusal.json',{'status':'REFUSED_OR_FAILED','type':type(error).__name__,'reason':str(error),'utc':utc()})
        print(type(error).__name__+': '+str(error),file=sys.stderr);return 1
    finally:
        for s,handler in old.items():signal.signal(s,handler)

if __name__=='__main__':raise SystemExit(main())
