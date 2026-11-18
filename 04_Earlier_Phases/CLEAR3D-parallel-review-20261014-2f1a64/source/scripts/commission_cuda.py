"""Review-only synthetic CUDA commissioning. Default is a plan, never execution.

No dataset/config/checkpoint input path is accepted. Model configuration unchanged; reviewed deterministic interpolation correction.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CODE_HASH = '74c5f99919e7ecda5e3ae12fd33c7a165a0501dbf6fc74c46a366af69a590eba'
GPU_UUID = 'GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90'
GIB = 1024**3
LIMITS = {'total_seconds': 900, 'worker_seconds': 180, 'rss_bytes': 8*GIB,
          'allocator_bytes': 8*GIB, 'device_process_bytes': 9*GIB, 'minimum_free_bytes': 10*GIB}
STAGES = ['fixture', 'probes', 'full-backward', 'repeat-a', 'repeat-b', 'interrupt', 'resume', 'compare']
PINS = {'torch': '2.7.1+cu118', 'numpy': '2.2.6', 'scipy': '1.15.3', 'pydicom': '3.0.2',
        'scikit-image': '0.25.2', 'scikit-learn': '1.7.2', 'PyYAML': '6.0.3', 'pytest': '8.4.2'}


def write(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, sort_keys=True, indent=2, allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan(gpu_uuid=GPU_UUID, shared_use=False):
    if type(shared_use) is not bool:
        raise ValueError("Shared-use permission must be an explicit boolean")
    if shared_use and gpu_uuid != GPU_UUID:
        raise ValueError("Shared-use permission covers only the reviewed GPU UUID")
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', gpu_uuid):
        raise ValueError('A complete explicit GPU UUID is required')
    return {'scope': 'GENERATED_SYNTHETIC_ONLY', 'reference_release': 'HOLD',
            'gpu_uuid': gpu_uuid, 'logical_device': 'cuda:0', 'limits': LIMITS,
            'shared_use_permission_confirmed': shared_use,
            'measurement_context': 'shared-GPU; not exclusive-device performance' if shared_use else 'no exclusive allocation claimed',
            'stages': STAGES, 'seed': 17, 'epochs': 2, 'training_candidates': 64,
            'validation_candidates': 2, 'validation_zero_candidate_scans': 1,
            'model': 'unchanged configs/reference.yaml model (hybrid, width16, patch48)',
            'determinism': 'strict; no warn_only/fallback/backend substitution',
            'environment': {'CUDA_VISIBLE_DEVICES': gpu_uuid, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                            'PYTHONHASHSEED': '17', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
            'package_code_sha256': CODE_HASH, 'target_packages': PINS,
            'success_means': 'bounded synthetic checks only; never release/training authorization'}


def command(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=15)
    if p.returncode:
        raise RuntimeError(f'Command failed {args}: {p.stderr}')
    return p.stdout


def gpu_processes(gpu_uuid):
    raw = command(['nvidia-smi', '-i', gpu_uuid,
                   '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'])
    processes = []
    for row in csv.reader(raw.splitlines()):
        if not row or not any(v.strip() for v in row):
            continue
        if len(row) != 2:
            raise ValueError('Malformed compute-process observation')
        pid, memory_mib = (int(v.strip()) for v in row)
        if pid <= 0 or memory_mib < 0:
            raise ValueError('Invalid compute-process observation')
        processes.append({'pid': pid, 'used_gpu_memory_mib': memory_mib})
    return processes


def process_observation(gpu_uuid, path, phase, owned_pid=None):
    record = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'gpu_uuid': gpu_uuid, 'phase': phase, 'owned_worker_pid': owned_pid}
    try:
        record['processes'] = gpu_processes(gpu_uuid)
    except BaseException as error:
        record['error'] = {'type': type(error).__name__, 'reason': str(error)}
        with Path(path).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, sort_keys=True) + '\n')
        raise
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, sort_keys=True) + '\n')
    return record['processes']


def gpu_preflight(gpu_uuid, shared_use=False):
    plan(gpu_uuid, shared_use)  # Explicit shared permission cannot be inherited from an environment variable.
    raw = command(['nvidia-smi', '-i', gpu_uuid,
                   '--query-gpu=uuid,name,driver_version,memory.total,memory.free', '--format=csv,noheader,nounits'])
    rows = list(csv.reader(raw.splitlines()))
    if len(rows) != 1:
        raise ValueError('GPU query did not identify exactly one device')
    values = [v.strip() for v in rows[0]]
    if values[0] != gpu_uuid or values[1] != 'NVIDIA GeForce RTX 2080 Ti':
        raise ValueError('Selected GPU identity/model differs from reviewed target')
    if values[2] != '535.230.02':
        raise ValueError('Driver differs from inspected version; review required')
    if int(values[4])*1024**2 < LIMITS['minimum_free_bytes']:
        raise ValueError('Insufficient free device memory')
    apps = gpu_processes(gpu_uuid)
    if apps and not shared_use:
        raise ValueError('Selected GPU has resident compute processes; explicit shared-use permission required')
    return {'raw_gpu': raw, 'compute_processes': apps, 'shared_use_permission_confirmed': shared_use,
            'note': 'Read-only check, not an exclusive reservation'}


def rss_bytes(pid):
    try:
        text = Path(f'/proc/{pid}/status').read_text()
    except FileNotFoundError:
        return 0
    for line in text.splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1])*1024
    return 0


def monitor(cmd, env, log_path, remaining, gpu_uuid=None, process_log=None):
    """Bound only the child we create; never signal another user's process."""
    started = time.monotonic(); peak = 0; peak_device = 0; reason = None; last_gpu = None
    if process_log is not None:
        process_observation(gpu_uuid, process_log, 'worker-before')
    with Path(log_path).open('x') as log:
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while p.poll() is None:
                elapsed = time.monotonic()-started
                peak = max(peak, rss_bytes(p.pid))
                if elapsed > min(LIMITS['worker_seconds'], remaining): reason = 'TIME_LIMIT'
                if peak > LIMITS['rss_bytes']: reason = 'RSS_LIMIT'
                if gpu_uuid and (last_gpu is None or elapsed-last_gpu >= 1):
                    processes = (process_observation(gpu_uuid, process_log, 'during-worker', p.pid)
                                 if process_log is not None else gpu_processes(gpu_uuid))
                    for row in processes:
                        if row['pid'] == p.pid:
                            peak_device = max(peak_device, row['used_gpu_memory_mib']*1024**2)
                    if peak_device > LIMITS['device_process_bytes']: reason = 'DEVICE_MEMORY_LIMIT'
                    last_gpu = elapsed
                if reason:
                    os.killpg(p.pid, signal.SIGTERM)
                    try: p.wait(timeout=5)
                    except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait()
                    break
                time.sleep(.1)
        except BaseException:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
            raise
        finally:
            if process_log is not None:
                process_observation(gpu_uuid, process_log, 'worker-after', p.pid)
    return {'command': cmd, 'pid': p.pid, 'exit_status': p.returncode, 'limit_failure': reason,
            'wall_seconds': time.monotonic()-started, 'sampled_peak_rss_bytes': peak,
            'sampled_peak_device_process_bytes': peak_device}


def equal(a, b, path='root'):
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            raise AssertionError('Tensor agreement failed: '+path)
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b): raise AssertionError('Array agreement failed: '+path)
    elif isinstance(a, dict):
        if a.keys() != b.keys(): raise AssertionError('Keys differ: '+path)
        for k in a: equal(a[k], b[k], path+'.'+str(k))
    elif isinstance(a, (tuple, list)):
        if type(a) != type(b) or len(a) != len(b): raise AssertionError('Sequence differs: '+path)
        for i,(x,y) in enumerate(zip(a,b)): equal(x,y,path+f'[{i}]')
    elif a != b: raise AssertionError('Value differs: '+path)


def cpu_tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu': raise AssertionError('Checkpoint tensor allocated on GPU')
    elif isinstance(value, dict):
        for v in value.values(): cpu_tensors(v)
    elif isinstance(value, (tuple,list)):
        for v in value: cpu_tensors(v)


def generated_fixture(out, small=False):
    import numpy as np
    import yaml
    from lidc3d.provenance import write_json, write_jsonl, sha256
    cfg = yaml.safe_load((ROOT/'configs/reference.yaml').read_text())
    cfg['training'].update(max_epochs=2, warmup_epochs=1)
    cfg['candidates']['threshold'] = .01  # arbitrary fixture constant, not threshold selection
    if small:
        cfg['model'].update(kind='cnn',width=4,patch_size=16)
        cfg['training'].update(microbatch=2,accumulation=2,amp=False)
    out.mkdir(exist_ok=False)
    size=cfg['model']['patch_size']+4; scans=[]; rows=[]
    count=4 if small else 64
    for i,split in enumerate(['train','val','val']):
        image=out/f'image-{i}.npy';valid=out/f'valid-{i}.npy'
        np.save(image,np.random.default_rng(i).random((size,size,size),dtype=np.float32))
        np.save(valid,np.ones((size,size,size),bool))
        center=[size//2]*3
        scans.append({'scan_id':f'SYNTHETIC-{i}','patient_id':f'SYNTHETIC-P-{i}',
          'pixel_fingerprint':f'GENERATED-FIXTURE-{i}','split':split,'status':'ok',
          'image_path':str(image),'valid_path':str(valid),
          'positive':[{'id':f'GENERATED-REFERENCE-{i}','center_lps':center,'radius_mm':1}], 'ignored':[]})
        for j in range(count if i==0 else 2 if i==1 else 0):
            c=center if j%2==0 else [2,2,2]
            rows.append({'id':f'GENERATED-{i}-{j}','scan_id':f'SYNTHETIC-{i}','zyx':c,'center_lps':c})
    for name,data in [('scans.jsonl',scans),('candidates.jsonl',rows)]:write_jsonl(out/name,data)
    write_json(out/'config.json',cfg)
    bundle={'scope':'SYNTHETIC_TEST_FIXTURE_NOT_A_RELEASE','files':{str(out/n):sha256(out/n) for n in ['scans.jsonl','candidates.jsonl','config.json']},
            'scans':str(out/'scans.jsonl'),'candidates':str(out/'candidates.jsonl'),'resolved_config':str(out/'config.json'),
            'reference_audit_approved':True,'development_config_frozen':True}
    # These booleans satisfy the runner only inside this generated fixture. They
    # do not attest to or alter any real reference review or frozen bundle.
    write_json(out/'synthetic-bundle.json',bundle)
    return bundle,cfg


def runtime_checks():
    import importlib.metadata
    import torch
    if sys.version_info[:3] != (3,10,17): raise ValueError('Python differs from proposed 3.10.17')
    for name,version in PINS.items():
        if importlib.metadata.version(name) != version: raise ValueError('Runtime pin mismatch: '+name)
    if torch.version.cuda != '11.8': raise ValueError('Unexpected bundled CUDA runtime')


def full_model_backward_probe(cfg, device):
    """Actual configured hybrid model, generated inputs, FP32 and AMP backward.

    CPU tests call this with device='cpu' (FP32 only); reviewed CUDA execution
    exercises both modes. No mode changes the model configuration or seed.
    """
    import torch
    from lidc3d.models import DetectorClassifier
    from lidc3d.provenance import seed_all
    modes = (False, True) if device == 'cuda' else (False,)
    results = {}
    for amp in modes:
        seed_all(17)
        model = DetectorClassifier(cfg['model']).to(device).train()
        n = cfg['model']['patch_size']
        x = torch.rand(2, 1, n, n, n, device=device, requires_grad=True)
        target = torch.tensor([0, 1], device=device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, target)
        if logits.shape != (2, 2) or not torch.isfinite(loss):
            raise AssertionError('Full-model forward is invalid')
        # Unscaled backward isolates operator compatibility from scaler dynamics;
        # the existing AMP and repeat stages still test GradScaler/optimizer.
        loss.backward()
        if x.grad is None or not torch.isfinite(x.grad).all() or not x.grad.abs().sum():
            raise AssertionError('Full-model input gradient is invalid')
        missing = [name for name, p in model.named_parameters()
                   if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
        if missing:
            raise AssertionError('Full-model parameter gradients invalid: ' + str(missing))
        results['AMP' if amp else 'FP32'] = {'loss': float(loss.detach().cpu()),
            'input_shape': list(x.shape), 'logits_dtype': str(logits.dtype),
            'parameter_gradient_tensors': sum(p.requires_grad for p in model.parameters()),
            'input_gradient_finite_nonzero': True}
        del model, x, logits, loss
    return results


def scaled_weighted_probe(bundle, cfg, out, device):
    """Use the real runner/weighted accumulation/scaler/clipping boundary path."""
    from lidc3d import training, checkpoints
    from lidc3d.provenance import read_jsonl
    training.train(bundle, cfg, out, 17, device)
    selected = checkpoints.load_set(out/'last.pt', {})
    counters = selected['last']['update_state']
    diagnostics = read_jsonl(out/'optimization_boundaries.jsonl')
    if counters['optimizer_updates'] <= 0 or not diagnostics:
        raise AssertionError('Scaled probe made no optimizer updates')
    return {'update_state': counters, 'scaler_state': selected['last']['scaler'],
            'boundary_evidence': str(out/'optimization_boundaries.jsonl'),
            'actual_training_path': True}


def run_worker(stage, out, gpu_uuid):
    import copy
    import resource
    import numpy as np
    import torch
    from lidc3d import training, checkpoints
    from lidc3d.provenance import seed_all, read_json, write_json, canonical_hash
    if checkpoints.code_identity()['sha256'] != CODE_HASH: raise ValueError('Reviewed package code changed')
    runtime_checks()
    cuda=stage not in ('fixture','compare')
    if cuda:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu_uuid or torch.cuda.device_count()!=1:
            raise ValueError('Exactly the selected GPU must be visible')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8': raise ValueError('cuBLAS setting missing')
        torch.cuda.set_device(0)
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(LIMITS['allocator_bytes']/total,0)
        torch.cuda.reset_peak_memory_stats()
    seed_all(17)
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
        raise ValueError('Strict determinism required')
    write(out/(stage+'-runtime.json'),{'torch':torch.__version__,'cuda_build':torch.version.cuda,
      'strict_deterministic':torch.are_deterministic_algorithms_enabled(),'warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
      'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,
      'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
      'flash_sdp':torch.backends.cuda.flash_sdp_enabled(),'mem_efficient_sdp':torch.backends.cuda.mem_efficient_sdp_enabled(),
      'math_sdp':torch.backends.cuda.math_sdp_enabled(),'code_identity':checkpoints.code_identity(),
      'script_sha256':sha(__file__)})
    metrics={'measurement_context': 'shared-GPU; not exclusive-device performance'
             if os.environ.get('LIDC3D_SHARED_GPU_PERMISSION') == '1' else 'no exclusive allocation claimed'}
    load_events=[]; optimizer_events=[]; scaler_events=[]
    if stage=='fixture':
        generated_fixture(out/'fixture'); return {'generated':True}
    if stage=='compare':
        states=[checkpoints.load_set(out/name/'last.pt',{}) for name in ['repeat-a','repeat-b','resume']]
        for current in states[1:]:
            for part in ['last','best']:
                for key in ['model','optimizer','scaler','update_state','epoch','scheduler','loader_rng','torch_rng','numpy_rng','python_rng','cuda_rng','best_cpm','stopping_best','stale']:
                    equal(states[0][part][key],current[part][key],part+'.'+key)
        for filename in ['selection.json','validation_predictions.jsonl']:
            if filename=='selection.json':
                first=read_json(out/'repeat-a'/filename)
                for name in ['repeat-b','resume']:
                    current=read_json(out/name/filename)
                    for key in ['selected_epoch','validation_cpm','operating_threshold']:equal(first[key],current[key])
            else:
                if len({sha(out/name/filename) for name in ['repeat-a','repeat-b','resume']})!=1:
                    raise AssertionError('FP32 prediction agreement failed')
        return {'repeat_exact':True,'resume_exact':True,'comparison':'exact tensors, RNG, optimizer and predictions; no tolerance relaxation'}
    real_load=torch.load
    def traced_load(*args,**kwargs):
        if kwargs.get('map_location')!='cpu':raise AssertionError('Checkpoint must load through CPU')
        torch.cuda.synchronize();before=torch.cuda.memory_allocated()
        result=real_load(*args,**kwargs);cpu_tensors(result)
        torch.cuda.synchronize();after=torch.cuda.memory_allocated()
        event={'before_allocated':before,'after_allocated':after,'map_location':'cpu'};load_events.append(event)
        if after!=before:raise AssertionError('GPU allocation changed during CPU checkpoint load')
        return result
    torch.load=traced_load
    try:
        if stage=='full-backward':
            cfg=read_json(out/'fixture/config.json')
            metrics['full_model_backward']=full_model_backward_probe(cfg, 'cuda')
            metrics['scaled_weighted_probe']=scaled_weighted_probe(
                read_json(out/'fixture/synthetic-bundle.json'), cfg, out/'scaled-probe', 'cuda:0')
        elif stage=='probes':
            bundle,cfg=read_json(out/'fixture/synthetic-bundle.json'),read_json(out/'fixture/config.json')
            from lidc3d.models import DetectorClassifier,signature
            model=DetectorClassifier(cfg['model']).cuda()
            metrics['architecture']=signature(model,cfg['model']['patch_size'])
            x=torch.rand(2,1,48,48,48,device='cuda');model.eval()
            torch.cuda.synchronize();started=time.perf_counter()
            with torch.no_grad():
                a=model(x);b=model(x)
            torch.cuda.synchronize();equal(a,b,'FP32-forward');assert a.dtype==torch.float32 and torch.isfinite(a).all()
            metrics['fp32_forward_samples_per_second']=4/(time.perf_counter()-started)
            del model,x,a,b
            results=[]
            for _ in range(2):
                seed_all(17);m=torch.nn.Linear(8,2).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001)
                scaler=torch.amp.GradScaler('cuda');before={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
                for step in range(2):
                    opt.zero_grad();x=torch.randn(4,8,device='cuda');y=torch.tensor([0,1,0,1],device='cuda')
                    with torch.autocast('cuda',dtype=torch.float16):loss=torch.nn.functional.cross_entropy(m(x),y)
                    assert torch.isfinite(loss)
                    scaler.scale(loss).backward();scaler.unscale_(opt)
                    assert all(torch.isfinite(p.grad).all() for p in m.parameters())
                    scaler.step(opt);scaler.update()
                if not opt.state or any(int(v['step'].item()) != 2 for v in opt.state.values()):
                    raise AssertionError('AMP primitive skipped an optimizer update')
                after={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
                assert any(not torch.equal(before[k],after[k]) for k in before)
                results.append((after,copy.deepcopy(opt.state_dict()),copy.deepcopy(scaler.state_dict())))
            equal(results[0],results[1],'AMP-primitive-repeat')
            metrics['amp_primitive_updates']=2;metrics['amp_primitive_repeat_exact']=True
        else:
            bundle=read_json(out/'fixture/synthetic-bundle.json');cfg=read_json(out/'fixture/config.json')
            real_adam=torch.optim.AdamW;real_scaler=torch.amp.GradScaler
            def traced_adam(*a,**kw):
                opt=real_adam(*a,**kw)
                def updated(opt,args,kwargs): optimizer_events.append({'update':len(optimizer_events)+1})
                opt.register_step_post_hook(updated);return opt
            class TracedScaler(real_scaler):
                def update(self,*a,**kw):
                    before=self.get_scale();result=super().update(*a,**kw)
                    scaler_events.append({'before':before,'after':self.get_scale()});return result
            torch.optim.AdamW=traced_adam;torch.amp.GradScaler=TracedScaler
            if stage=='interrupt':
                real_publish=training.publish_set
                def interrupt(*a,**kw):
                    real_publish(*a,**kw)
                    if a[1]['epoch']==0:raise InterruptedError('PLANNED_SYNTHETIC_EPOCH_BOUNDARY')
                training.publish_set=interrupt
            torch.cuda.synchronize();started=time.perf_counter()
            try:
                training.train(bundle,cfg,out/stage,17,'cuda:0',out/'interrupt/last.pt' if stage=='resume' else None)
            except InterruptedError as error:
                if stage!='interrupt' or str(error)!='PLANNED_SYNTHETIC_EPOCH_BOUNDARY':raise
                checkpoints.load_set(out/'interrupt/last.pt',{})
                if (out/'interrupt/selection.json').exists():raise AssertionError('Unexpected completion on interruption')
                metrics['planned_interruption_verified']=True
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            epochs=1 if stage in ('interrupt','resume') else 2
            from lidc3d.provenance import read_jsonl
            boundary_rows=[row for row in read_jsonl(out/stage/'optimization_boundaries.jsonl')
                           if row['kind']=='accumulation_boundary']
            updated=sum(row['status']=='UPDATED' for row in boundary_rows)
            skipped=sum(row['status']=='SKIPPED_AMP_NONFINITE' for row in boundary_rows)
            if (len(boundary_rows)!=4*epochs or updated<=0 or updated+skipped!=len(boundary_rows)
                    or len(optimizer_events)!=updated or len(scaler_events)!=len(boundary_rows)):
                raise AssertionError('Actual updates/skips do not match boundary evidence')
            metrics.update(optimizer_updates=updated, skipped_updates=skipped,
                           scaler_update_calls=len(scaler_events),
                           training_candidate_exposures=64*epochs,training_plus_validation_seconds=seconds,
                           end_to_end_candidate_exposures_per_second=64*epochs/seconds)
    finally:
        # Persist instrumentation even on deterministic-kernel/OOM/AMP failure.
        write(out/(stage+'-instrumentation.json'),{'cpu_checkpoint_loads':load_events,
              'optimizer_updates':optimizer_events,'scaler_updates':scaler_events,'partial_metrics':metrics,
              'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
              'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    return metrics


def execute(out,gpu_uuid,shared_use=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();report={'plan':plan(gpu_uuid,shared_use),'stages':{s:{'status':'NOT_RUN'} for s in STAGES},'status':'INCOMPLETE'}
    write(out/'invocation.json',{'argv':sys.argv,'script_sha256':sha(__file__),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan':report['plan']})
    try:
        if shared_use:
            process_observation(gpu_uuid, out/'gpu-process-observations.jsonl', 'before-execution')
        write(out/'preflight.json',gpu_preflight(gpu_uuid,shared_use))
        for stage in STAGES:
            remaining=LIMITS['total_seconds']-(time.monotonic()-started)
            if remaining<=0:raise TimeoutError('Total commissioning deadline exceeded')
            env=os.environ.copy();env.update(report['plan']['environment'])
            env.update(PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',LIDC3D_SYNTHETIC_COMMISSION='1',
                       LIDC3D_SHARED_GPU_PERMISSION='1' if shared_use else '0')
            if stage in ('fixture','compare'):env['CUDA_VISIBLE_DEVICES']=''
            cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',stage,'--output',str(out),'--gpu-uuid',gpu_uuid]
            report['stages'][stage]={'status':'RUNNING','command':cmd}
            write(out/(stage+'-started.json'),report['stages'][stage])
            try:
                if shared_use:
                    result=monitor(cmd,env,out/(stage+'.log'),remaining,gpu_uuid,
                                   process_log=out/'gpu-process-observations.jsonl')
                else:
                    result=monitor(cmd,env,out/(stage+'.log'),remaining,gpu_uuid if stage not in ('fixture','compare') else None)
            except BaseException as error:
                report['stages'][stage]={'status':'FAIL','command':cmd,'monitor_error':str(error)}
                raise
            result['status']='PASS' if result['exit_status']==0 and not result['limit_failure'] else 'FAIL'
            report['stages'][stage]=result
            write(out/(stage+'-process.json'),result)
            if result['status']!='PASS':raise RuntimeError('Commissioning stage failed: '+stage)
        report['status']='SYNTHETIC_CHECKS_PASS_NOT_RELEASE_APPROVAL'
    except BaseException as error:
        report['failure']={'type':type(error).__name__,'reason':str(error),'traceback':traceback.format_exc()}
        raise
    finally:
        active_failure = sys.exc_info()[0] is not None
        try:
            if shared_use:
                process_observation(gpu_uuid, out/'gpu-process-observations.jsonl', 'after-execution')
        except BaseException as error:
            report['observation_failure']={'type':type(error).__name__,'reason':str(error)}
            report['status']='INCOMPLETE'
            if not active_failure:
                raise
        finally:
            report['wall_seconds']=time.monotonic()-started;write(out/'commissioning-report.json',report)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-synthetic-cuda',action='store_true')
    parser.add_argument('--gpu-uuid',default=GPU_UUID)
    parser.add_argument('--acknowledge-shared-gpu-permission',action='store_true',
                        help='Confirm existing permission for shared use of the reviewed GPU; not an exclusive allocation')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--worker',choices=STAGES,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.worker:
        if os.environ.get('LIDC3D_SYNTHETIC_COMMISSION')!='1':parser.error('Internal worker requires explicit parent execution')
        try:result=run_worker(args.worker,args.output,args.gpu_uuid);write(args.output/(args.worker+'-result.json'),result)
        except BaseException as error:
            write(args.output/(args.worker+'-failure.json'),{'type':type(error).__name__,'reason':str(error),'traceback':traceback.format_exc()});raise
    elif args.execute_synthetic_cuda:
        if args.output is None:parser.error('--output must name a new directory')
        execute(args.output,args.gpu_uuid,args.acknowledge_shared_gpu_permission)
    else:
        print(json.dumps(plan(args.gpu_uuid,args.acknowledge_shared_gpu_permission),indent=2))

if __name__=='__main__':main()
