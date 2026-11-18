"""Plan-only by default; reference access requires a separately adopted binding."""
import argparse,hashlib,json,os,sys,time,shutil,resource,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in [ROOT/'src',ROOT/'scripts']:sys.path.insert(0,str(p))

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--binding');p.add_argument('--binding-sha256');p.add_argument('--disposition');p.add_argument('--disposition-sha256');a=p.parse_args()
 if not a.execute:
  print(json.dumps({'status':'PLAN_ONLY','scope':'cap4000-fixed12-reference-selection-v1','reference_release':'HOLD','cap4000_policy':'NOT_ADOPTED','research_inputs_opened':False}));return 0
 from lidc3d import cap4000_labels as m
 import importlib.metadata
 m.require(Path(m.__file__).resolve()==ROOT/'src/lidc3d/cap4000_labels.py','Import location')
 m.require(all([a.binding,a.binding_sha256,a.disposition,a.disposition_sha256]),'Missing binding/adoption')
 m.require(m.hash_file(a.binding)==a.binding_sha256,'Binding hash');b=m.bounded_json(a.binding)
 m.require(m.hash_file(a.disposition)==a.disposition_sha256,'Adoption hash');m.gate(m.bounded_json(a.disposition),a.binding_sha256,b['inputs'])
 m.require(str(ROOT)==b['source_root'] and b['limits']==m.LIMITS,'Source/limit identity')
 files={x.relative_to(ROOT).as_posix() for x in ROOT.rglob('*') if x.is_file()}
 m.require(files==set(b['sources']),'Exact source inventory')
 for rel,h in b['sources'].items():m.require(m.hash_file(ROOT/rel)==h,'Source hash '+rel)
 m.require(str(Path(sys.executable).absolute())==b['interpreter'] and hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()==b['interpreter_sha256'],'Interpreter identity')
 for name,version in b['pins']:m.require(importlib.metadata.version(name)==version,'Environment pin '+name)
 m.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not os.environ.get('PYTHONPATH') and not os.environ.get('PYTHONHOME') and os.environ.get('PYTHONNOUSERSITE')=='1','Child environment')
 out=m.safe(b['output']);m.require(out.parent.is_dir() and not out.exists(),'Fresh output/real parent required');m.require(shutil.disk_usage(out.parent).free>=m.LIMITS['minimum_disk'],'Disk headroom')
 available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
 m.require(available>=m.LIMITS['minimum_available_ram'],'RAM headroom')
 out.mkdir();start=time.monotonic();last=-1
 def tick():
  nonlocal last
  elapsed=time.monotonic()-start;rss=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
  if elapsed-last>=1:
   with (out/'resources.jsonl').open('a') as f:f.write(json.dumps({'elapsed':elapsed,'rss_bytes':rss,'pid':os.getpid()})+'\n');f.flush();os.fsync(f.fileno())
   last=elapsed
  m.require(elapsed<=m.LIMITS['seconds'] and rss<=m.LIMITS['rss_bytes'],'Sampled resource refusal')
 try:
  m.save(out/'invocation.json',{'argv':sys.argv,'binding_sha256':a.binding_sha256,'adoption_sha256':a.disposition_sha256,'pid':os.getpid(),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')})
  for key,item in b['inputs'].items():
   m.require(m.safe(item['path']).stat().st_size==item['bytes'],'Input byte binding '+key)
   m.require(m.hash_file(item['path'],tick)==item['sha256'],'Input hash '+key)
  selection=m.bounded_json(b['inputs']['parent_selection']['path'])
  pool=json.loads((ROOT/'src/lidc3d/mixed_contract.json').read_text())['pool']
  result=m.build(b['inputs']['index']['path'],selection,pool,out,tick)
  sidehash=m.hash_file(out/'labels.sqlite',tick);rows=m.LabelRows(out/'labels.sqlite',sidehash,tick)
  try:
   spec={'policy':'positive-cycle-negative-hash-v1','positive_draws':240,'negative_draws':720,'max_source_rows':283200,'max_epoch_rows':65536};sample=m.DiskTrainingSample(rows,spec,17,tick)
   for epoch in range(3):sample.set_epoch(epoch);tick();m.save(out/f'epoch-{epoch}-draws.json',{'indices':sample.indices,'receipt':sample.receipt})
  finally:rows.close()
  for key,item in b['inputs'].items():m.require(m.hash_file(item['path'],tick)==item['sha256'],'Post-input binding '+key)
  for rel,h in b['sources'].items():m.require(m.hash_file(ROOT/rel,tick)==h,'Post-source binding '+rel)
  tick();m.save(out/'receipt.json',{'status':'COMPLETE_EXPLORATORY','selection_sha256':m.hash_file(out/'selection.json'),'sidecar_sha256':sidehash,'validation_file_sha256':m.hash_file(out/'validation-unchanged.json'),'input_bindings':b['inputs'],'source_binding_sha256':a.binding_sha256,'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'exit_status':0,'training_ready':False,'reference_release':'HOLD','cap4000_policy':'NOT_ADOPTED'});return 0
 except BaseException as ex:
  m.save(out/'launcher-failure.json',{'status':'FAILED','error':repr(ex),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-start,'exit_status':1,'no_retry':True});return 1

if __name__=='__main__':raise SystemExit(main())
