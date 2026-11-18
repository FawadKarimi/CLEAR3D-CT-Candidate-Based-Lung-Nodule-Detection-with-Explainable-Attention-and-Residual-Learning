import json,sys,time,collections,traceback
from pathlib import Path
import loader
base=Path('/experiment-storage/chaidir/fawad-dsai')
p=base/'CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-comparison-protocol-20261025/runs/phase-cf'
root=base/'CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared'
out=base/'CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-broader-loader-root/audit-001.json'
if out.exists(): raise RuntimeError('output exists')
t0=time.monotonic(); split=loader.get_split(p/'split.json'); report={'roles':{},'scope':'metadata/geometry audit; no model training; reference HOLD', 'split_sha256':loader.hash_file(p/'split.json')}
for role in ('fit','dev'):
    path=p/(role+'.jsonl'); before=loader.hash_file(path); rows=loader.candidate_rows(path,role,split[role]);groups=collections.defaultdict(list)
    for r in rows:groups[r['scan']].append(r)
    failures=[]; passed=0
    for scan,group in groups.items():
        try:
            ds=loader.build_dataset(root,group)
            if len(ds)!=len(group):raise ValueError('count mismatch')
            passed+=len(group)
        except Exception as exc:failures.append({'scan':scan,'rows':len(group),'error':repr(exc)})
    after=loader.hash_file(path)
    report['roles'][role]={'input_sha256':before,'input_stable':before==after,'rows':len(rows),'scans':len(groups),'passed_rows':passed,'failures':failures,'class_counts':dict(collections.Counter(str(r['label']) for r in rows))}
    print(role,'rows',len(rows),'passed',passed,'failure_scans',len(failures),flush=True)
report['seconds']=time.monotonic()-t0; report['pass']=all(not v['failures'] and v['input_stable'] and v['rows']==v['passed_rows'] for v in report['roles'].values())
with out.open('x') as f:json.dump(report,f,indent=2,sort_keys=True)
print('REPORT',out,'PASS',report['pass'],flush=True)
sys.exit(0 if report['pass'] else 1)
