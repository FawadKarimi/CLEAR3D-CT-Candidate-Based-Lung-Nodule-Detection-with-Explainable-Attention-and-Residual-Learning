import hashlib,json,zipfile
from pathlib import Path
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
# Replay additive patch independently, admitting only new paths and exact content.
text=(R/'exact-additive.patch').read_text();parts=text.split('--- /dev/null\n')[1:];replayed=[]
for part in parts:
 lines=part.splitlines(keepends=True);assert lines[0].startswith('+++ b/');rel=lines[0][6:].strip()
 assert lines[1].startswith('@@ -0,0 ')
 assert all(x.startswith('+') for x in lines[2:])
 data=''.join(x[1:] for x in lines[2:]).encode();assert data==(R/rel).read_bytes();replayed.append(rel)
b=json.loads((R/'accepted/training-binding.json').read_text())
assert {p.relative_to(R/'accepted/source').as_posix():sha(p) for p in (R/'accepted/source').rglob('*') if p.is_file()}==b['sources']
(R/'patch-replay.json').write_text(json.dumps({'status':'PASS','files':replayed,'parent_files_unchanged':133},indent=2)+'\n')
skip={'cpu-transfer.zip','prepare_transfer.py'}
files=[p for p in R.rglob('*') if p.is_file() and p.name not in skip and p.name!='SHA256SUMS' and '__pycache__' not in p.parts and p.suffix!='.zip']
# Preserve nested historical checksum files verbatim as evidence.
files += [p for p in R.rglob('SHA256SUMS') if p.parent!=R]
files=sorted(set(files));(R/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(R).as_posix()+'\n' for p in files)+''.join(h+'  source/'+n+'\n' for n,h in sorted(b['sources'].items())),encoding='utf-8',newline='\n')
z=R.parent/(R.name+'-review.zip')
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as out:
 for p in files+[R/'SHA256SUMS']:out.write(p,p.relative_to(R).as_posix())
 # Exact parent runtime copy; the accepted/ snapshot remains unchanged.
 for p in (R/'accepted/source').rglob('*'):
  if p.is_file():out.write(p,'source/'+p.relative_to(R/'accepted/source').as_posix())
with zipfile.ZipFile(z) as q:
 assert q.testzip() is None
 for line in (R/'SHA256SUMS').read_text().splitlines():
  h,n=line.split('  ',1);assert hashlib.sha256(q.read(n)).hexdigest()==h
 for n,h in b['sources'].items():assert hashlib.sha256(q.read('source/'+n)).hexdigest()==h
v={'path':str(z),'bytes':z.stat().st_size,'sha256':sha(z),'members':len(q.namelist()),'checksum_verification':'PASS','parent_source_count':133}
(R.parent/(z.name+'.sha256')).write_text(v['sha256']+'  '+z.name+'\n');print(json.dumps(v))
