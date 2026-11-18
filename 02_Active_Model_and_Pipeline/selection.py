"""Reproducible exploratory fit/dev selection; not a released test protocol."""
import collections,csv,json,random,hashlib
from pathlib import Path
import loader
BASE=Path('/experiment-storage/chaidir/fawad-dsai')
P=BASE/'CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-comparison-protocol-20261025/runs/phase-cf'
REF=BASE/'data/reference/luna16-zenodo-3723295/resolved-reference-v1/annotations_excluded.csv'
OUT=BASE/'CLEAR3D_CT project/02_Active_Model_and_Pipeline/selection-001.json'
EXPECTED='73a5c139257294fe45a68620ed1b2cf383ba8b22af1501d2f498abbd6b802c25'

def select(rows,seed,neg_per_pos=3):
 pos=[r for r in rows if r['label']==1]; neg=[r for r in rows if r['label']==0]
 if not pos or len(neg)<len(pos)*neg_per_pos:raise ValueError('insufficient class coverage')
 rng=random.Random(seed); chosen=pos+rng.sample(neg,len(pos)*neg_per_pos); rng.shuffle(chosen)
 if len({r['id'] for r in chosen})!=len(chosen):raise ValueError('duplicate identity')
 return chosen

def main():
 if OUT.exists():raise ValueError('output already exists')
 split=loader.get_split(P/'split.json');assert loader.hash_file(REF)==EXPECTED
 refs=collections.defaultdict(list)
 with REF.open(newline='') as f:
  for r in csv.DictReader(f):
   d=float(r['diameter_mm']); refs[r['seriesuid']].append(([float(r['coordX']),float(r['coordY']),float(r['coordZ'])],5. if d<0 else d/2))
 result={'scope':'exploratory, sampled candidate classification; no untouched test or FROC', 'reference_release':'HOLD','seed':17,'ratio':'all positives and 3 sampled negatives per positive','source_sha256':{'split':loader.hash_file(P/'split.json'),'excluded':loader.hash_file(REF)}}
 for role in ('fit','dev'):
  src=P/(role+'.jsonl'); before=loader.hash_file(src); rows=loader.candidate_rows(src,role,split[role]); rows,removed=loader.filter_excluded(rows,refs)
  picked=select(rows,17 if role=='fit' else 18)
  if loader.hash_file(src)!=before:raise ValueError('input changed')
  result[role]={'source_sha256':before,'removed_excluded':removed,'candidate_population':len(rows),'selected':picked,'counts':dict(collections.Counter(str(r['label']) for r in picked)),'patients':len({r['patient'] for r in picked})}
 if set(r['patient'] for r in result['fit']['selected'])&set(r['patient'] for r in result['dev']['selected']):raise ValueError('patient overlap')
 with OUT.open('x') as f:json.dump(result,f,sort_keys=True,allow_nan=False);f.write('\n')
 print('SELECTION_SHA256',loader.hash_file(OUT),flush=True)
 print({r:{k:result[r][k] for k in ('candidate_population','counts','patients','removed_excluded')} for r in ('fit','dev')},flush=True)
if __name__=='__main__':main()
