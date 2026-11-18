import csv,json,collections,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import loader
base=Path('/experiment-storage/chaidir/fawad-dsai'); p=base/'CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-comparison-protocol-20261025/runs/phase-cf'; split=loader.get_split(p/'split.json')
refs=base/'data/reference/luna16-zenodo-3723295/resolved-reference-v1/annotations_excluded.csv'
assert loader.hash_file(refs)=='73a5c139257294fe45a68620ed1b2cf383ba8b22af1501d2f498abbd6b802c25'
regions=collections.defaultdict(list)
with refs.open(newline='') as f:
 for r in csv.DictReader(f):
  d=float(r['diameter_mm']); regions[r['seriesuid']].append(([float(r['coordX']),float(r['coordY']),float(r['coordZ'])],5. if d<0 else d/2))
report={}
for role in ('fit','dev'):
 rows=loader.candidate_rows(p/(role+'.jsonl'),role,split[role]); groups=collections.defaultdict(list)
 for r in rows:
  if r['label']==0:groups[r['series']].append(r)
 matches=0; scans=0
 for uid,rr in groups.items():
  regs=regions.get(uid,[])
  if not regs:continue
  xyz=np.array([[r['x'],r['y'],r['z']] for r in rr]); tree=cKDTree(xyz); chosen=set()
  for center,radius in regs:
   # Strict sphere convention; query may include the boundary.
   for i in tree.query_ball_point(center,radius):
    if np.linalg.norm(xyz[i]-center)<radius:chosen.add(i)
  if chosen: scans+=1; matches+=len(chosen)
 report[role]={'negative_rows_in_excluded_region':matches,'affected_series':scans,'negative_rows':sum(len(x) for x in groups.values())}
print(json.dumps(report,sort_keys=True),flush=True)
out=base/'CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-broader-loader-root/excluded-overlap-001.json'
with out.open('x') as f:json.dump({'counts':report,'excluded_sha256':loader.hash_file(refs),'convention':'strict radius, 5 mm for negative diameter; exploratory, no labels altered'},f,indent=2)
