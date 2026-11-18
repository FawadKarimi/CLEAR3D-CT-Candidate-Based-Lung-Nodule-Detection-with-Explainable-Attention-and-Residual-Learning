"""Stable bounded top-k matching (-response, id) stable-sort semantics."""
import hashlib,heapq,json,math
from pathlib import Path
class Worst:
    __slots__=('key','row')
    def __init__(self,key,row):self.key,self.row=key,row
    def __lt__(self,other):return self.key>other.key

def top_rows(source,sid,threshold,cap):
    if type(cap) is not int or not 0<=cap<=400 or not math.isfinite(threshold):raise ValueError('Selection bounds')
    heap=[]
    for ordinal,row in enumerate(source):
        if row.get('scan_id')!=sid:continue
        value=row['log_response']
        if not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('Nonfinite response')
        if not isinstance(row.get('id'),str):raise ValueError('Invalid candidate ID')
        if value<threshold or cap==0:continue
        item=Worst((-value,row['id'],ordinal),row)
        if len(heap)<cap:heapq.heappush(heap,item)
        elif item.key<heap[0].key:heapq.heapreplace(heap,item)
    return [v.row for v in sorted(heap,key=lambda v:v.key)]

def verified_stream(path,sid,expected,count):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('Unsafe proposal path')
    digest=hashlib.sha256();n=0;size=0
    with path.open('rb') as f:
        while True:
            raw=f.readline(16385)
            if not raw:break
            if len(raw)>16384 or not raw.strip() or n>=count:raise ValueError('Proposal admission limit')
            digest.update(raw);size+=len(raw);row=json.loads(raw)
            if row.get('scan_id')!=sid or row.get('id')!=f'{sid}:candidate:{n:06d}':raise ValueError('Proposal ordinal mismatch')
            n+=1;yield row
    if n!=count or size!=expected['bytes'] or digest.hexdigest()!=expected['sha256']:raise ValueError('Incomplete/changed proposal stream')
