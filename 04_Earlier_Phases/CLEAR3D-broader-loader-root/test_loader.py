import json, hashlib
import numpy as np
import pytest
from loader import get_split, candidate_rows, build_dataset

def dump(p,v): p.write_text(json.dumps(v))
def test_geometry_identity_and_pixel_support(tmp_path):
    d=tmp_path/'scan_1';d.mkdir();dump(d/'record.json',{'patient_id':'P','series_uid':'S'})
    a=[[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]];dump(d/'geometry.json',{'affine':a})
    image=np.ones((50,50,50),np.float32)*.7; valid=np.ones_like(image,bool);valid[24,24,24]=False
    np.save(d/'image.npy',image);np.save(d/'valid.npy',valid)
    row={'patient':'P','series':'S','scan':'scan_1','id':'fit:1','x':24.,'y':24.,'z':24.,'label':1}
    ds=build_dataset(tmp_path,[row]); p=ds[0]
    assert tuple(p['image'].shape)==(1,48,48,48)
    assert p['image'][0,24,24,24]==0 and p['image'][0,23,23,23]==pytest.approx(.7)
    assert ds.rows[0]['zyx']==[24.,24.,24.]
    with pytest.raises(ValueError,match='identity'): build_dataset(tmp_path,[{**row,'patient':'WRONG'}])
    with pytest.raises(ValueError,match='outside'): build_dataset(tmp_path,[{**row,'x':500}])

def test_split_and_exclusions(tmp_path):
    p=tmp_path/'split';dump(p,{'fit':['P','LIDC-IDRI-0138'],'dev':['Q']}); h=hashlib.sha256(p.read_bytes()).hexdigest();s=get_split(p,h)
    assert s=={'fit':{'P'},'dev':{'Q'}}
    rows=[{'patient':'P','label':1,'x':1,'y':2,'z':3},{'patient':'LIDC-IDRI-0138','label':0,'x':1,'y':2,'z':3}]
    q=tmp_path/'rows';q.write_text(''.join(json.dumps(x)+'\n' for x in rows));assert len(candidate_rows(q,'fit',s['fit']))==1
    rows[0]['patient']='Q';q.write_text(json.dumps(rows[0]));
    with pytest.raises(ValueError,match='outside'):candidate_rows(q,'fit',s['fit'])
    with pytest.raises(ValueError,match='hash'):get_split(p,'wrong')
