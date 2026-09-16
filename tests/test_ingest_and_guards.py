from pathlib import Path
import json
import copy
import numpy as np
import pytest
import yaml
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage
from lidc3d.ingest import load_series, inventory
from lidc3d.references import parse_xml
from lidc3d.provenance import write_json, write_jsonl, sha256, verify_bundle, safe_output
from lidc3d.training import train, predict
from lidc3d.models import DetectorClassifier
import torch


def config():
    return yaml.safe_load((Path(__file__).parents[1]/"configs/reference.yaml").read_text())


def dicom_fixture(tmp_path, positions=(10,12,14)):
    paths=[]
    for i,z in enumerate(positions):
        meta=FileMetaDataset()
        meta.TransferSyntaxUID=ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID=CTImageStorage
        meta.MediaStorageSOPInstanceUID=f"1.2.826.0.1.3680043.10.999.{i+1}"
        path=tmp_path/f"{i}.dcm"
        ds=FileDataset(str(path),{},file_meta=meta,preamble=b'\0'*128)
        ds.SOPClassUID=CTImageStorage
        ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
        ds.PatientID="SYNTHETIC_ONLY"
        ds.StudyInstanceUID="1.2.826.0.1.3680043.10.999.100"
        ds.SeriesInstanceUID="1.2.826.0.1.3680043.10.999.101"
        ds.Modality="CT"
        ds.ImageType=['ORIGINAL','PRIMARY','AXIAL']
        ds.ImageOrientationPatient=[1,0,0,0,1,0]
        ds.ImagePositionPatient=[20,-10,z]
        ds.PixelSpacing=[.8,.7]
        ds.Rows=ds.Columns=12
        ds.PhotometricInterpretation="MONOCHROME2"
        ds.SamplesPerPixel=1
        ds.BitsAllocated=ds.BitsStored=16
        ds.HighBit=15
        ds.PixelRepresentation=1
        ds.RescaleSlope=2
        ds.RescaleIntercept=-1000
        pixels=np.full((12,12),100+i,np.int16)
        pixels[0,0]=-2000
        ds.PixelPaddingValue=-2000
        ds.PixelData=pixels.tobytes()
        ds.save_as(path,enforce_file_format=True)
        paths.append(str(path))
    return paths


def test_dicom_physical_sort_hu_and_padding(tmp_path):
    paths=dicom_fixture(tmp_path)
    v=load_series(list(reversed(paths)),config()['geometry'])
    assert v.hu[:,1,1].tolist()==[-800,-798,-796]
    assert not v.valid[:,0,0].any()
    assert np.allclose(v.affine[:3,0],[0,0,2])
    assert np.allclose(v.affine[:3,1],[0,.8,0])
    assert np.allclose(v.affine[:3,2],[.7,0,0])
    inv=inventory(tmp_path)
    assert len(inv['slices'])==3 and not inv['errors']


@pytest.mark.parametrize('positions',[(10,10,12),(10,12,18)])
def test_duplicate_positions_and_gaps_rejected(tmp_path,positions):
    paths=dicom_fixture(tmp_path,positions)
    with pytest.raises(ValueError):
        load_series(paths,config()['geometry'])


def test_xml_contour_holes_and_sop_linkage(tmp_path):
    v=load_series(dicom_fixture(tmp_path),config()['geometry'])
    sop=v.metadata['sop_uids'][1]
    def roi(points,included):
        edges=''.join(f'<edgeMap><xCoord>{x}</xCoord><yCoord>{y}</yCoord></edgeMap>' for x,y in points)
        return f'<roi><imageSOP_UID>{sop}</imageSOP_UID><imageZposition>12</imageZposition><inclusion>{included}</inclusion>{edges}</roi>'
    outer=roi([(2,2),(8,2),(8,8),(2,8)],'TRUE')
    hole=roi([(4,4),(6,4),(6,6),(4,6)],'FALSE')
    xml=f'<root xmlns="urn:test"><SeriesInstanceUid>{v.metadata["series_uid"]}</SeriesInstanceUid><readingSession><servicingRadiologistID>r1</servicingRadiologistID><unblindedReadNodule><noduleID>n1</noduleID>{outer}{hole}</unblindedReadNodule></readingSession></root>'
    path=tmp_path/'reference.xml';path.write_text(xml)
    marks=parse_xml(path,v)
    voxels=set(map(tuple,marks[0]['coords']))
    assert (1,2,2) in voxels and (1,5,5) not in voxels
    path.write_text(xml.replace(sop,'9.9.9'))
    with pytest.raises(ValueError,match='SOP'):
        parse_xml(path,v)


def test_output_guard_and_exclusive_write(tmp_path):
    raw=tmp_path/'raw'
    with pytest.raises(ValueError):
        safe_output(raw/'output',raw)
    p=tmp_path/'artifact.json'
    write_json(p,{'a':1})
    with pytest.raises(FileExistsError):
        write_json(p,{'a':2})


def make_bundle(tmp_path):
    cfg=config()
    cfg['model'].update(kind='cnn',width=4,patch_size=16)
    cfg['training'].update(max_epochs=2,warmup_epochs=1,microbatch=2,accumulation=2,amp=False)
    cfg['candidates']['threshold']=.01
    scans=[];candidates=[]
    for i,split in enumerate(['train','val']):
        image=tmp_path/f'image{i}.npy';valid=tmp_path/f'valid{i}.npy'
        np.save(image,np.random.default_rng(i).random((20,20,20)).astype(np.float32))
        np.save(valid,np.ones((20,20,20),bool))
        scans.append({'scan_id':f's{i}','patient_id':f'p{i}','pixel_fingerprint':f'synthetic-fingerprint-{i}',
            'split':split,'status':'ok','image_path':str(image),'valid_path':str(valid),
            'positive':[{'id':f'r{i}','center_lps':[10,10,10],'radius_mm':1}], 'ignored':[]})
        for j,center in enumerate([[10,10,10],[3,3,3],[17,17,17]]):
            candidates.append({'id':f'p{i}c{j}','scan_id':f's{i}','zyx':center,'center_lps':center})
    scan_path=tmp_path/'scans.jsonl';candidate_path=tmp_path/'candidates.jsonl';cfg_path=tmp_path/'config.json'
    write_jsonl(scan_path,scans);write_jsonl(candidate_path,candidates);write_json(cfg_path,cfg)
    bundle={'files':{str(p):sha256(p) for p in [scan_path,candidate_path,cfg_path]},
        'scans':str(scan_path),'candidates':str(candidate_path),'resolved_config':str(cfg_path),
        'reference_audit_approved':True,'development_config_frozen':True}
    return bundle,cfg,scans,candidates


def test_hash_tamper_and_patient_leakage(tmp_path):
    bundle,cfg,scans,candidates=make_bundle(tmp_path)
    assert len(verify_bundle(bundle))==2
    Path(bundle['scans']).write_text('changed')
    with pytest.raises(ValueError,match='changed'):
        verify_bundle(bundle)


def test_training_epoch_resume_and_atomic_scan_failure(tmp_path):
    bundle,cfg,scans,candidates=make_bundle(tmp_path)
    output=tmp_path/'run'
    model=train(bundle,cfg,output,17,'cpu')
    assert (output/'selection.json').exists()
    state=torch.load(output/'last.pt',weights_only=False)
    assert state['epoch']==1 and 'loader_rng' in state
    # Same completed schedule resumes as a no-op while preserving the best.
    resumed=tmp_path/'resumed'
    train(bundle,cfg,resumed,17,'cpu',output/'last.pt')
    assert sha256(output/'best.pt')==sha256(resumed/'best.pt')
    broken=copy.deepcopy(scans[1]);broken['image_path']=str(tmp_path/'missing.npy')
    predictions,status=predict(model,[broken],[r for r in candidates if r['scan_id']=='s1'],16,torch.device('cpu'))
    assert predictions==[] and status[0]['status']=='failed'


def test_test_bundle_and_changed_preprocessing_rejected(tmp_path):
    bundle,cfg,scans,candidates=make_bundle(tmp_path)
    changed=copy.deepcopy(cfg);changed['preprocess']['window_hu']=[-1000,200]
    with pytest.raises(ValueError,match='prepared'):
        train(bundle,changed,tmp_path/'bad_run',17,'cpu')
