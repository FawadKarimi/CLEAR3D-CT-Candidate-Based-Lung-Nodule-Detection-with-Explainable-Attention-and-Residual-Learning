"""CPU-only integration tests for binding, current split and adoption gates.

Fake modules replace project imports; this test does not attest the real
watchdog, provenance verifier, checkpoint, sidecar or GPU runtime.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


HERE = Path(__file__).parent
HASH = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        d = self.root / 'D-fit'
        d.mkdir()
        for name in ('fixed_batch_fit.py', 'fit_selection.py'):
            shutil.copyfile(HERE / name, d / name)
        (self.root / 'source' / 'scripts').mkdir(parents=True)
        self.spec = importlib.util.spec_from_file_location('runner_fixture', d / 'fixed_batch_fit.py')
        self.runner = importlib.util.module_from_spec(self.spec)
        self.spec.loader.exec_module(self.runner)
        self.split = {'fit':['p1','LIDC-IDRI-0138'], 'dev':['p2'], 'confirm':['p3'], 'excluded_reviewed':[]}
        self.plan = {'indices': list(range(960))}
        self.selection = {'entries': [{'scan': {'scan_id': 'a', 'patient_id': 'p1'}},
                                      {'scan': {'scan_id': 'b', 'patient_id': 'p2'}},
                                      {'scan': {'scan_id': 'c', 'patient_id': 'LIDC-IDRI-0138'}}]}
        self.rows = {i: {'id':str(i), 'scan_id':'a', 'label':int(i < 240)} for i in range(960)}
        self.rows[0]['scan_id'] = 'b'
        self.rows[240]['scan_id'] = 'c'
        self.expected = [1,2,3,4] + list(range(241,253))
        self.inputs = {}
        for name, value in [('preparation', {}), ('sidecar', {}), ('parent_selection', self.selection),
                            ('label_selection', {}), ('draw0', self.plan)]:
            path = self.write(name+'.json', value)
            self.inputs[name] = {'path':str(path), 'bytes':path.stat().st_size, 'sha256':HASH(path)}
        split_path = self.write('current-split.json', self.split)
        checkpoint = self.write('checkpoint.json', {'fixture':True})
        provenance = self.write('provenance.json', {'fixture':True})
        config = self.write('configuration.json', {'medicalnet': {'depth':18, 'checkpoint':str(checkpoint),
                            'provenance':str(provenance), 'provenance_sha256':HASH(provenance)}})
        self.old = {'diagnostic_data': {'inputs':self.inputs}}
        self.binding = {'root':str(self.root), 'policy':self.runner.POLICY, 'limits':self.runner.LIMITS,
                        'new_sources':{f'D-fit/{n}':HASH(d / n) for n in ('fixed_batch_fit.py','fit_selection.py')},
                        'parent_binding':'fake', 'parent_binding_sha256':'fake',
                        'configuration': {'path':str(config), 'sha256':HASH(config)},
                        'current_split': {'path':str(split_path), 'sha256':HASH(split_path), 'bytes':split_path.stat().st_size},
                        'indices':self.expected}
        self.binding_path = self.write('binding.json', self.binding)
        disposition = {'status':'ADOPTED', 'scope':self.runner.SCOPE, 'binding_sha256':HASH(self.binding_path),
                       'policy':self.runner.POLICY, 'limits':self.runner.LIMITS,
                       'reference_release':'HOLD', 'cap4000_policy':'NOT_ADOPTED',
                       'supplied_by':'SYNTHETIC', 'instruction':'SYNTHETIC', 'date':'SYNTHETIC',
                       'holds':['reference_origin','excluded_reference_outliers','geometry_and_source_scope',
                                'protocol_evaluation_disclosure','prior_test_access','independent_reference_review']}
        for key in ('label_guided','prior_exposure_uncertain','pretraining_overlap_uncertain',
                    'sampled_monitoring','no_scientific_approval','train_only'):
            disposition[key] = True
        self.disposition_path = self.write('disposition.json', disposition)
        self.request = {'binding':str(self.binding_path), 'binding_sha256':HASH(self.binding_path),
                        'disposition':str(self.disposition_path), 'disposition_sha256':HASH(self.disposition_path),
                        'acknowledge_shared_gpu_permission':True, 'acknowledge_sampled_risk':True}
        self.fake_watchdog = types.ModuleType('user_watchdog')
        self.fake_watchdog.__file__ = str(self.root / 'source' / 'scripts' / 'user_watchdog.py')
        self.fake_watchdog.bindings = lambda *_: self.old
        self.parent = types.ModuleType('lidc3d.diagnostic_training')
        self.parent.check_config = lambda _: None
        self.parent.base = types.SimpleNamespace(WEIGHTS={18:HASH(checkpoint)}, provenance=lambda _:None)
        self.labels = types.ModuleType('lidc3d.cap4000_labels')
        fixture = self
        class Rows:
            def __init__(self, *_): pass
            def __getitem__(self, key): return fixture.rows[key]
            def close(self): pass
        self.labels.LabelRows = Rows
        self.lidc = types.ModuleType('lidc3d')
        self.lidc.__path__ = []
        self.lidc.diagnostic_training = self.parent

    def tearDown(self): self.temp.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def gate(self):
        modules = {'user_watchdog':self.fake_watchdog, 'lidc3d':self.lidc,
                   'lidc3d.diagnostic_training':self.parent, 'lidc3d.cap4000_labels':self.labels}
        # Ensure fit_selection is imported from the copied, hash-bound path.
        oldpath = list(sys.path)
        oldselector = sys.modules.pop('fit_selection', None)
        try:
            sys.path.insert(0, str(self.root / 'D-fit'))
            with patch.dict(sys.modules, modules): return self.runner.gates(self.request)
        finally:
            sys.path[:] = oldpath
            sys.modules.pop('fit_selection', None)
            if oldselector is not None: sys.modules['fit_selection'] = oldselector

    def test_valid_fake_binding_selects_only_current_fit(self):
        binding, _, _, _ = self.gate()
        self.assertEqual(binding['indices'], self.expected)

    def test_tampered_split_or_selector_fails(self):
        split = self.root / 'current-split.json'
        split.write_text(split.read_text() + ' ')
        with self.assertRaises(ValueError): self.gate()
        split.write_text(json.dumps(self.split))
        selector = self.root / 'D-fit' / 'fit_selection.py'
        selector.write_text(selector.read_text() + '\n# mutation\n')
        with self.assertRaises(ValueError): self.gate()

    def test_sidecar_drift_and_missing_shared_acknowledgement_fail(self):
        self.rows[1]['scan_id'] = 'b'
        with self.assertRaises(ValueError): self.gate()
        self.rows[1]['scan_id'] = 'a'
        self.request['acknowledge_shared_gpu_permission'] = False
        with self.assertRaises(ValueError): self.gate()

    def test_disposition_and_binding_mismatch_fail(self):
        self.request['disposition_sha256'] = '0'*64
        with self.assertRaises(ValueError): self.gate()
        self.request['disposition_sha256'] = HASH(self.disposition_path)
        self.request['binding_sha256'] = '0'*64
        with self.assertRaises(ValueError): self.gate()


if __name__ == '__main__': unittest.main()
