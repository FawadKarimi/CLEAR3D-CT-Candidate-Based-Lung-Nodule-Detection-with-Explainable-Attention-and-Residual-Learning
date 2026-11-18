"""Reconstruct UI-read evidence; accept only exact server-reported hashes."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;b=json.loads((R/'accepted/training-binding.json').read_text());s='/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64'
binding={'source_files':133,'pins':b['pins'],'interpreter':b['interpreter'],'interpreter_sha256':b['interpreter_sha256']}
commands=[[b['interpreter'],'-B',s+'/C-tests/test_model_path.py','--source',s+'/source/src'],[b['interpreter'],'-B',s+'/C-tests/test_fixed_batch.py','--source',s+'/source/src','--runner',s+'/D-plan/fixed_batch.py','--checkpoints','enabled']]
result={'before':binding,'runs':[{'argv':cmd,'exit':0,'seconds':t,'cumulative_child_peak_rss_kib':1152456} for cmd,t in zip(commands,[5.931520560756326,3.5752566792070866])],'cuda_visible_devices':'','after':binding,'exit':0}
text=json.dumps(result,indent=2)
assert hashlib.sha256(text.encode()).hexdigest()=='ca89fda58deae4668c7140cfa5be28ba4411261ec5dd15c9c8e651c5b89e8be9'
out=R/'cpu-results';out.mkdir(exist_ok=True);(out/'results.json').write_bytes(text.encode())
t='''test_adoption_draft_missing_holds_and_binding_refusal (__main__.FixedBatchTests) ... ok
test_choose_first_unique_order_and_reject_drift (__main__.FixedBatchTests) ... ok
test_default_nonexecuting_no_torch (__main__.FixedBatchTests) ... ok
test_generated_24_updates_denominator_checkpoint_and_diagnostics (__main__.FixedBatchTests) ... ok

----------------------------------------------------------------------
Ran 4 tests in 2.987s

OK
{"generated_fixture": true, "checkpoints_tested": true, "torch": "2.7.1+cu118"}
'''
assert hashlib.sha256(t.encode()).hexdigest()=='d121b3b5114cc937b679d6f46d43b3e41d576e3e557927b804ef14ce56975abe'
(out/'test-1.log').write_bytes(t.encode())
t='''test_balanced_weighted_constant_predictor_stationary_point (__main__.ModelPathTests) ... ok
test_corrected_shortcut_keeps_input_gradient (__main__.ModelPathTests) ... ok
test_observers_preserve_exact_logits_gradients_update_and_rng (__main__.ModelPathTests) ... ok
test_pooling_does_not_cancel_channel_specific_signal (__main__.ModelPathTests) ... ok
test_production_shape_input_variation_and_all_families_backward (__main__.ModelPathTests) ... /experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64/source/src/lidc3d/medicalnet_vendor.py:173: FutureWarning: `nn.init.kaiming_normal` is now deprecated in favor of `nn.init.kaiming_normal_`.
  m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
ok

----------------------------------------------------------------------
Ran 5 tests in 4.109s

OK
{"python": "3.10.17 (main, Apr  9 2025, 08:54:15) [GCC 9.4.0]", "torch": "2.7.1+cu118", "source": "/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64/source/src", "cuda_visible_devices": "", "torch_threads": 1, "generated_random_weights_only": true, "not_pretrained_or_cuda_evidence": true}
GENERATED_GRADIENT_EVIDENCE {"bridges": 0.15021257241212538, "classifier": 2.4974205283532176, "encoder": 0.5270521673915127, "fusion": 2.400405705364641, "transformer": 1.9814929984353304}
'''
assert hashlib.sha256(t.encode()).hexdigest()=='e35195debf550478e49bfc7b30e70668124155d1b4a429e1671cdc4f952dc44c'
(out/'test-0.log').write_bytes(t.encode())
print('All three UI-read files reproduce exact server SHA256')
