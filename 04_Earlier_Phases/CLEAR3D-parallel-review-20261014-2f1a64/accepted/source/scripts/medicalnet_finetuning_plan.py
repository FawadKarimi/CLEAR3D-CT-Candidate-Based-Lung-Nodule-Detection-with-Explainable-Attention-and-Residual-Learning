"""Nonexecuting proposal. Research execution remains blocked by approval guards."""
import json
import sys

if len(sys.argv) != 1:
    raise SystemExit('Plan-only entry point: no execution or approval bypass')
print(json.dumps({'status': 'PLAN_ONLY', 'reference_release': 'HOLD',
                  'backbones': [18, 10], 'precision': 'FP32', 'epochs': 3,
                  'gpu_budget_seconds_total': 1800, 'per_backbone_seconds': 900,
                  'training_draws_per_epoch': {'positive': 240, 'negative': 720},
                  'production_reference_guards': 'UNCHANGED',
                  'scoped_execution_adapter': 'REQUIRES_SEPARATE_REVIEW',
                  'torch_imported': 'torch' in sys.modules}))
