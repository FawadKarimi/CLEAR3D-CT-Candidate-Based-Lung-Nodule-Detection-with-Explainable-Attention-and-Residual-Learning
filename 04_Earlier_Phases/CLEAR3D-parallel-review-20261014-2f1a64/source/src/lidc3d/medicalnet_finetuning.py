"""Isolated multi-epoch transfer integration; does not grant reference approval."""
from pathlib import Path
from collections.abc import Sequence
import hashlib
from . import medicalnet_contract as contract


def validate(config, seed):
    spec = config['medicalnet']
    contract.require(set(spec) == {'depth', 'checkpoint', 'checkpoint_sha256',
                                  'provenance', 'provenance_sha256'}, 'Transfer fields changed')
    depth = spec['depth']
    contract.require(type(depth) is int and depth in (18, 10), 'Unreviewed backbone')
    contract.require(seed == 17 and config['training']['amp'] is False, 'Seed/FP32 policy')
    contract.require(config['model']['kind'] == 'hybrid' and
                     config['model']['width'] == 16 and config['model']['patch_size'] == 48,
                     'Unreviewed model configuration')
    path = Path(spec['checkpoint']).absolute()
    contract.require(path.is_file() and not any(p.is_symlink() for p in [path, *path.parents]),
                     'Unsafe checkpoint path')
    contract.require(spec['checkpoint_sha256'] == contract.WEIGHTS[depth] == contract.digest(path),
                     'Checkpoint mismatch')
    provenance = contract.read(spec['provenance'], spec['provenance_sha256'])
    contract.provenance(provenance)
    # no_resume is an execution protocol field of the earlier one-pass pilot,
    # not a mathematical operator change. This version explicitly replaces it.
    transfer = {**contract.transfer(depth), 'no_resume': False,
                'execution_protocol': 'medicalnet-epoch-boundary-v1'}
    return {'transfer': transfer, 'provenance_sha256': spec['provenance_sha256'],
            'official_archive_sha256': provenance['archive_sha256'],
            'scope': 'production approval guards remain mandatory'}


def build(config, seed):
    validate(config, seed)
    from .medicalnet_adapter import build_transfer
    return build_transfer(config['model'], config['medicalnet']['depth'],
                          config['medicalnet']['checkpoint'], seed)


class TrainingSample(Sequence):
    """Integer-only epoch view; no proposal payload pool or validation sampling.

    Positives cycle with explicit replacement. Negatives are sampled without
    replacement. Repeated source IDs keep the accepted same-ID augmentation.
    This changes the empirical objective and is not an unbiased risk estimator.
    """
    validated_index = True

    def __init__(self, rows, spec, seed):
        contract.require(getattr(rows, 'role', None) == 'train', 'Training-only sampling')
        contract.require(set(spec) == {'policy', 'positive_draws', 'negative_draws',
                                      'max_source_rows', 'max_epoch_rows'}, 'Sampling fields')
        contract.require(spec['policy'] == 'positive-cycle-negative-hash-v1', 'Sampling policy')
        contract.require(all(type(spec[k]) is int and spec[k] > 0 for k in spec if k != 'policy'),
                         'Finite positive integer sampling limits required')
        contract.require(spec['max_source_rows'] <= 283200 and spec['max_epoch_rows'] <= 65536,
                         'Sampling admission limit')
        contract.require(len(rows) <= spec['max_source_rows'] and
                         spec['positive_draws'] + spec['negative_draws'] <= spec['max_epoch_rows'],
                         'Sampling admission before allocation')
        self.rows, self.spec, self.seed = rows, dict(spec), seed
        self.classes = {0: [], 1: []}
        seen = set()
        for i, row in enumerate(rows):
            contract.require(row['label'] in (0, 1), 'Ignored row in training sample')
            contract.require(row['id'] not in seen, 'Duplicate source identity')
            seen.add(row['id']); self.classes[row['label']].append(i)
        contract.require(self.classes[1] and len(self.classes[0]) >= spec['negative_draws'],
                         'Insufficient class coverage; no substitution')
        self.indices = []
        self.set_epoch(0)

    def set_epoch(self, epoch):
        contract.require(type(epoch) is int and epoch >= 0, 'Invalid sampling epoch')
        def rank(i):
            return hashlib.sha256(f'{self.seed}:{epoch}:{i}'.encode()).digest(), i
        positives = sorted(self.classes[1], key=rank)
        negatives = sorted(self.classes[0], key=rank)[:self.spec['negative_draws']]
        self.indices = [positives[i % len(positives)] for i in range(self.spec['positive_draws'])] + negatives
        self.receipt = {'epoch': epoch, 'policy': self.spec, 'seed': self.seed,
                        'positive_draws': self.spec['positive_draws'],
                        'negative_draws': self.spec['negative_draws'],
                        'unique_positive_sources': min(len(positives), self.spec['positive_draws']),
                        'source_index_sha256': contract.canonical(self.indices)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if not isinstance(i, int):
            raise TypeError('Integer sample index required')
        return self.rows[self.indices[i]]
