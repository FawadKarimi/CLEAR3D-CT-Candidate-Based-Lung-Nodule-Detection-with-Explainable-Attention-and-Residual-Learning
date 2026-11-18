"""FP32 observations around the unchanged weighted accumulation implementation.

No alternate objective, gradient repair, skip policy, clipping or update dispatch.
The production entry requires exactly the reviewed 768-row scan-major pass.
"""
import copy
import math
import torch
from . import optimization as core
from .positive_optimization import parameter_hash, optimizer_steps


def require(ok, message):
    if not ok:
        raise ValueError(message)


def expected_labels():
    from .mixed_engineering import CONTRACT
    return [(s['scan_id'], label) for s in CONTRACT['pool'] if s['role'] == 'train'
            for label in [1] * s['planned_positive'] + [0] * s['planned_negative']]


def _check_rows(rows):
    require(len(rows) == 768 and [(r['scan_id'], r['label']) for r in rows] == expected_labels(),
            'Mixed scan-major identities/classes changed')
    require(len({(r['scan_id'], r['id']) for r in rows}) == 768 and
            len({(r['scan_id'], r['source_ordinal']) for r in rows}) == 768,
            'Duplicate mixed row consumption')
    for r in rows:
        require(type(r['source_ordinal']) is int and r['source_ordinal'] >= 0 and
                r['id'] == r['scan_id'] + ':candidate:' + format(r['source_ordinal'], '06d'),
                'Mixed ordinal identity changed')


def coverage_summary(events):
    boundaries = [e for e in events if e.get('kind') == 'accumulation_boundary']
    require(len(boundaries) == 48, 'Exactly 48 mixed boundaries required')
    consumed = []
    mixed = negative = denominator = 0
    for i, e in enumerate(boundaries):
        require(e['status'] == 'UPDATED' and e['optimizer_updates'] == 1 and e['amp'] is False
                and e['scale_before'] == e['scale_after'] == 1 and e['scaler_update_calls'] == 1
                and e['epoch'] == 0 and e['last_microbatch'] == 8 * i + 7
                and e['incomplete_group'] is False, 'Mixed update dispatch changed')
        c = core.validate_update_state(e['counters'])
        require(c['boundaries'] == c['optimizer_updates'] == c['scaler_update_calls'] == i + 1
                and c['skipped_updates'] == c['consecutive_skips'] == c['scale_changes'] == 0,
                'Mixed cumulative counters changed')
        micro, notes, proof, precision = e['microbatches'], e['class_microbatches'], e['optimizer_evidence'], e['fp32_evidence']
        require(len(micro) == len(notes) == 8 and [m['step'] for m in micro] == list(range(8*i,8*i+8)),
                'Mixed microbatch sequence changed')
        for m, n in zip(micro, notes):
            require(m['samples'] == len(n['rows']) == 2 and m['step'] == n['step']
                    and m['scaled_loss_finite'] and math.isfinite(m['weighted_loss_sum'])
                    and m['scale'] == 1, 'Mixed loss/microbatch evidence incomplete')
            labels = [r['label'] for r in n['rows']]
            require(all(x in (0,1) for x in labels) and n['positive'] == labels.count(1)
                    and n['negative'] == labels.count(0) and m['weight_sum'] == 3*n['positive']+n['negative']
                    and math.isfinite(n['positive_weighted_loss_sum'])
                    and math.isfinite(n['negative_weighted_loss_sum']), 'Mixed class loss evidence incomplete')
            consumed.extend(n['rows'])
        counts = {k:sum(n[k] for n in notes) for k in ('positive','negative')}
        require(e['class_counts'] == counts and e['denominator'] == 3*counts['positive']+counts['negative'],
                'Mixed denominator/count evidence changed')
        require(counts['negative'] > 0, 'Unexpected positive-only boundary')
        mixed += int(counts['positive'] > 0)
        negative += int(counts['positive'] == 0)
        denominator += e['denominator']
        require(math.isfinite(e['norm_before_clip']) and e['norm_before_clip'] > 0,
                'No finite nonzero normalized gradient norm')
        for key in ('scaled_gradients','unscaled_gradients','normalized_gradients'):
            require(e[key]['gradient_tensors'] > 0 and not e[key]['nonfinite'], 'Nonfinite mixed gradients')
        require(precision['model_dtype'] == 'float32' and precision['input_dtypes'] == ['float32']*8
                and precision['output_dtypes'] == ['float32']*8 and precision['scaler_enabled'] is False
                and precision['post_clip_finite'] is True and math.isfinite(precision['post_clip_norm'])
                and precision['post_clip_norm'] > 0, 'FP32 boundary evidence incomplete')
        names = proof['gradient_parameter_names']
        require(names and proof['parameters_changed'] is True and proof['optimizer_step_advanced'] is True
                and proof['parameter_sha256_before'] != proof['parameter_sha256_after']
                and all(proof['steps_before'][n] == i and proof['steps_after'][n] == i+1 for n in names),
                'AdamW advancement/parameter evidence incomplete')
    _check_rows(consumed)
    require(mixed == 9 and negative == 39 and denominator == 798, 'Mixed boundary class totals changed')
    return {'status':'PASS_MIXED_FP32_ENGINEERING','actual_updates':48,'skipped_updates':0,
            'rows':768,'positive':15,'negative':753,'mixed_boundaries':9,'negative_only_boundaries':39,
            'denominator':798,'not_amp_coverage':True,'not_model_quality':True}


def train_mixed_epoch(model, loader, optimizer, scaler, weights, device, amp,
                      accumulation, max_norm, state, emit, epoch, *, ordered_rows):
    notes, pending, events, handles = [], {}, [], []
    offset = 0
    precision = {}

    def reset_precision():
        precision.clear()
        precision.update(model_dtype='float32',input_dtypes=[],output_dtypes=[],scaler_enabled=False)

    def check_precision():
        require(amp is False and not scaler.is_enabled() and scaler.get_scale() == 1,
                'FP32 only: AMP/scaler enabled')
        require(not torch.is_autocast_enabled(device.type), 'FP32 only: external autocast enabled')
        require(all(not t.is_floating_point() or t.dtype == torch.float32
                    for t in list(model.parameters())+list(model.buffers())), 'FP32 model dtype required')

    class ObservedLoader:
        def __len__(self): return len(loader)
        def __iter__(self):
            nonlocal offset
            for step,batch in enumerate(loader):
                indices = batch['index'].detach().cpu().tolist()
                labels = batch['label'].detach().cpu().tolist()
                expected = ordered_rows[offset:offset+2]
                require(len(labels) == 2 and len(expected) == 2 and indices == [offset,offset+1]
                        and labels == [r['label'] for r in expected], 'Mixed row/label consumption changed')
                notes.append({'step':step,'rows':copy.deepcopy(expected),
                              'positive':labels.count(1),'negative':labels.count(0)})
                offset += 2
                yield batch

    def pre(module,args):
        check_precision()
        require(args[0].dtype == torch.float32 and args[0].shape[0] == 2, 'FP32 microbatch 2 required')
        precision['input_dtypes'].append('float32')

    def post(module,args,logits):
        require(logits.dtype == torch.float32, 'FP32 output required')
        precision['output_dtypes'].append('float32')
        n = notes[-1]
        y = torch.tensor([r['label'] for r in n['rows']],device=logits.device,dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(logits.detach(),y,weight=weights,reduction='none')
        n.update(positive_weighted_loss_sum=float(loss[y==1].sum()),negative_weighted_loss_sum=float(loss[y==0].sum()))
        if not bool(torch.isfinite(loss).all()): raise FloatingPointError('Nonfinite class diagnostic loss')

    def before_step(opt,args,kwargs):
        grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        finite = bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
        norm = math.sqrt(sum(float(g.double().square().sum()) for g in grads)) if finite else float('nan')
        precision.update(post_clip_finite=finite,post_clip_norm=norm)
        if not finite or not math.isfinite(norm) or norm <= 0: raise FloatingPointError('No finite nonzero clipped gradient')
        pending.clear()
        pending.update(parameter_sha256_before=parameter_hash(model),steps_before=optimizer_steps(model,opt),
                       gradient_parameter_names=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is not None])

    def after_step(opt,args,kwargs):
        pending.update(parameter_sha256_after=parameter_hash(model),steps_after=optimizer_steps(model,opt))
        pending['parameters_changed'] = pending['parameter_sha256_before'] != pending['parameter_sha256_after']
        pending['optimizer_step_advanced'] = bool(pending['gradient_parameter_names']) and all(
            pending['steps_after'][n] == pending['steps_before'][n]+1 for n in pending['gradient_parameter_names'])

    def record(event):
        event = copy.deepcopy(event)
        if event.get('kind') == 'accumulation_boundary':
            selected = [notes[m['step']] for m in event['microbatches']]
            event.update(class_microbatches=copy.deepcopy(selected),
                         class_counts={k:sum(n[k] for n in selected) for k in ('positive','negative')},
                         optimizer_evidence=copy.deepcopy(pending) if event['optimizer_updates'] else None,
                         fp32_evidence=copy.deepcopy(precision))
            events.append(event)
        emit(event)  # Preserve every boundary before enforcing observation-only acceptance.
        if event.get('kind') == 'accumulation_boundary':
            if event['status'] == 'UPDATED':
                require(pending.get('parameters_changed') and pending.get('optimizer_step_advanced'),
                        'AdamW did not advance/change parameters')
            pending.clear()
            reset_precision()

    try:
        reset_precision();check_precision();_check_rows(ordered_rows)
        require(isinstance(optimizer,torch.optim.AdamW), 'Reviewed AdamW required')
        require(epoch == 0 and accumulation == 8 and len(loader) == 384 and max_norm == 1., 'Mixed pass configuration changed')
        require(weights.dtype == torch.float32 and weights.tolist() == [1.,3.], 'Reviewed class weights required')
        require(state == core.new_update_state() and all(v == 0 for v in optimizer_steps(model,optimizer).values()), 'Fresh mixed state required')
        handles = [model.register_forward_pre_hook(pre),model.register_forward_hook(post),
                   optimizer.register_step_pre_hook(before_step),optimizer.register_step_post_hook(after_step)]
        loss = core.train_epoch(model,ObservedLoader(),optimizer,scaler,weights,device,False,8,max_norm,state,record,0)
        require(offset == 768, 'Incomplete mixed row consumption')
        coverage_summary(events)
        return loss
    except BaseException as ex:
        emit({'kind':'mixed_diagnostic_failure','error':repr(ex),'rows_consumed':offset,'expected_rows':768,
              'precision_evidence':copy.deepcopy(precision),'class_microbatches':copy.deepcopy(notes),'no_retry':True})
        raise
    finally:
        for handle in handles: handle.remove()
