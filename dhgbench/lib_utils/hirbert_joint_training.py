"""Shared-encoder pretraining with CPU graph caches and bounded GPU packets."""

import math
import random
import time
import traceback
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import torch

from hir import chunk_hir
from lib_utils.hirbert_local_sampling import LocalBudgetError, LocalLimits, LocalPretrainSampler, move_batch, target_keys
from lib_utils.hirbert_microbatch import MicroBatchPolicy, memory_peaks, packet_cost, synchronize
from lib_utils.hirbert_pretrain_utils import (
    UnlabeledGraph, completed_metrics, cpu_state, grad_norm, make_scheduler, restore_rng,
    rng_state, save_json, training_arguments, unlabeled_graph, write_history,
)
from lib_utils.utils import fix_seed


STAT_FIELDS = ('dataset', 'num_nodes', 'num_hyperedges', 'token_count', 'num_chunks',
               'samples_per_epoch', 'peak_gpu_memory_mb', 'status', 'failure_reason',
               'trained_samples', 'epoch', 'actual_samples_per_epoch', 'avg_local_tokens',
               'avg_local_hyperedges', 'avg_local_nodes', 'avg_membership_targets',
               'avg_relation_targets', 'avg_feature_targets', 'avg_packet_tokens',
               'avg_packets_per_sample', 'avg_local_chunks',
               'avg_self_supervised_targets', 'avg_bert_forward_tokens', 'max_bert_forward_tokens',
               'peak_gpu_reserved_memory_mb', 'samples_per_second', 'tokens_per_second',
               'epoch_oom_retries', 'budget_retries', 'micro_batch_size', 'chunk_batch_size', 'max_micro_batch_tokens',
               'oom_retries', 'max_step_tokens', 'max_step_nodes', 'data_path')

SAMPLE_STATS = {'avg_local_tokens': 'local_tokens', 'avg_local_hyperedges': 'local_hyperedges',
                'avg_local_nodes': 'local_nodes', 'avg_membership_targets': 'membership_count',
                'avg_relation_targets': 'relation_count', 'avg_feature_targets': 'feature_count',
                'avg_packet_tokens': 'packet_tokens', 'avg_packets_per_sample': 'packet_count',
                'avg_local_chunks': 'local_chunks', 'avg_self_supervised_targets': 'self_supervised_targets',
                'avg_bert_forward_tokens': 'bert_forward_tokens'}


def write_stats(directory, stats):
    from benchmark_hirbert_ablation import write_csv
    write_csv(directory / 'pretrain_dataset_stats.csv', list(stats.values()), STAT_FIELDS)


def amp_dtype(args):
    if torch.device(args.device).type != 'cuda' or args.amp == 'off':
        return None
    with torch.cuda.device(args.device):
        bf16 = torch.cuda.is_bf16_supported()
    if args.amp in ('auto', 'bf16') and bf16:
        return torch.bfloat16
    if args.amp == 'bf16':
        print('[AMP] BF16 unavailable; using FP16 with GradScaler')
    return torch.float16


def is_oom(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and 'out of memory' in str(error).lower() and 'cuda' in str(error).lower())


def export_encoder(model, samplers, stats):
    shared = cpu_state(model.encoder)
    adapters = {name: cpu_state(adapter) for name, adapter in model.adapters.items()}
    first = next(iter(samplers))
    combined = {**shared, **{'feature_projection.' + key: value for key, value in adapters[first].items()}}
    return {'shared_encoder_state_dict': shared, 'dataset_adapters': adapters,
            'feature_dimensions': {name: sampler.graph.num_features for name, sampler in samplers.items()},
            'adapter_trained_samples': {name: stats[name]['trained_samples'] for name in adapters},
            # Single-graph checkpoint consumers retain their original contract.
            'encoder_state_dict': combined, 'num_features': samplers[first].graph.num_features if len(samplers) == 1 else None}


def _packet_group(model, samplers, items, epoch, split, limits, args, scaler, dtype, divisor, stats, policy):
    """Stream CPU packets into bounded joint forwards; retain one microbatch.

    A transaction contains several logical samples. Every packet loss already
    has its per-target fragment normalization; dividing by logical sample count
    keeps partial/long samples from changing the optimization weight.
    """
    totals = [defaultdict(float) for _ in items]
    targets = [{task: set() for task in ('membership', 'feature', 'relation')} for _ in items]
    nodes, edges = [set() for _ in items], [set() for _ in items]
    profile = {'calls': [], 'micro_batches': 0, 'packets': 0, 'max_packets': 0, 'max_micro_batch_tokens': 0,
               'peak_gpu_memory_mb': 0., 'peak_gpu_reserved_memory_mb': 0.}
    records, costs = [], []
    current_name = items[0][0]
    started = time.perf_counter()

    def flush():
        nonlocal current_name
        if not records:
            return
        current_name = items[records[0][0]][0]
        names = {items[owner][0] for owner, _ in records}
        model.encoder.chunk_batch_size = min(policy.state.chunk_batch_size,
                                             *(limits[name].chunk_batch_size for name in names))
        if policy.cuda:
            torch.cuda.reset_peak_memory_stats(args.device)
        synchronize(args)
        micro_started = time.perf_counter()
        gpu_batches = [move_batch(batch, args.device) for _, batch in records]
        context = torch.autocast('cuda', dtype=dtype) if dtype is not None else nullcontext()
        with context:
            losses, metrics, calls = model.forward_packets(gpu_batches, policy.state.forward_tokens)
            loss = torch.stack(losses).sum() / divisor
        if not math.isfinite(loss.item()):
            raise FloatingPointError('Non-finite microbatch SSL loss')
        if split == 'train':
            scaler.scale(loss).backward()
        synchronize(args)
        duration = time.perf_counter() - micro_started
        allocated, reserved = memory_peaks(args)
        profile['micro_batches'] += 1
        profile['packets'] += len(records)
        profile['max_packets'] = max(profile['max_packets'], len(records))
        profile['peak_gpu_memory_mb'] = max(profile['peak_gpu_memory_mb'], allocated / 1024 ** 2)
        profile['peak_gpu_reserved_memory_mb'] = max(profile['peak_gpu_reserved_memory_mb'], reserved / 1024 ** 2)
        profile['calls'].extend({**{key: value for key, value in call.items() if not key.startswith('owner_')},
                                 'datasets': sorted({items[records[position][0]][0] for position in call['owner_tokens']})}
                                for call in calls)
        total_tokens = sum(call['tokens'] for call in calls)
        profile['max_micro_batch_tokens'] = max(profile['max_micro_batch_tokens'], total_tokens)
        for position, ((owner, batch), values) in enumerate(zip(records, metrics)):
            name = items[owner][0]
            own_tokens = sum(call['owner_tokens'].get(position, 0) for call in calls)
            owned_calls = [call for call in calls if position in call['owner_tokens']]
            state = totals[owner]
            state['sample_seconds'] += duration * own_tokens / max(1, total_tokens)
            state['local_tokens'] += own_tokens
            state['bert_padded_tokens'] += sum(call['owner_padded_tokens'].get(position, 0) for call in calls)
            state['bert_forward_token_sum'] += sum(call['tokens'] for call in owned_calls)
            state['bert_forward_calls'] += len(owned_calls)
            state['max_bert_forward_tokens'] = max(state['max_bert_forward_tokens'],
                                                  max((call['tokens'] for call in owned_calls), default=0))
            state['peak_gpu_memory_mb'] = max(state['peak_gpu_memory_mb'], allocated / 1024 ** 2)
            state['peak_gpu_reserved_memory_mb'] = max(state['peak_gpu_reserved_memory_mb'], reserved / 1024 ** 2)
            state['packet_count'] += 1
            chunks = batch.view.chunks if {'membership', 'feature'} & set(args.tasks) else ()
            if len(batch.relation_labels):
                chunks += batch.view.relation_chunks
            state['local_chunks'] += len(chunks)
            nodes[owner].update(batch.view.node_ids.tolist())
            edges[owner].update(batch.view.program.source_edge_ids[edge]
                                for chunk in chunks for _, edge, _ in chunk.readouts)
            for task, keys in target_keys(batch).items():
                targets[owner][task].update(keys)
            for key, value in values.items():
                scale = batch.task_scales.get(key.split('_')[0], batch.task_scales['context'])
                state[key] += value if key == 'loss' else value * scale
            stat = stats[name]
            stat['max_step_tokens'] = max(stat['max_step_tokens'], packet_cost(batch)[0])
            stat['max_step_nodes'] = max(stat['max_step_nodes'], len(batch.view.x))
            stat['max_micro_batch_tokens'] = max(stat.get('max_micro_batch_tokens', 0), total_tokens)
            stat['peak_gpu_memory_mb'] = max(stat['peak_gpu_memory_mb'], allocated / 1024 ** 2)
            stat['peak_gpu_reserved_memory_mb'] = max(stat.get('peak_gpu_reserved_memory_mb', 0), reserved / 1024 ** 2)
        del gpu_batches, losses, loss
        records.clear()
        costs.clear()

    try:
        for owner, (name, index) in enumerate(items):
            current_name = name
            iterator = iter(samplers[name].sample(split, epoch, index, limits[name]))
            while True:
                current_name = name
                sample_started = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    totals[owner]['sample_seconds'] += time.perf_counter() - sample_started
                    break
                totals[owner]['sample_seconds'] += time.perf_counter() - sample_started
                cost = packet_cost(batch)
                if records and not policy.fits(costs, cost):
                    flush()
                if not policy.fits(costs, cost):
                    raise LocalBudgetError('One legal packet exceeds the configured microbatch budgets')
                records.append((owner, batch))
                costs.append(cost)
        flush()
        pending = []
        for owner, (name, _) in enumerate(items):
            state = totals[owner]
            if not any(targets[owner][task] for task in args.tasks):
                current_name = name
                raise ValueError('Local view contains no usable self-supervised targets')
            state.update({task + '_count': len(keys) for task, keys in targets[owner].items()})
            state.update(local_nodes=len(nodes[owner]), local_hyperedges=len(edges[owner]),
                         self_supervised_targets=sum(len(keys) for keys in targets[owner].values()),
                         packet_tokens=state['local_tokens'] / max(1, state['packet_count']),
                         bert_forward_tokens=state['bert_forward_token_sum'] / max(1, state['bert_forward_calls']))
            pending.append((name, dict(state)))
        profile['seconds'] = time.perf_counter() - started
        return pending, None, profile
    except Exception as error:
        allocated, reserved = memory_peaks(args)
        profile['peak_gpu_memory_mb'] = max(profile['peak_gpu_memory_mb'], allocated / 1024 ** 2)
        profile['peak_gpu_reserved_memory_mb'] = max(profile['peak_gpu_reserved_memory_mb'], reserved / 1024 ** 2)
        stats[current_name]['peak_gpu_memory_mb'] = max(stats[current_name]['peak_gpu_memory_mb'], allocated / 1024 ** 2)
        stats[current_name]['peak_gpu_reserved_memory_mb'] = max(
            stats[current_name].get('peak_gpu_reserved_memory_mb', 0), reserved / 1024 ** 2)
        kind = 'oom' if is_oom(error) else ('budget' if isinstance(error, LocalBudgetError) else 'failed')
        reason = type(error).__name__ + ': ' + str(error) + '\n' + traceback.format_exc(limit=4)
        return [], (current_name, kind, reason), profile


def _logical_sample(model, sampler, index, epoch, split, limits, args, scaler, dtype, divisor, stat):
    """Single-sample compatibility helper; production batches multiple samples."""
    name = sampler.graph.dataset
    pending, failure, _ = _packet_group(model, {name: sampler}, [(name, index)], epoch, split, {name: limits},
                                        args, scaler, dtype, divisor, {name: stat}, MicroBatchPolicy(args))
    return (None, failure[1], failure[2]) if failure else (pending[0][1], None, '')


def _merge_profile(total, profile):
    total['calls'].extend(profile['calls'])
    total['micro_batches'] += profile['micro_batches']
    total['packets'] = total.get('packets', 0) + profile['packets']
    total['max_packets'] = max(total.get('max_packets', 0), profile['max_packets'])
    total['max_micro_batch_tokens'] = max(total.get('max_micro_batch_tokens', 0), profile['max_micro_batch_tokens'])
    for key in ('peak_gpu_memory_mb', 'peak_gpu_reserved_memory_mb'):
        total[key] = max(total.get(key, 0.), profile[key])


def _merge_memory(total, profile):
    for key in ('peak_gpu_memory_mb', 'peak_gpu_reserved_memory_mb'):
        total[key] = max(total.get(key, 0.), profile[key])


def _profile_row(row, prefix, profile, seconds):
    calls = profile['calls']
    tokens = sum(call['tokens'] for call in calls)
    padded = sum(call['padded_tokens'] for call in calls)
    count = row[prefix + '_samples']
    row.update({prefix + '_seconds': seconds, prefix + '_samples_per_second': count / max(1e-12, seconds),
                prefix + '_tokens_per_second': tokens / max(1e-12, seconds),
                prefix + '_actual_tokens': tokens, prefix + '_bert_padded_tokens': padded,
                prefix + '_bert_padding_fraction': (padded - tokens) / max(1, padded),
                prefix + '_bert_forward_calls': len(calls), prefix + '_micro_batches': profile['micro_batches'],
                prefix + '_micro_batch_packets_mean': profile.get('packets', 0) / max(1, profile['micro_batches']),
                prefix + '_micro_batch_packets_max': profile.get('max_packets', 0),
                prefix + '_micro_batch_tokens_mean': tokens / max(1, profile['micro_batches']),
                prefix + '_micro_batch_tokens_max': profile.get('max_micro_batch_tokens', 0),
                prefix + '_bert_forward_tokens': tokens / max(1, len(calls)),
                prefix + '_max_bert_forward_tokens': max((call['tokens'] for call in calls), default=0),
                prefix + '_max_bert_sequence_length': max((call['max_sequence_length'] for call in calls), default=0),
                # These are measured non-padding input counts for EVERY BERT
                # call, excluding checkpoint recomputation during backward.
                prefix + '_bert_token_batches': ','.join(str(call['tokens']) for call in calls),
                prefix + '_bert_padded_token_batches': ','.join(str(call['padded_tokens']) for call in calls),
                prefix + '_peak_gpu_memory_mb': profile.get('peak_gpu_memory_mb', 0.),
                prefix + '_peak_gpu_reserved_memory_mb': profile.get('peak_gpu_reserved_memory_mb', 0.)})


def _clear(optimizer, args):
    optimizer.zero_grad(set_to_none=True)
    if torch.device(args.device).type == 'cuda':
        torch.cuda.empty_cache()


def plot_datasets(history, names, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(len(names), 2, figsize=(12, 3 * len(names)), squeeze=False, constrained_layout=True)
    for row, name in enumerate(names):
        for column, split in enumerate(('train', 'val')):
            axis = axes[row, column]
            for task in ('loss', 'membership_loss', 'feature_loss', 'relation_loss'):
                recorded = [(entry['epoch'], entry[f'{split}_{name}_{task}']) for entry in history
                            if isinstance(entry[f'{split}_{name}_{task}'], (int, float))]
                if recorded:
                    axis.plot(*zip(*recorded), label=task)
            axis.set(title=f'{name}: {split}', xlabel='Epoch', ylabel='SSL loss')
            if axis.lines:
                axis.legend(fontsize=8)
    figure.savefig(directory / 'pretrain_dataset_losses.png', dpi=130)
    plt.close(figure)


def run_joint_pretraining(data, args, directory, loader, model_factory, save, plot):
    from benchmark_hirbert_ablation import model_config
    from pretrain_hirbert import pretrain_config

    scope = 'multi_graph' if args.pretrain_mode == 'multi_graph' else args.pretrain_datasets[0]
    directory = Path(directory or args.output / scope / 'pretrain')
    directory.mkdir(parents=True, exist_ok=True)
    stats, samplers, infos = {}, {}, {}
    supplied = data if isinstance(data, dict) else ({data.dataset if isinstance(data, UnlabeledGraph) else args.dname: data}
                                                  if data is not None else {})
    samples_per_graph = args.local_samples_per_graph * args.pretrain_steps_per_epoch
    print(f'[pretrain sampling] {samples_per_graph} train / {args.validation_samples_per_graph} fixed validation samples '
          f'per graph; {args.local_min_edges}-{args.local_max_edges} edge goal; '
          f'{args.local_max_tokens} tokens/packet; microbatch={args.micro_batch_size}; '
          f'accumulation={args.gradient_accumulation_steps}')
    fix_seed(args.pretrain_seed)
    for offset, name in enumerate(args.requested_pretrain_datasets):
        stat = dict.fromkeys(STAT_FIELDS, 0)
        stat.update(dataset=name, status='running', failure_reason='', data_path='', samples_per_epoch=samples_per_graph)
        stats[name] = stat
        if name in args.excluded_pretrain_datasets:
            stat.update(status='skipped', failure_reason='Excluded target dataset by request', samples_per_epoch=0)
            continue
        try:
            graph_args = SimpleNamespace(**{**vars(args), 'dname': name, 'device': 'cpu',
                                            'pretrain_seed': args.pretrain_seed + offset * 104729})
            graph = supplied.get(name)
            if graph is None:
                graph = loader(graph_args)
            graph = (replace(graph, x=graph.x.detach().cpu(), dataset=name) if isinstance(graph, UnlabeledGraph)
                     else unlabeled_graph(graph, name))
            chunks, _ = chunk_hir(graph.program, args.hir_max_seq_length)
            stat.update(num_nodes=graph.num_nodes, num_hyperedges=len(graph.program.hyperedges),
                        token_count=len(graph.program.tokens), num_chunks=len(chunks),
                        data_path=(graph.source_paths or {}).get('raw', 'provided CPU graph'))
            samplers[name] = LocalPretrainSampler(graph, graph_args)
            infos[name] = {'dataset': name, 'num_nodes': graph.num_nodes, 'num_features': graph.num_features,
                           'num_hyperedges': len(graph.program.hyperedges), 'setting': 'local_unlabeled_pretraining'}
        except Exception as error:
            stat.update(status='failed', failure_reason=type(error).__name__ + ': ' + str(error))
            print(f'[pretrain dataset failed] {name}: {stat["failure_reason"]}')
    dtype = amp_dtype(args)
    config = pretrain_config(args)
    info = (next(iter(infos.values())) if len(infos) == 1 else
            {'dataset': 'multi_graph', 'graphs': infos, 'setting': 'multi_graph_shared_encoder'})
    expected = {'training': config, 'dataset': info, 'amp_effective': str(dtype)}
    encoder_path = directory / 'pretrained_encoder.pt'
    done = completed_metrics(directory, expected,
                             ('pretrained_encoder.pt', 'pretrain_history.csv', 'pretrain_loss.png', 'pretrain_dataset_stats.csv'),
                             'pretrain_metrics.json')
    if done:
        print(f'[skip complete pretraining] {encoder_path}')
        return encoder_path
    write_stats(directory, stats)
    if not samplers:
        save_json({'config': expected, 'completed': False, 'status': 'failed', 'datasets': stats}, directory / 'pretrain_metrics.json')
        raise RuntimeError(f'No usable pretraining datasets; see {directory / "pretrain_dataset_stats.csv"}')
    specs = {name: (sampler.graph.num_features, sampler.feature_statistics()) for name, sampler in samplers.items()}
    first = next(iter(specs))
    fix_seed(args.pretrain_seed)
    try:
        policy = MicroBatchPolicy(args)
        model = model_factory(specs[first][0], model_config(args), args.weights, specs[first][1],
                              args.node_batch_size, dataset_specs=specs).to(args.device)
    except Exception as error:
        for name in samplers:
            stats[name].update(status='failed', failure_reason='Shared model initialization: ' + str(error))
        write_stats(directory, stats)
        save_json({'config': expected, 'completed': False, 'datasets': stats}, directory / 'pretrain_metrics.json')
        raise RuntimeError(f'Shared encoder initialization failed; report: {directory}') from error
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.pretrain_lr, weight_decay=args.pretrain_wd)
    scheduler = make_scheduler(optimizer, args.pretrain_scheduler, args.pretrain_epochs,
                               min(args.pretrain_warmup_epochs, args.pretrain_epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=dtype == torch.float16)
    limits = {name: LocalLimits(args.local_max_tokens, args.local_max_nodes, args.local_max_chunks,
                                args.local_target_count, args.bert_max_chunk_batch_size, args.local_max_edges,
                                args.local_feature_targets, args.local_relation_targets, args.hir_max_seq_length) for name in samplers}
    history, best_export, best_epoch, bad_epochs = [], None, 0, 0
    best_loss, accumulation, elapsed = float('inf'), args.gradient_accumulation_steps, 0.0
    last_path = directory / 'pretrain_last.pt'
    if last_path.exists():
        if not args.resume:
            raise ValueError('Incomplete pretraining exists; enable --resume or choose another --output')
        saved = torch.load(last_path, map_location=args.device, weights_only=True)
        if saved['config'] != expected:
            raise ValueError('Pretraining resume configuration or dataset dimensions differ')
        model.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict'])
        scaler.load_state_dict(saved['scaler_state_dict'])
        history, best_export = saved['history'], saved['best_export']
        best_epoch, best_loss, bad_epochs = saved['best_epoch'], saved['best_loss'], saved['bad_epochs']
        stats, accumulation = saved['stats'], saved['accumulation']
        elapsed = saved.get('elapsed', 0.0)
        limits = {name: LocalLimits(**values) for name, values in saved['limits'].items()}
        policy.load_state_dict(saved['micro_batch_state'])
        restore_rng(saved['rng_state'])
        print(f'[resume shared pretraining] after epoch {saved["epoch"]}')
    for epoch in range(len(history) + 1, args.pretrain_epochs + 1):
        if bad_epochs >= args.pretrain_patience:
            break
        started = time.perf_counter()
        epoch_oom_before = {name: stat['oom_retries'] for name, stat in stats.items()}
        profiles = {split: {'calls': [], 'micro_batches': 0, 'peak_gpu_memory_mb': 0., 'peak_gpu_reserved_memory_mb': 0.}
                    for split in ('train', 'val')}
        active = [name for name in samplers if stats[name]['status'] != 'failed']
        schedule = [(name, index) for name in active for index in range(samples_per_graph)]
        random.Random(args.pretrain_seed + epoch * 7919).shuffle(schedule)
        training, validation, norms, bert_norms = defaultdict(list), {}, [], []
        for stat in stats.values():
            stat.update(epoch=epoch, actual_samples_per_epoch=0)
            stat.update(dict.fromkeys(SAMPLE_STATS, 0.0))
        position, graph_order = 0, []
        while position < len(schedule):
            width = accumulation * policy.state.packets
            group = [(name, index) for name, index in schedule[position:position + width]
                     if stats[name]['status'] != 'failed']
            position += width
            saved_rng, retries = rng_state(), defaultdict(int)
            overflow_retries = 0
            while group:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                pending, failed, profile = _packet_group(model, samplers, group, epoch, 'train', limits,
                                                          args, scaler, dtype, len(group), stats, policy)
                _merge_memory(profiles['train'], profile)  # Include failed/OOM attempts in peak memory.
                if failed:
                    name, kind, reason = failed
                    _clear(optimizer, args)
                    restore_rng(saved_rng)
                    retries[name] += 1
                    if kind == 'oom':
                        stats[name]['oom_retries'] += 1
                    elif kind == 'budget':
                        stats[name]['budget_retries'] += 1
                    if kind in ('oom', 'budget') and retries[name] <= args.oom_retries:
                        previous_packets = policy.state.packets
                        if kind == 'budget' or not policy.reduce():
                            limits[name].shrink()
                        if policy.state.packets < previous_packets:
                            accumulation = min(256, accumulation * previous_packets // policy.state.packets)
                        print(f'[{kind} retry] {name}: microbatch={policy.state_dict()}; '
                              f'local={asdict(limits[name])}; accumulation={accumulation}')
                    else:
                        stats[name].update(status='failed', failure_reason=reason)
                        print(f'[pretrain dataset failed] {name}: {reason}')
                        group = [(dataset, index) for dataset, index in group if dataset != name]
                    write_stats(directory, stats)
                    continue
                scaler.unscale_(optimizer)
                norms.append(grad_norm(model.encoder))
                bert_norms.append(grad_norm(model.encoder.bert))
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                try:
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                except Exception as error:
                    for name, _ in group:
                        stats[name].update(status='failed', failure_reason='Shared optimizer step: ' + str(error))
                    write_stats(directory, stats)
                    save_json({'config': expected, 'completed': False, 'datasets': stats}, directory / 'pretrain_metrics.json')
                    raise RuntimeError(f'Optimizer failed; resume committed epoch; reports: {directory}') from error
                if scaler.get_scale() < previous_scale:
                    # GradScaler skipped this optimizer update. Replay with
                    # its reduced scale and do not count it as learned data.
                    overflow_retries += 1
                    _clear(optimizer, args)
                    restore_rng(saved_rng)
                    if overflow_retries > args.oom_retries:
                        for name, _ in group:
                            stats[name].update(status='failed', failure_reason='Repeated FP16 gradient overflow')
                        write_stats(directory, stats)
                        break
                    continue
                # Tune only after an optimizer transaction commits. Growing
                # mid-transaction could repeatedly undo an OOM reduction.
                allocated, reserved = memory_peaks(args)
                profile['peak_gpu_memory_mb'] = max(profile['peak_gpu_memory_mb'], allocated / 1024 ** 2)
                profile['peak_gpu_reserved_memory_mb'] = max(profile['peak_gpu_reserved_memory_mb'], reserved / 1024 ** 2)
                _merge_profile(profiles['train'], profile)
                policy.observe(max(allocated, profile['peak_gpu_memory_mb'] * 1024 ** 2),
                               max(reserved, profile['peak_gpu_reserved_memory_mb'] * 1024 ** 2),
                               profile['max_packets'])
                for name, result in pending:
                    for key in ('peak_gpu_memory_mb', 'peak_gpu_reserved_memory_mb'):
                        result[key] = max(result[key], profile[key])
                        stats[name][key] = max(stats[name][key], profile[key])
                    training[name].append(result)
                    stats[name]['trained_samples'] += 1
                    # Update only after the accumulation group commits. Failed
                    # attempts never inflate sample counts or size averages.
                    count = stats[name]['actual_samples_per_epoch'] + 1
                    for column, metric in SAMPLE_STATS.items():
                        stats[name][column] += (result[metric] - stats[name][column]) / count
                    stats[name]['actual_samples_per_epoch'] = count
                    graph_order.append(name)
                break
        synchronize(args)
        train_seconds = time.perf_counter() - started
        validation_started = time.perf_counter()
        model.eval()
        for name in active:
            if stats[name]['status'] == 'failed':
                continue
            values, retries = [], 0
            while True:
                values.clear()
                failure = None
                dataset_profile = {'calls': [], 'micro_batches': 0, 'peak_gpu_memory_mb': 0., 'peak_gpu_reserved_memory_mb': 0.}
                with torch.no_grad():
                    count = max(args.validation_samples_per_graph, len(samplers[name].val_relations))
                    step = policy.state.packets
                    for index in range(0, count, step):
                        items = [(name, offset) for offset in range(index, min(count, index + step))]
                        pending, failed, profile = _packet_group(model, samplers, items, 0, 'valid', limits,
                                                                  args, scaler, dtype, 1, stats, policy)
                        _merge_memory(profiles['val'], profile)
                        if failed:
                            failure = failed
                            break
                        values.extend(result for _, result in pending)
                        _merge_profile(dataset_profile, profile)
                if failure is None:
                    validation[name] = values.copy()
                    _merge_profile(profiles['val'], dataset_profile)
                    break
                retries += 1
                if failure[1] == 'oom':
                    stats[name]['oom_retries'] += 1
                elif failure[1] == 'budget':
                    stats[name]['budget_retries'] += 1
                if failure[1] in ('oom', 'budget') and retries <= args.oom_retries:
                    if failure[1] == 'budget' or not policy.reduce():
                        limits[name].shrink()
                    _clear(optimizer, args)
                else:
                    stats[name].update(status='failed', failure_reason='Validation: ' + failure[2])
                    print(f'[validation failed] {name}: {failure[2]}')
                    break
        synchronize(args)
        validation_seconds = time.perf_counter() - validation_started
        if not validation:
            write_stats(directory, stats)
            save_json({'config': expected, 'completed': False, 'datasets': stats}, directory / 'pretrain_metrics.json')
            raise RuntimeError(f'All pretraining datasets failed; see reports in {directory}')
        cohort = ','.join(sorted(validation))
        validation_budgets = repr({name: asdict(limits[name]) for name in sorted(validation)})
        if history and (history[-1].get('validation_datasets', cohort) != cohort or
                        history[-1].get('validation_budgets', validation_budgets) != validation_budgets):
            # Dataset failures or OOM repacking change the validation average's
            # context. Compare best losses only within an unchanged protocol.
            best_loss, bad_epochs = float('inf'), 0
        keys = next(iter(validation.values()))[0].keys()
        row = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'],
               'encoder_grad_norm': sum(norms) / max(1, len(norms)), 'bert_grad_norm': sum(bert_norms) / max(1, len(bert_norms)),
               'graph_order': ','.join(graph_order), 'accumulation_steps': accumulation,
               'micro_batch_size': policy.state.packets, 'chunk_batch_size': policy.state.chunk_batch_size,
               'bert_max_batch_tokens': policy.state.forward_tokens,
               'epoch_oom_retries': sum(stat['oom_retries'] - epoch_oom_before[name] for name, stat in stats.items())}
        row['validation_datasets'] = cohort
        row['validation_budgets'] = validation_budgets
        for split, groups in (('train', training), ('val', validation)):
            means = {name: {key: (max(item[key] for item in values) if 'peak_gpu' in key or key.startswith('max_')
                                 else sum(item[key] for item in values) / len(values)) for key in keys}
                     for name, values in groups.items() if values}
            row[f'{split}_samples'] = sum(map(len, groups.values()))
            for name in args.requested_pretrain_datasets:
                row[f'{split}_{name}_samples'] = len(groups.get(name, ()))
                for column, metric in SAMPLE_STATS.items():
                    row[f'{split}_{name}_{column}'] = means[name][metric] if name in means else ''
            for key in keys:
                row[f'{split}_{key}'] = (max((values[key] for values in means.values()), default=0)
                                         if 'peak_gpu' in key or key.startswith('max_') else
                                         sum(values[key] for values in means.values()) / max(1, len(means)))
                if key == 'feature_metric' and len({decoder.kind for decoder in model.feature_decoders.values()}) > 1:
                    row[f'{split}_{key}'] = ''  # Recall and RMSE have no common numerical scale.
                for name in args.requested_pretrain_datasets:
                    row[f'{split}_{name}_{key}'] = means[name][key] if name in means else ''
            _profile_row(row, split, profiles[split], train_seconds if split == 'train' else validation_seconds)
            actual_targets = sum(value['self_supervised_targets'] for values in groups.values() for value in values)
            row[f'{split}_actual_self_supervised_targets'] = actual_targets
            row[f'{split}_targets_per_second'] = actual_targets / max(1e-12, train_seconds if split == 'train' else validation_seconds)
            for name in args.requested_pretrain_datasets:
                calls = [call for call in profiles[split]['calls'] if name in call['datasets']]
                row[f'{split}_{name}_bert_forward_calls'] = len(calls)
                row[f'{split}_{name}_bert_forward_tokens'] = sum(call['tokens'] for call in calls) / max(1, len(calls))
                row[f'{split}_{name}_max_bert_forward_tokens'] = max((call['tokens'] for call in calls), default=0)
                values = groups.get(name, ())
                duration = sum(value['sample_seconds'] for value in values)
                row[f'{split}_{name}_samples_per_second'] = len(values) / max(1e-12, duration)
                row[f'{split}_{name}_tokens_per_second'] = sum(value['local_tokens'] for value in values) / max(1e-12, duration)
        for name, stat in stats.items():
            stat.update(epoch_oom_retries=stat['oom_retries'] - epoch_oom_before[name],
                        micro_batch_size=policy.state.packets, chunk_batch_size=min(
                            policy.state.chunk_batch_size, limits[name].chunk_batch_size) if name in limits else 0,
                        samples_per_second=row[f'train_{name}_samples_per_second'],
                        tokens_per_second=row[f'train_{name}_tokens_per_second'],
                        avg_bert_forward_tokens=row[f'train_{name}_bert_forward_tokens'],
                        max_bert_forward_tokens=row[f'train_{name}_max_bert_forward_tokens'])
            row[f'train_{name}_epoch_oom_retries'] = stat['epoch_oom_retries']
        history.append(row)
        if row['val_loss'] < best_loss:
            best_loss, best_epoch, bad_epochs = row['val_loss'], epoch, 0
            best_export = export_encoder(model, samplers, stats)
        else:
            bad_epochs += 1
        scheduler.step()
        elapsed += time.perf_counter() - started
        save({'config': expected, 'epoch': epoch, 'model_state_dict': cpu_state(model),
              'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
              'scaler_state_dict': scaler.state_dict(), 'rng_state': rng_state(), 'history': history,
              'best_export': best_export, 'best_epoch': best_epoch, 'best_loss': best_loss,
              'bad_epochs': bad_epochs, 'stats': stats, 'limits': {name: asdict(value) for name, value in limits.items()},
              'accumulation': accumulation, 'micro_batch_state': policy.state_dict(), 'elapsed': elapsed}, last_path)
        write_history(directory / 'pretrain_history.csv', history)
        write_stats(directory, stats)
        if epoch == 1 or epoch % args.display_step == 0:
            print(f'[shared pretrain] epoch={epoch} graphs={len(validation)} train_loss={row["train_loss"]:.4f} '
                  f'val_loss={row["val_loss"]:.4f} samples/s={row["train_samples_per_second"]:.2f} '
                  f'tokens/s={row["train_tokens_per_second"]:.0f} microbatch={policy.state.packets} '
                  f'allocated={row["train_peak_gpu_memory_mb"]:.0f}MiB '
                  f'reserved={row["train_peak_gpu_reserved_memory_mb"]:.0f}MiB '
                  f'oom={row["epoch_oom_retries"]} bert_grad={row["bert_grad_norm"]:.4g}')
    for name in samplers:
        if stats[name]['status'] != 'failed':
            stats[name]['status'] = 'completed' if stats[name]['trained_samples'] else 'skipped'
    failed = [name for name in stats if stats[name]['status'] == 'failed']
    write_history(directory / 'pretrain_history.csv', history)
    plot(history, directory / 'pretrain_loss.png')
    plot_datasets(history, args.requested_pretrain_datasets, directory)
    write_stats(directory, stats)
    save({**best_export, 'kind': 'hirbert_pretrained_encoder', 'model_config': model.model_config,
          'dataset_info': info, 'pretrain_datasets': list(samplers), 'seed': args.pretrain_seed,
          'best_epoch': best_epoch, 'best_val_loss': best_loss, 'pretrain_config': config,
          'training_config': training_arguments(args), 'tasks': args.tasks, 'loss_weights': args.weights,
          'completed': True, 'completed_with_failures': bool(failed), 'failed_datasets': failed}, encoder_path)
    save_json({'config': expected, 'completed': True, 'completed_with_failures': bool(failed),
               'failed_datasets': failed, 'best_epoch': best_epoch, 'last_epoch': len(history),
               'best_val_loss': best_loss, 'best_metrics': history[best_epoch - 1], 'datasets': stats, 'train_time': elapsed,
               'feature_loss_kinds': {name: decoder.kind for name, decoder in model.feature_decoders.items()},
               'amp_effective': str(dtype), 'limits': {name: asdict(value) for name, value in limits.items()},
               'micro_batch_state': policy.state_dict(),
               'split_summary': {name: sampler.summary() for name, sampler in samplers.items()},
               'parameter_count': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)},
              directory / 'pretrain_metrics.json')
    if failed:
        print(f'[pretraining completed with failures] {failed}; report: {directory / "pretrain_dataset_stats.csv"}')
    return encoder_path
