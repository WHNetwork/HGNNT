"""Joint structural pretraining and node finetuning with bounded GPU execution.

Reuses HIRBERT's CPU graph caches, target sampling, MicroBatchPolicy,
fragment readouts, scheduler and atomic/RNG saves.
No feature adapter/decoder or feature-aware encoder is constructed.
"""

import csv
import gc
import json
import math
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from lib_models.hirgraphbert import HIRGraphBERT, load_encoder, model_config
from lib_models.hirgraphbert_pretrain import HIRGraphPretrainer
from lib_utils.hirbert_microbatch import MicroBatchPolicy, memory_peaks, synchronize
from lib_utils.hirbert_pretrain_utils import (atomic_save, completed_metrics, cpu_state, grad_norm, make_scheduler,
                                            restore_rng, rng_state, save_json, write_history)
from lib_utils.hirgraphbert_data import StructureSampler, TASKS, load_graph


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def amp_dtype(args):
    if torch.device(args.device).type != 'cuda' or args.amp == 'off':
        return None
    with torch.cuda.device(args.device):
        bf16 = torch.cuda.is_bf16_supported()
    if args.amp in ('auto', 'bf16') and bf16:
        return torch.bfloat16
    if args.amp == 'bf16':
        print('[AMP] BF16 unavailable; falling back to FP16 + GradScaler')
    return torch.float16


def autocast(dtype):
    return torch.autocast('cuda', dtype=dtype) if dtype is not None else nullcontext()


def is_oom(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and 'cuda' in str(error).lower() and 'out of memory' in str(error).lower())


def clear_memory(optimizer, args):
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    if torch.device(args.device).type == 'cuda':
        torch.cuda.empty_cache()


def write_csv(path, rows):
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with Path(path).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)


def configuration(args, stage):
    keys = (('pretrain_seed', 'pretrain_epochs', 'pretrain_patience', 'pretrain_lr', 'pretrain_wd',
             'pretrain_datasets', 'local_samples_per_graph', 'validation_samples_per_graph',
             'local_max_nodes', 'local_target_count',
             'local_relation_targets', 'local_min_edges', 'local_max_edges', 'local_hops',
             'membership_mask_ratio', 'negative_ratio', 'hir_mask_ratio', 'local_hir_targets', 'hir_payload_candidates',
             'relation_pairs_per_class', 'relation_max_pairs', 'ssl_val_ratio', 'gradient_accumulation_steps',
             'lambda_m', 'lambda_r', 'lambda_h', 'micro_batch_size', 'micro_batch_max_size',
             'micro_batch_max_tokens', 'micro_batch_max_nodes', 'micro_batch_max_chunks', 'bert_max_batch_tokens',
             'bert_max_chunk_batch_size', 'auto_micro_batch', 'micro_batch_tune_interval',
             'gpu_memory_budget_gb', 'gpu_target_memory_gb', 'oom_retries') if stage == 'pretrain' else
            ('dname', 'epochs', 'patience', 'lr', 'wd', 'train_prop', 'valid_prop'))
    result = {**{key: getattr(args, key) for key in keys}, 'model_config': model_config(args),
            'amp': args.amp, 'grad_clip': args.grad_clip, 'warmup_epochs': args.warmup_epochs,
            'scheduler': args.scheduler, 'input_mode': 'Structure-only', 'model': 'HIR-GraphBERT',
            'feature_noise': args.feature_noise, 'data_root': str(Path(args.data_root).resolve())}
    if stage == 'pretrain':
        # The objective changed while encoder weights stayed compatible. An
        # old training transaction must not silently resume with new labels.
        result['hir_objective'] = 'visible_member_anchor_and_hidden_instruction_fields'
        result['membership_negatives'] = 'disjoint_train_validation_banks'
        result['membership_context'] = 'one_visible_member_per_duplicate_family'
    return result


def _transaction(model, samplers, items, args, policy, scaler, dtype, split, epoch, event_writer=None):
    """Accumulate complete logical samples, backpropagating one microbatch at a time."""
    totals = [defaultdict(float) for _ in items]
    records, costs = [], []
    profile = {'tokens': 0, 'padded_tokens': 0, 'calls': 0, 'micro_batches': 0,
               'max_forward_tokens': 0, 'peak_gpu_memory_mb': 0., 'peak_gpu_reserved_memory_mb': 0., 'max_packets': 0,
               'max_packet_tokens': 0, 'max_packet_nodes': 0, 'max_packet_chunks': 0,
               'view_splits': 0, 'oversized_single_views': 0}

    def flush():
        if not records:
            return
        if policy.cuda:
            torch.cuda.reset_peak_memory_stats(args.device)
        model.encoder.chunk_batch_size = min(policy.state.chunk_batch_size, args.bert_max_chunk_batch_size)
        try:
            with autocast(dtype):
                losses, metrics, calls = model.forward_packets([batch for _, batch in records], policy.state.forward_tokens)
                loss = torch.stack(losses).sum() / len(items)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite structural SSL loss')
            if split == 'train':
                scaler.scale(loss).backward()
            synchronize(args)
        except RuntimeError as error:
            if event_writer is not None and is_oom(error):
                event_writer({'event': 'micro_batch_failure', 'kind': 'cuda_oom', 'epoch': epoch,
                              'split': split, 'samples': [items[owner] for owner, _ in records],
                              'packet_costs': list(costs),
                              'forward_tokens': policy.state.forward_tokens,
                              'chunk_batch_size': model.encoder.chunk_batch_size, 'reason': str(error)})
            raise
        allocated, reserved = memory_peaks(args)
        profile['peak_gpu_memory_mb'] = max(profile['peak_gpu_memory_mb'], allocated / 1024 ** 2)
        profile['peak_gpu_reserved_memory_mb'] = max(profile['peak_gpu_reserved_memory_mb'], reserved / 1024 ** 2)
        profile['micro_batches'] += 1
        profile['max_packets'] = max(profile['max_packets'], len(records))
        profile['calls'] += len(calls)
        profile['tokens'] += sum(call['tokens'] for call in calls)
        profile['padded_tokens'] += sum(call['padded_tokens'] for call in calls)
        profile['max_forward_tokens'] = max(profile['max_forward_tokens'], max((call['padded_tokens'] for call in calls), default=0))
        for (owner, batch), row in zip(records, metrics):
            for key, value in row.items():
                task = key.split('_')[0]
                scale = batch.task_scales.get(task, batch.task_scales['context'])
                totals[owner][key] += value if key == 'loss' or key.endswith('_count') else value * scale
            totals[owner]['packets'] += 1
            totals[owner]['tokens'] += batch.cost[0]
        records.clear()
        costs.clear()

    for owner, (name, index) in enumerate(items):
        for packet_index, batch in enumerate(samplers[name].sample(split, epoch, index)):
            cost = batch.cost
            singleton = not policy.fits([], cost)
            if event_writer is not None and (index == 0 or singleton):
                event_writer({'event': 'packet', 'epoch': epoch, 'split': split, 'dataset': name,
                              'sample_index': index, 'sample_packet_index': packet_index, 'cost': cost,
                              'view_costs': batch.view_costs, 'singleton_chunk_execution': singleton,
                              'micro_batch_grouping_caps': (args.micro_batch_max_tokens, args.micro_batch_max_nodes,
                                                            args.micro_batch_max_chunks),
                              'forward_tokens': policy.state.forward_tokens,
                              'chunk_batch_size': policy.state.chunk_batch_size})
            profile['max_packet_tokens'] = max(profile['max_packet_tokens'], cost[0])
            profile['max_packet_nodes'] = max(profile['max_packet_nodes'], cost[1])
            profile['max_packet_chunks'] = max(profile['max_packet_chunks'], cost[2])
            profile['view_splits'] += int(packet_index > 0)
            if records and not policy.fits(costs, cost):
                flush()
            records.append((owner, batch))
            costs.append(cost)
            if singleton:
                # These caps limit grouping, not whether a legal view exists.
                # encode_many forwards complete chunks under the forward cap;
                # all NODE/EDGE outputs are pooled before this task's one loss.
                profile['oversized_single_views'] += 1
                flush()
    flush()
    for values in totals:
        if not any(values[task + '_count'] for task in TASKS):
            raise ValueError('Local sample has no structural targets')
    return list(zip([name for name, _ in items], map(dict, totals))), profile


def _averages(records):
    return {key: sum(record[key] for record in records) / len(records) for key in records[0]} if records else {}


def run_pretraining(args, graphs=None, loader=load_graph):
    directory = Path(args.output) / 'multi_graph' / 'pretrain'
    directory.mkdir(parents=True, exist_ok=True)
    def record_event(event):
        with (directory / 'pretrain_packet_budget.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event) + '\n')

    fix_seed(args.pretrain_seed)
    samplers, graph_info = {}, {}
    for offset, name in enumerate(args.pretrain_datasets):
        graph_args = SimpleNamespace(**{**vars(args), 'dname': name, 'device': 'cpu',
                                        'pretrain_seed': args.pretrain_seed + offset * 104729})
        graph = graphs[name] if graphs is not None else loader(graph_args)
        samplers[name] = StructureSampler(graph, graph_args)
        graph_info[name] = {'num_nodes': graph.num_nodes, 'num_hyperedges': len(graph.program.hyperedges),
                            'logical_tokens': len(graph.program.tokens), 'source_paths': graph.source_paths,
                            'split_summary': samplers[name].summary()}
    config = {**configuration(args, 'pretrain'), 'graphs': graph_info}
    done = completed_metrics(directory, config, ('pretrained_encoder.pt', 'pretrain_history.csv'), 'pretrain_metrics.json')
    if done:
        print(f'[skip completed HIR-GraphBERT pretraining] {directory}')
        return directory / 'pretrained_encoder.pt'
    dtype = amp_dtype(args)
    policy = MicroBatchPolicy(args)
    fix_seed(args.pretrain_seed)
    model = HIRGraphPretrainer(model_config(args), {'membership': args.lambda_m, 'relation': args.lambda_r,
                                                   'hir': args.lambda_h}).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.pretrain_lr, weight_decay=args.pretrain_wd)
    scheduler = make_scheduler(optimizer, args.scheduler, args.pretrain_epochs, min(args.warmup_epochs, args.pretrain_epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=dtype == torch.float16)
    print(f'[structural execution] no local token/chunk admission limit; chunk<={args.hir_max_seq_length} '
          f'forward_tokens<={args.bert_max_batch_tokens}; microbatch caps only control grouping')
    history, best_encoder, best_epoch, bad_epochs, elapsed = [], None, 0, 0, 0.
    best_loss = float('inf')
    trained = {name: 0 for name in samplers}
    last_path = directory / 'pretrain_last.pt'
    if last_path.exists():
        if not args.resume:
            raise ValueError('Pretraining exists; enable --resume or choose another --output')
        saved = torch.load(last_path, map_location=args.device, weights_only=True)
        if saved['config'] != config:
            raise ValueError('Pretraining resume configuration differs')
        model.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict'])
        scaler.load_state_dict(saved['scaler_state_dict'])
        history, best_encoder, best_epoch = saved['history'], saved['best_encoder'], saved['best_epoch']
        best_loss, bad_epochs, elapsed, trained = saved['best_loss'], saved['bad_epochs'], saved['elapsed'], saved['trained_samples']
        policy.load_state_dict(saved['micro_batch_state'])
        restore_rng(saved['rng_state'])
        print(f'[resume Structure-only pretraining] after epoch {len(history)}')
    counts = model.encoder.parameter_counts() | {'total_pretraining': sum(p.numel() for p in model.parameters())}
    print(f'[HIR-GraphBERT parameters] {counts}; AMP={dtype}; budget={args.gpu_memory_budget_gb}GiB')
    for epoch in range(len(history) + 1, args.pretrain_epochs + 1):
        if bad_epochs >= args.pretrain_patience:
            break
        started = time.perf_counter()
        schedule = [(name, index) for name in samplers for index in range(args.local_samples_per_graph)]
        random.Random(args.pretrain_seed + epoch * 7919).shuffle(schedule)
        row = {'epoch': epoch, 'seed': args.pretrain_seed, 'input_mode': 'Structure-only',
               'lr': optimizer.param_groups[0]['lr'], 'oom_retries': 0}
        for split in ('train', 'valid'):
            model.train(split == 'train')
            records = defaultdict(list)
            split_schedule = schedule if split == 'train' else [
                (name, index) for name in samplers for index in range(args.validation_samples_per_graph)]
            position, norms = 0, []
            profiles = []
            while position < len(split_schedule):
                width = policy.state.packets * (args.gradient_accumulation_steps if split == 'train' else 1)
                items = split_schedule[position:position + width]
                transaction_rng, retries = rng_state(), 0
                while True:
                    optimizer.zero_grad(set_to_none=True)
                    unscaled = False
                    try:
                        context = nullcontext() if split == 'train' else torch.no_grad()
                        with context:
                            results, profile = _transaction(model, samplers, items, args, policy, scaler, dtype,
                                                            split, epoch, record_event)
                        if split == 'train':
                            scaler.unscale_(optimizer)
                            unscaled = True
                            norm = grad_norm(model.encoder)
                            if not math.isfinite(norm):
                                if dtype != torch.float16:
                                    raise FloatingPointError('Non-finite encoder gradients')
                            if args.grad_clip:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                            before = scaler.get_scale()
                            # OOM during an optimizer mutation must not be replayed.
                            try:
                                scaler.step(optimizer)
                                scaler.update()
                            except RuntimeError as error:
                                raise RuntimeError('Optimizer update failed; resume the committed epoch checkpoint') from error
                            if scaler.get_scale() < before:
                                retries += 1
                                if retries > args.oom_retries:
                                    raise FloatingPointError('Repeated FP16 gradient overflow')
                                clear_memory(optimizer, args)
                                restore_rng(transaction_rng)
                                continue
                            norms.append(norm)
                        break
                    except RuntimeError as error:
                        if not is_oom(error):
                            raise
                        reason = str(error)
                        record_event({'event': 'failure', 'kind': 'cuda_oom',
                                      'epoch': epoch, 'split': split, 'samples': items, 'reason': reason,
                                      'micro_batch': policy.state_dict()})
                        error.__traceback__ = None
                        if unscaled and scaler.is_enabled():
                            # An OOM in gradient diagnostics/clipping happened
                            # after unscale. Reset that transaction's scaler
                            # stage before replay, keeping the numerical scale.
                            scaler.update(new_scale=scaler.get_scale())
                        clear_memory(optimizer, args)
                        restore_rng(transaction_rng)
                        retries += 1
                        row['oom_retries'] += 1
                        if retries > args.oom_retries or not policy.reduce():
                            raise RuntimeError(f'Structural {split} exhausted CUDA OOM retries: {reason}; '
                                               'local views and validation targets were kept fixed') from error
                        print(f'[Structure-only CUDA OOM retry] {split} retry={retries} '
                              f'microbatch={policy.state_dict()}; sample inputs and targets unchanged')
                if split == 'train':
                    policy.observe(profile['peak_gpu_memory_mb'] * 1024 ** 2,
                                   profile['peak_gpu_reserved_memory_mb'] * 1024 ** 2, profile['max_packets'])
                for name, result in results:
                    records[name].append(result)
                    if split == 'train':
                        trained[name] += 1
                profiles.append(profile)
                position += len(items)
            prefix = 'val' if split == 'valid' else split
            means = {name: _averages(values) for name, values in records.items()}
            for key in next(iter(means.values())):
                row[prefix + '_' + key] = sum(value[key] for value in means.values()) / len(means)
                for name, values in means.items():
                    row[f'{prefix}_{name}_{key}'] = values[key]
            row[prefix + '_samples'] = sum(len(values) for values in records.values())
            for name, values in records.items():
                row[f'{prefix}_{name}_samples'] = len(values)
            for key in profiles[0]:
                row[prefix + '_' + key] = (max(value[key] for value in profiles) if key.startswith(('max_', 'peak_'))
                                          else sum(value[key] for value in profiles))
            row[prefix + '_encoder_grad_norm'] = sum(norms) / max(1, len(norms))
        if row['val_loss'] < best_loss:
            best_loss, best_epoch, bad_epochs, best_encoder = row['val_loss'], epoch, 0, cpu_state(model.encoder)
        else:
            bad_epochs += 1
        scheduler.step()
        synchronize(args)
        row['seconds'] = time.perf_counter() - started
        row['train_samples_per_second'] = row['train_samples'] / row['seconds']
        row['micro_batch_size'], row['chunk_batch_size'] = policy.state.packets, policy.state.chunk_batch_size
        elapsed += row['seconds']
        history.append(row)
        atomic_save({'config': config, 'model_state_dict': cpu_state(model), 'optimizer_state_dict': optimizer.state_dict(),
                     'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict(),
                     'rng_state': rng_state(), 'history': history, 'best_encoder': best_encoder, 'best_epoch': best_epoch,
                     'best_loss': best_loss, 'bad_epochs': bad_epochs, 'trained_samples': trained, 'elapsed': elapsed,
                     'micro_batch_state': policy.state_dict()}, last_path)
        write_csv(directory / 'pretrain_history.csv', history)
        if epoch == 1 or epoch % args.display_step == 0:
            losses = ' '.join(f'{task}={row["train_" + task + "_loss"]:.4f}/{row["train_" + task + "_acc"]:.3f}' for task in TASKS)
            print(f'[Structure-only pretrain] epoch={epoch} {losses} (loss/acc) val={row["val_loss"]:.4f} '
                  f'node_var={row["train_node_repr_variance"]:.5f} samples={row["train_samples"]} '
                  f'packet_tokens(train/val)={row["train_max_packet_tokens"]}/{row["val_max_packet_tokens"]} '
                  f'packet_chunks(train/val)={row["train_max_packet_chunks"]}/{row["val_max_packet_chunks"]} '
                  f'view_splits(train/val)={row["train_view_splits"]}/{row["val_view_splits"]} '
                  f'allocated={row["train_peak_gpu_memory_mb"]:.0f}MiB reserved={row["train_peak_gpu_reserved_memory_mb"]:.0f}MiB')
    if best_encoder is None:
        raise RuntimeError('No pretraining epoch completed')
    encoder_path = directory / 'pretrained_encoder.pt'
    atomic_save({'kind': 'hirgraphbert_pretrained_encoder', 'input_mode': 'Structure-only',
                 'shared_encoder_state_dict': best_encoder, 'model_config': model_config(args),
                 'seed': args.pretrain_seed, 'best_epoch': best_epoch, 'best_val_loss': best_loss,
                 'pretrain_datasets': list(samplers), 'pretrain_config': config, 'trained_samples': trained,
                 'tasks': list(TASKS), 'loss_weights': model.weights,
                 'parameter_counts': counts, 'completed': True}, encoder_path)
    stats = [{'dataset': name, **{key: value for key, value in info.items() if key not in ('split_summary', 'source_paths')},
              'trained_samples': trained[name], 'samples_per_epoch': args.local_samples_per_graph,
              'status': 'completed', 'input_mode': 'Structure-only'} for name, info in graph_info.items()]
    write_csv(directory / 'pretrain_dataset_stats.csv', stats)
    save_json({'completed': True, 'config': config, 'input_mode': 'Structure-only', 'seed': args.pretrain_seed,
               'best_epoch': best_epoch, 'best_val_loss': best_loss, 'last_epoch': len(history), 'train_time': elapsed,
               'amp_effective': str(dtype), 'parameter_counts': counts, 'datasets': stats,
               'best_metrics': history[best_epoch - 1]}, directory / 'pretrain_metrics.json')
    return encoder_path


def make_masks(data, args, seed):
    known = torch.where(data.y >= 0)[0]
    order = known[torch.randperm(len(known), generator=torch.Generator().manual_seed(seed))]
    first, second = int(len(order) * args.train_prop), int(len(order) * (args.train_prop + args.valid_prop))
    partitions = (order[:first], order[first:second], order[second:])
    if any(not len(ids) for ids in partitions):
        raise ValueError('Train/validation/test label splits must all be nonempty')
    masks = {}
    for name, ids in zip(('train', 'valid', 'test'), partitions):
        mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=args.device)
        mask[ids.to(args.device)] = True
        masks[name] = mask
    return masks


def _node_operation(model, data, masks, args, policy, dtype, optimizer=None, scaler=None, splits=('train', 'valid')):
    saved_rng = rng_state()
    for attempt in range(args.oom_retries + 1):
        logits = loss = None
        model.encoder.chunk_batch_size = policy.state.chunk_batch_size
        model.forward_token_budget = policy.state.forward_tokens
        if policy.cuda:
            torch.cuda.reset_peak_memory_stats(args.device)
        try:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                with autocast(dtype):
                    logits, _ = model(data)
                    loss = F.cross_entropy(logits[masks['train']].float(), data.y[masks['train']])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite node classification loss')
                scaler.scale(loss).backward()
                return {'train_loss': loss.detach().item(), 'node_repr_variance': model.last_node_variance}
            with torch.no_grad(), autocast(dtype):
                logits, _ = model(data)
                result = {'node_repr_variance': model.last_node_variance}
                for split in splits:
                    prefix, mask = ('val' if split == 'valid' else split), masks[split]
                    result[prefix + '_loss'] = F.cross_entropy(logits[mask].float(), data.y[mask]).item()
                    result[prefix + '_acc'] = 100 * (logits[mask].argmax(-1) == data.y[mask]).float().mean().item()
            return result
        except RuntimeError as error:
            if not is_oom(error) or attempt == args.oom_retries or not policy.reduce():
                raise
            error.__traceback__ = None
            # This function's frame survives the retry. Its failed logits/loss
            # otherwise retain the autograd graph even after empty_cache().
            logits = loss = None
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            gc.collect()
            if policy.cuda:
                torch.cuda.empty_cache()
            restore_rng(saved_rng)
            print(f'[node classification OOM retry] chunks/forward={policy.state.chunk_batch_size}')


def finetune_run(data, args, variant, seed, checkpoint=None):
    directory = Path(args.output) / args.dname / variant / f'seed_{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    config = {**configuration(args, 'finetune'), 'seed': seed, 'variant': variant,
              'num_nodes': data.num_nodes, 'num_classes': data.num_classes,
              'pretrained_checkpoint': str(Path(checkpoint).resolve()) if variant == 'pretrained' else None}
    done = completed_metrics(directory, config, ('training.csv', 'best_checkpoint.pt'))
    if done:
        print(f'[skip completed Structure-only classification] {variant}/seed_{seed}')
        return done
    fix_seed(seed)
    run_args = SimpleNamespace(**{**vars(args), 'seed': seed})
    model = HIRGraphBERT(data.num_classes, run_args).to(args.device)
    model.reset_parameters()
    # Reset is deliberately before transfer, never after it.
    report = load_encoder(model, checkpoint) if variant == 'pretrained' else None
    if report:
        print(f'[encoder transfer] {len(report["loaded_keys"])} tensors; missing=[] unexpected=[]')
    masks = make_masks(data, args, seed)
    data = SimpleNamespace(**{**vars(data), 'y': data.y.to(args.device)})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = make_scheduler(optimizer, args.scheduler, args.epochs, min(args.warmup_epochs, args.epochs))
    dtype, policy = amp_dtype(args), MicroBatchPolicy(args)
    policy.state.packets = 1  # Full-graph node aggregation is never split into independent training graphs.
    scaler = torch.cuda.amp.GradScaler(enabled=dtype == torch.float16)
    history, best_state, best_epoch, best_acc, best_loss, bad_epochs, elapsed = [], None, 0, -1., float('inf'), 0, 0.
    last_path = directory / 'last_checkpoint.pt'
    if last_path.exists():
        if not args.resume:
            raise ValueError('Classification exists; enable --resume or choose another --output')
        saved = torch.load(last_path, map_location=args.device, weights_only=True)
        if saved['config'] != config:
            raise ValueError('Classification resume configuration differs')
        if any(not torch.equal(masks[key].cpu(), saved['masks'][key].cpu()) for key in masks):
            raise ValueError('Classification resume label splits differ')
        model.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict'])
        scaler.load_state_dict(saved['scaler_state_dict'])
        history, best_state, best_epoch = saved['history'], saved['best_state'], saved['best_epoch']
        best_acc, best_loss, bad_epochs, elapsed = saved['best_acc'], saved['best_loss'], saved['bad_epochs'], saved['elapsed']
        policy.load_state_dict(saved['micro_batch_state'])
        restore_rng(saved['rng_state'])
    for epoch in range(len(history) + 1, args.epochs + 1):
        if bad_epochs >= args.patience:
            break
        started = time.perf_counter()
        model.train()
        for overflow in range(args.oom_retries + 1):
            step_rng = rng_state()
            trained_row = _node_operation(model, data, masks, args, policy, dtype, optimizer, scaler)
            scaler.unscale_(optimizer)
            norm = grad_norm(model.encoder)
            if not math.isfinite(norm) and dtype != torch.float16:
                raise FloatingPointError('Non-finite node encoder gradients')
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= before:
                break
            if overflow == args.oom_retries:
                raise FloatingPointError('Repeated node classification FP16 overflow')
            clear_memory(optimizer, args)
            restore_rng(step_rng)
        train_allocated, train_reserved = memory_peaks(args)
        model.eval()
        current = _node_operation(model, data, masks, args, policy, dtype)
        allocated, reserved = memory_peaks(args)
        row = {'epoch': epoch, 'seed': seed, 'input_mode': 'Structure-only', **current,
               'train_eval_loss': current['train_loss'], 'train_loss': trained_row['train_loss'],
               'encoder_grad_norm': norm, 'lr': optimizer.param_groups[0]['lr'],
               'peak_gpu_memory_mb': max(train_allocated, allocated) / 1024 ** 2,
               'peak_gpu_reserved_memory_mb': max(train_reserved, reserved) / 1024 ** 2,
               'oom_retries': policy.state.oom_retries}
        improved = (current['val_acc'], -current['val_loss']) > (best_acc, -best_loss)
        if improved:
            best_acc, best_loss, best_epoch, bad_epochs = current['val_acc'], current['val_loss'], epoch, 0
            best_state = cpu_state(model)
        else:
            bad_epochs += 1
        scheduler.step()
        synchronize(args)
        elapsed += time.perf_counter() - started
        history.append(row)
        atomic_save({'config': config, 'model_state_dict': cpu_state(model), 'best_state': best_state,
                     'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                     'scaler_state_dict': scaler.state_dict(), 'history': history, 'best_epoch': best_epoch,
                     'best_acc': best_acc, 'best_loss': best_loss, 'bad_epochs': bad_epochs, 'elapsed': elapsed,
                     'rng_state': rng_state(), 'micro_batch_state': policy.state_dict(),
                     'masks': {key: value.cpu() for key, value in masks.items()}}, last_path)
        write_history(directory / 'training.csv', history)
        if epoch == 1 or epoch % args.display_step == 0:
            print(f'[Structure-only {variant}/seed_{seed}] epoch={epoch} val_acc={current["val_acc"]:.2f}% '
                  f'node_var={current["node_repr_variance"]:.5f} allocated={row["peak_gpu_memory_mb"]:.0f}MiB '
                  f'reserved={row["peak_gpu_reserved_memory_mb"]:.0f}MiB')
    if best_state is None:
        raise RuntimeError('No node classification epoch completed')
    # Test labels are read only after validation-only model selection finishes.
    model.load_state_dict(best_state, strict=True)
    model.eval()
    best = _node_operation(model, data, masks, args, policy, dtype, splits=('train', 'valid', 'test'))
    counts = model.encoder.parameter_counts() | {'total': sum(p.numel() for p in model.parameters())}
    atomic_save({'kind': 'hirgraphbert_node_classifier', 'input_mode': 'Structure-only', 'seed': seed,
                 'model_state_dict': best_state, 'num_classes': data.num_classes, 'model_config': model.model_config,
                 'config': config, 'dataset': args.dname, 'best_epoch': best_epoch,
                 'encoder_load_report': report, 'masks': {key: value.cpu() for key, value in masks.items()}},
                directory / 'best_checkpoint.pt')
    result = {'completed': True, 'config': config, 'dataset': args.dname, 'variant': variant, 'seed': seed,
              'model': 'HIR-GraphBERT', 'input_mode': 'Structure-only', 'best_epoch': best_epoch,
              'last_epoch': len(history), **best, 'parameter_counts': counts, 'parameter_count': counts['total'],
              'encoder_load_report': report, 'train_time': elapsed, 'accuracy_unit': 'percent',
              'encoder_grad_norm_mean': sum(row['encoder_grad_norm'] for row in history) / len(history),
              'peak_gpu_memory_mb': max(row['peak_gpu_memory_mb'] for row in history),
              'peak_gpu_reserved_memory_mb': max(row['peak_gpu_reserved_memory_mb'] for row in history)}
    save_json(result, directory / 'metrics.json')
    print(f'[Structure-only {variant}/seed_{seed}] best_epoch={best_epoch} test_acc={best["test_acc"]:.2f}%')
    return result
