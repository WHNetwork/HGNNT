"""Small per-seed HIRBERT artifacts alongside the existing result CSVs."""

import csv
import json
from pathlib import Path

import torch


def save_hirbert_run(model, data, masks, history, metrics, train_time, args, run_id, seed):
    directory = Path(args.results_dir) / 'hirbert' / args.dname / run_id / f'seed_{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'training.csv').open('w', newline='', encoding='utf-8') as file:
        fields = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'test_acc']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    with (directory / 'compilation.json').open('w', encoding='utf-8') as file:
        json.dump(model.compilation_stats, file, indent=2)
    summary = {
        'seed': seed, 'run_id': run_id, 'dataset': args.dname,
        'metrics': {key: [float(value) for value in values] for key, values in metrics.items()},
        'train_time': train_time,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'model_config': model.model_config,
        'lr': args.lr, 'wd': args.wd, 'epochs': args.epochs,
    }
    with (directory / 'metrics.json').open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)
    graph = {key: getattr(data, key).detach().cpu() for key in ('x', 'y', 'hyperedge_index')}
    graph.update(num_nodes=data.x.shape[0], num_hyperedges=len(model.program.hyperedges),
                 num_features=model.num_features, num_classes=model.num_classes)
    torch.save({
        **summary,
        'num_features': model.num_features, 'num_classes': model.num_classes,
        'model_state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()},
        'graph': graph,
        'masks': {key: value.detach().cpu() for key, value in masks.items()},
        'compilation_stats': model.compilation_stats,
    }, directory / 'checkpoint.pt')
    return directory
