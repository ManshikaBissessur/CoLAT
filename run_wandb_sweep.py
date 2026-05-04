import argparse
import os
import json
import subprocess
import sys

try:
    import wandb
except Exception as e:
    print('Error: wandb is required. Install with `pip install wandb`')
    raise


def convert_search_space_to_wandb(sspec: dict):
    params = {}
    for k, v in sspec.items():
        _type = v.get('_type')
        _value = v.get('_value')
        if _type == 'choice':
            params[k] = {'values': list(_value)}
        elif _type == 'uniform':
            lo, hi = _value
            params[k] = {'min': float(lo), 'max': float(hi)}
        elif _type in ('quniform', 'loguniform', 'randint'):
            if isinstance(_value, list) and len(_value) >= 2:
                params[k] = {'min': float(_value[0]), 'max': float(_value[1])}
            else:
                params[k] = {'values': list(_value) if isinstance(_value, list) else [_value]}
        else:
            try:
                params[k] = {'values': list(_value)}
            except Exception:
                params[k] = {'value': _value}
    return params


def build_sweep(dataset, poison_type, poison_rate, sweep_method='random', metric_name='test',
                metric_goal='maximize', seed=None, device='cuda:0', tags=None, backbone=None):
    ss_path = os.path.join('config', '_search_space', f"cagnn_{dataset}.json")
    if not os.path.exists(ss_path):
        ss_path = os.path.join('config', '_search_space', 'cagnn.json')
    if not os.path.exists(ss_path):
        raise FileNotFoundError(f"Search space JSON not found: {ss_path}")
    with open(ss_path, 'r') as f:
        sspec = json.load(f)
    params = convert_search_space_to_wandb(sspec)
    params['dataset'] = {'value': dataset}
    params['poison_type'] = {'value': poison_type}
    params['poison_rate'] = {'value': float(poison_rate)}
    params['device'] = {'value': device}
    if seed is not None:
        params['seed'] = {'value': int(seed)}

    cmd = [sys.executable, '${program}', '--dataset', dataset,
           '--poison_type', poison_type, '--poison_rate', str(poison_rate), '--device', device]
    if seed is not None:
        cmd.extend(['--seed', str(seed)])
    if tags:
        cmd.extend(['--tags', ','.join(tags)])
    if backbone is not None:
        cmd.extend(['--backbone', backbone])

    return {
        'program': 'run.py',
        'method': sweep_method,
        'metric': {'name': metric_name, 'goal': metric_goal},
        'parameters': params,
        'command': cmd,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--poison_type', default='ntk', choices=['ntk', 'smulti', 'lafak'])
    parser.add_argument('--poison_rate', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--project', default='colat')
    parser.add_argument('--entity', default=None)
    parser.add_argument('--sweep_method', default='random', choices=['random', 'grid', 'bayes'])
    parser.add_argument('--metric_name', default='test')
    parser.add_argument('--metric_goal', default='maximize', choices=['minimize', 'maximize'])
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--run_cap', type=int, default=None)
    parser.add_argument('--register_only', action='store_true')
    parser.add_argument('--tags', type=str, default=None)
    parser.add_argument('--backbone', type=str, default=None)
    args = parser.parse_args()

    _tags = [t.strip() for t in args.tags.split(',')] if args.tags else []
    sweep = build_sweep(args.dataset, args.poison_type, args.poison_rate, args.sweep_method,
                        args.metric_name, args.metric_goal, seed=args.seed, device=args.device,
                        tags=_tags, backbone=args.backbone)
    if args.run_cap is not None:
        sweep['run_cap'] = int(args.run_cap)

    sweep_id = wandb.sweep(sweep, project=args.project, entity=args.entity)
    print('Registered sweep id:', sweep_id)
    print('To run agents manually: wandb agent', f"{args.entity}/{args.project}/{sweep_id}")

    if not args.register_only:
        agent_cmd = [sys.executable, '-m', 'wandb', 'agent']
        agent_cmd.append(f"{args.entity}/{args.project}/{sweep_id}" if args.entity else f"{args.project}/{sweep_id}")
        agent_cmd.extend(['--count', str(args.count)])
        print(f"Launching agents: {' '.join(agent_cmd)}")
        subprocess.check_call(agent_cmd)


if __name__ == '__main__':
    main()