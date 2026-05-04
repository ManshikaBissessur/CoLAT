import argparse
import os
import time
import torch
from utils.dataloader import Dataset
from utils.tools import load_conf, setup_seed
from utils.poisonlabel import label_process
from predictor.CoLAT_Predictor import CoLAT_Predictor

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '--data', type=str, default='cora_ml',
                    choices=['cora_ml', 'citeseer', 'pubmed'])
parser.add_argument('--poison_type', type=str, default='ntk',
                    choices=['ntk', 'smulti', 'lafak'])
parser.add_argument('--poison_rate', type=float, default=0.3)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=3000)
parser.add_argument('--num_runs', type=int, default=5)
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--weight_decay', type=float, default=None)
parser.add_argument('--backbone', type=str, default=None)
parser.add_argument('--split_idx', type=int, default=0)
args = parser.parse_args()


if __name__ == '__main__':
    print(args)
    data_conf = load_conf('./config/_dataset/' + args.dataset + '.yaml')

    if WANDB_AVAILABLE and os.environ.get('NO_WANDB', '0') != '1':
        try:
            wandb.init(project=os.environ.get('WANDB_PROJECT', 'colat'), config=vars(args), reinit=True)
            wb_seed = dict(wandb.config).get('seed')
            if wb_seed is not None:
                args.seed = int(wb_seed)
        except Exception as e:
            print('Warning: W&B init failed:', e)

    setup_seed(args.seed)

    model_conf = load_conf(None, 'CoLAT', args.dataset)

    if WANDB_AVAILABLE and os.environ.get('NO_WANDB', '0') != '1':
        try:
            for k, v in dict(wandb.config).items():
                if v is None or k in ['dataset', 'poison_type', 'poison_rate']:
                    continue
                if k == 'seed':
                    args.seed = int(v)
                    setup_seed(args.seed)
                elif k in model_conf.training:
                    model_conf.training[k] = v
                else:
                    model_conf.model[k] = v
        except Exception as e:
            print('Warning: failed to merge W&B config:', e)

    if args.lr is not None:
        model_conf.training['lr'] = args.lr
    if args.weight_decay is not None:
        model_conf.training['weight_decay'] = args.weight_decay
    if args.backbone is not None:
        model_conf.model['backbone'] = args.backbone

    model_conf.training['debug'] = True

    base_seed = args.seed
    test_scores, val_scores, run_times = [], [], []

    for run_idx in range(args.num_runs):
        current_seed = base_seed + run_idx
        setup_seed(current_seed)
        print(f"\n=== Run {run_idx+1}/{args.num_runs} seed={current_seed} ===")

        data = Dataset(args.dataset,
                       feat_norm=data_conf.norm['feat_norm'],
                       adj_norm=data_conf.norm['adj_norm'],
                       add_self_loop=data_conf.modify['add_self_loop'],
                       split_type=data_conf.split['split_type'],
                       device=args.device)

        data.noisy_label, modified_mask = label_process(labels=data.labels,
                                                        features=data.feats,
                                                        n_classes=data.n_classes,
                                                        poison_type=args.poison_type, poison_rate=args.poison_rate,
                                                        random_seed=current_seed, debug=(run_idx == 0),
                                                        dataset_name=args.dataset,
                                                        split_idx=args.split_idx)

        model_conf.model['n_feat'] = data.dim_feats
        model_conf.model['n_classes'] = data.n_classes

        t0 = time.time()
        predictor = CoLAT_Predictor(model_conf, data, args.device)
        predictor.modified_mask = modified_mask
        result = predictor.train()
        run_time = time.time() - t0
        run_times.append(run_time)

        test_scores.append(float(result['test']))
        val_scores.append(float(result.get('valid', 0.0)))
        print(f"Run {run_idx+1} — Test: {result['test']:.4f}, Val: {result.get('valid', 0.0):.4f}, Time: {run_time:.2f}s")

    avg_test = sum(test_scores) / len(test_scores)
    avg_val  = sum(val_scores)  / len(val_scores)
    avg_time = sum(run_times)   / len(run_times)
    spread   = (max(test_scores) - min(test_scores)) / 2
    print(f"\n=== Average over {args.num_runs} runs ===")
    print(f"Test : {avg_test:.4f} ± {spread:.4f}")
    print(f"Val  : {avg_val:.4f}")
    print(f"Time : {avg_time:.2f}s / run")

    if WANDB_AVAILABLE and os.environ.get('NO_WANDB', '0') != '1':
        try:
            wandb.log({'test': avg_test, 'val': avg_val, 'test_std': spread,
                       'avg_time_per_run': avg_time})
            wandb.finish()
        except Exception as e:
            print('Warning: W&B log failed:', e)