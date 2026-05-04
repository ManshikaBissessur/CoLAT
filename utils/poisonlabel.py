import numpy as np
import torch
import pickle as pkl
import os


def _find_pkl_path(base_path, filename):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, base_path, filename)


def _load_poisoned_labels(dataset_name, split_idx, poison_pct, attack):
    if attack == 'ntk':
        filename  = f"{dataset_name}_ntk_cv_poisoned_labels.pkl"
        base_path = "ntk_attack"
    elif attack == 'smulti':
        filename  = f"{dataset_name}_sgcmulti_cv_poisoned_labels.pkl"
        base_path = "smulti_neurips_cv"
    elif attack == 'lafak':
        filename  = f"{dataset_name}_lafak_random_cv_poisoned_labels.pkl"
        base_path = "lafak_random_cv_labels"
    else:
        raise ValueError(f"Unsupported attack: '{attack}'. Choose from ntk, smulti, lafak.")

    pkl_path = _find_pkl_path(base_path, filename)
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"PKL file not found: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        data = pkl.load(f)

    split_key  = f"split_{split_idx}"
    poison_key = f"{int(round(poison_pct))}_percent_poison"
    if split_key not in data:
        raise KeyError(f"Split key '{split_key}' not found in {pkl_path}")
    if poison_key not in data[split_key]:
        raise KeyError(f"Poison key '{poison_key}' not found under '{split_key}' in {pkl_path}. "
                       f"Available: {list(data[split_key].keys())}")

    poisoned = np.asarray(data[split_key][poison_key])
    if poisoned.ndim > 1:
        poisoned = poisoned.argmax(axis=1)
    return poisoned


def label_process(labels, features, n_classes, poison_type, poison_rate, random_seed=5,
                  debug=True, dataset_name=None, split_idx=0):
    poison_pct = int(round(poison_rate * 100)) if poison_rate <= 1 else int(round(poison_rate))
    poisoned = _load_poisoned_labels(dataset_name, split_idx, poison_pct, poison_type)

    labels_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else np.asarray(labels)
    poisoned  = poisoned.astype(np.int64)
    noisy_labels  = torch.tensor(poisoned, dtype=torch.long).to(
        labels.device if hasattr(labels, 'device') else 'cpu')
    modified_mask = np.where(noisy_labels.cpu().numpy() != labels_np)[0]

    if debug:
        print(f"Attack={poison_type}, poison={poison_pct}%, modified={len(modified_mask)} nodes")
    return noisy_labels, modified_mask