import torch
import numpy as np


def accuracy(labels, logits):
    return np.sum(logits.argmax(1) == labels) / len(labels)


def normalize(mx, style='symmetric', add_loop=True):
    """Normalize a sparse adj (symmetric GCN-style) or dense feature matrix (row-wise)."""
    if style == 'symmetric':
        adj = mx.coalesce()
        if add_loop:
            adj = adj + torch.eye(adj.shape[0], device=adj.device).to_sparse()
            adj = adj.coalesce()
        inv_sqrt = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()) + 1e-12)
        D = inv_sqrt[adj.indices()[0]] * inv_sqrt[adj.indices()[1]]
        return torch.sparse_coo_tensor(adj.indices(), adj.values() * D, adj.size())
    elif style == 'row':
        r_inv = mx.sum(1).pow(-1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        return torch.diag(r_inv) @ mx
    else:
        raise KeyError(f"Unknown normalize style: {style}")
