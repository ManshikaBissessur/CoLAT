import torch
import pickle as pkl
import numpy as np
import scipy.sparse as sp
from types import SimpleNamespace
from utils.functional import normalize
from torch_geometric.utils import degree
import os


class Dataset:
    def __init__(self, data, feat_norm=False, adj_norm=False, verbose=True, path='./data/',
                 train_size=None, val_size=None, test_size=None,
                 train_percent=None, val_percent=None, test_percent=None,
                 train_examples_per_class=None, val_examples_per_class=None, test_examples_per_class=None,
                 add_self_loop=True, split_type='default', from_npz=False, device='cuda:0'):
        self.name = data
        self.path = path
        self.device = torch.device(device)
        self.split_type = split_type

        self.prepare_data(data, feat_norm)
        self.feats = self.feats.to(torch.float)
        self.split_data(verbose)

        if add_self_loop:
            self.adj = self.adj + torch.eye(self.adj.shape[0], device=self.adj.device).to_sparse()
        if adj_norm:
            self.adj = normalize(self.adj, add_loop=False)
        self.adj = self.adj.coalesce()

    def prepare_data(self, ds_name, feat_norm):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pkl_path = os.path.join(repo_root, 'small_val_random', f'{ds_name}.pkl')
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"Dataset PKL not found: {pkl_path}\n"
                f"Place it in small_val_random/{ds_name}.pkl under the repo root."
            )

        with open(pkl_path, 'rb') as f:
            data = pkl.load(f)
        print(f"Loaded dataset '{ds_name}' from {pkl_path}")

        features = data.get('X')
        labels   = data.get('labels')
        adj      = data.get('sym_adj')

        if not sp.isspmatrix_csr(adj):
            adj = adj.tocsr()
        adj = adj.maximum(adj.T).tocsr()
        adj.setdiag(0)
        adj.eliminate_zeros()

        coo = adj.tocoo()
        n_nodes = features.shape[0]

        self.feats = torch.tensor(features, dtype=torch.float32).to(self.device)
        self.labels = torch.tensor(labels, dtype=torch.int64)
        if self.labels.dim() > 1:
            self.labels = torch.argmax(self.labels, dim=1)
        self.labels = self.labels.to(self.device)
        self.n_nodes   = n_nodes
        self.dim_feats = int(self.feats.shape[1]) if self.feats.ndim > 1 else 1
        self.n_classes = int(self.labels.max().item()) + 1

        indices = np.vstack((coo.row, coo.col))
        i = torch.tensor(indices, dtype=torch.int64)
        v = torch.ones(indices.shape[1], dtype=torch.float32)
        self.adj = torch.sparse_coo_tensor(i, v, [n_nodes, n_nodes]).to(self.device).coalesce()
        self.n_edges = int(self.adj.indices().shape[1] / 2)

        split = data.get('split_0') or data.get('split')
        if split is not None:
            train_ids = split.get('train_ids', [])
            val_ids   = split.get('val_ids',   [])
            test_ids  = split.get('test_ids',  [])
        else:
            perm    = np.random.permutation(n_nodes)
            n_train = int(0.1 * n_nodes)
            n_val   = int(0.1 * n_nodes)
            train_ids = perm[:n_train].tolist()
            val_ids   = perm[n_train:n_train + n_val].tolist()
            test_ids  = perm[n_train + n_val:].tolist()
            print("  No split info found in PKL, falling back to random split")

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask   = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask  = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[train_ids] = True
        val_mask[val_ids]     = True
        test_mask[test_ids]   = True

        self.g = SimpleNamespace(train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

        if feat_norm:
            self.feats = normalize(self.feats, style='row')

        d = degree(self.adj.indices()[0], self.n_nodes)
        self.ave_degree = float(torch.mean(d))

        print("----Data statistics------\n"
              f"  Name: {self.name}  Nodes: {self.n_nodes}  "
              f"Edges: {self.n_edges}  Classes: {self.n_classes}  "
              f"Ave degree: {self.ave_degree:.2f}")

    def split_data(self, verbose=True):
        if self.split_type != 'default':
            raise ValueError(f"Only split_type='default' is supported. Got: {self.split_type}")
        if not hasattr(self.g, 'train_mask'):
            raise RuntimeError(f"Dataset '{self.name}' has no default split.")

        self.train_masks = torch.nonzero(self.g.train_mask, as_tuple=False).squeeze().numpy()
        self.val_masks   = torch.nonzero(self.g.val_mask,   as_tuple=False).squeeze().numpy()
        self.test_masks  = torch.nonzero(self.g.test_mask,  as_tuple=False).squeeze().numpy()

        if verbose:
            print(f"  Train: {len(self.train_masks)}  "
                  f"Val: {len(self.val_masks)}  "
                  f"Test: {len(self.test_masks)}")