import torch
import torch.nn as nn
import torch.nn.functional as F
from predictor.module.GNNs import build_backbone


class ContrastiveGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_layers=2, dropout=0.5,
                 norm_info=None, act='F.relu', input_layer=False, output_layer=False,
                 backbone='gcn'):
        super(ContrastiveGNN, self).__init__()
        self.gnn = build_backbone(backbone, in_channels=in_channels,
                                  hidden_channels=hidden_channels,
                                  out_channels=hidden_channels,
                                  n_layers=n_layers, dropout=dropout,
                                  norm_info=norm_info, act=act,
                                  input_layer=input_layer, output_layer=output_layer)

    def forward(self, x, adj):
        h = self.gnn(x, adj)
        return F.normalize(h, dim=1)


def contrastive_loss_and_noise_score(h, edge_index, labels, n_nodes, margin=0.5,
                                      node_degree=None):
    src, dst = edge_index
    not_self = src != dst
    src, dst = src[not_self], dst[not_self]

    cos_sim = (h[src] * h[dst]).sum(dim=1)
    same = (labels[src] == labels[dst]).float()

    loss_pos = same * (1.0 - cos_sim)
    loss_neg = (1.0 - same) * F.relu(cos_sim - margin)
    per_edge_loss = (loss_pos + loss_neg).detach()

    noise_score = torch.zeros(n_nodes, device=h.device)
    weight_sum  = torch.zeros(n_nodes, device=h.device)

    if node_degree is not None:
        w = 1.0 / (node_degree[src].clamp(min=1) * node_degree[dst].clamp(min=1)).sqrt()
    else:
        w = torch.ones(src.size(0), device=h.device)

    noise_score.scatter_add_(0, src, per_edge_loss * w)
    noise_score.scatter_add_(0, dst, per_edge_loss * w)
    weight_sum.scatter_add_(0, src, w)
    weight_sum.scatter_add_(0, dst, w)

    noise_score = noise_score / (weight_sum + 1e-8)
    lo, hi = noise_score.min(), noise_score.max()
    noise_score = (noise_score - lo) / (hi - lo + 1e-8)

    return (loss_pos + loss_neg).mean(), noise_score