# CoLAT — Contrastive Label-flipping Adversarial Training

GNN node classification under label poisoning attacks.

## Method

Two models are trained jointly:

- **ContrastiveGNN** (`model_mi`) — encodes node embeddings and is trained with a margin-based contrastive loss over graph edges. Nodes whose neighborhood label agreement is low receive a high noise score.
- **Backbone GNN** (`model`) — the main classifier

Each epoch, the top-`flip_budget`% of training nodes by noise score have their labels flipped to the model's worst label prediction. The contrastive encoder is refreshed every `mi_update_interval` epochs.

## Datasets

| Dataset | Supported |
|---------|-----------|
| `cora_ml` | yes |
| `citeseer` | yes |
| `pubmed` | yes |

Poison types: `ntk`, `smulti`, `lafak`.

## Usage

```bash
source venv/bin/activate
pip install -r requirements.txt

python run.py \
  --dataset cora_ml \
  --poison_type ntk \
  --poison_rate 0.3 \
  --device cuda:0 \
  --num_runs 5 \
  --seed 3000
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `cora_ml` | Dataset name |
| `--poison_type` | `ntk` | Label poisoning attack type |
| `--poison_rate` | `0.3` | Fraction of labels corrupted |
| `--backbone` | `gcn` | GNN backbone (`gcn`, `gat`, `sage`) |
| `--num_runs` | `5` | Number of runs (results are averaged) |
| `--seed` | `3000` | Base random seed |
| `--device` | `cuda:0` | Device |

## Configuration

Per-dataset training configs live in `config/CoLAT/`. The hyperparameter search space is in `config/_search_space/CoLAT.json` — only `lr`, `weight_decay`, `margin`, and `flip_budget` are tuned.

## W&B

W&B logging is enabled automatically if `wandb` is installed. Disable it with:

```bash
NO_WANDB=1 python run.py ...
```

---

## :open_book: Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{colat,
      title={Contrastive Adversarial Training for Robust Graph Neural Networks under Label Poisoning},
      author={Bissessur, Manshika Charvi and Bal, Melis Ilayda and Muehlebach, Michael},
      year={2026},
      journal={Neural Information Processing Systems (NeurIPS)},
}
```


