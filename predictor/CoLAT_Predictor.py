from predictor.Base_Predictor import Predictor
from predictor.module.CoLAT import ContrastiveGNN, contrastive_loss_and_noise_score
from predictor.module.GNNs import build_backbone
import time
import torch
import torch.nn.functional as F
from copy import deepcopy

try:
    import wandb as _wandb
except Exception:
    _wandb = None


class CoLAT_Predictor(Predictor):
    def __init__(self, conf, data, device="cuda:0"):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        self.model_mi = ContrastiveGNN(
            in_channels=conf.model["n_feat"],
            hidden_channels=conf.model["n_hidden"],
            n_layers=conf.model["n_layer"],
            dropout=conf.model["dropout"],
            norm_info=conf.model["norm_info"],
            act=conf.model["act"],
            input_layer=conf.model["input_layer"],
            output_layer=conf.model["output_layer"],
            backbone=conf.model.get("backbone", "gcn")).to(self.device)

        self.model = build_backbone(
            conf.model.get("backbone", "gcn"),
            in_channels=conf.model["n_feat"],
            hidden_channels=conf.model["n_hidden"],
            out_channels=conf.model["n_classes"],
            n_layers=conf.model["n_layer"],
            dropout=conf.model["dropout"],
            norm_info=conf.model["norm_info"],
            act=conf.model["act"],
            input_layer=conf.model["input_layer"],
            output_layer=conf.model["output_layer"]).to(self.device)

        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=conf.training["lr"],
            weight_decay=conf.training["weight_decay"])

        self.optim_mi = torch.optim.Adam(
            self.model_mi.parameters(),
            lr=conf.training["lr"],
            weight_decay=conf.training["weight_decay"])

    def get_prediction(self, features, adj, label=None, mask=None):
        output = F.log_softmax(self.model(features, adj), dim=1)
        loss, acc = None, None
        if (label is not None) and (mask is not None):
            loss = F.nll_loss(output[mask], label[mask])
            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
        return output, loss, acc

    def _update_model_mi(self, features, adj, working_labels):
        self.model_mi.train()
        for _ in range(self.conf.training["mi_steps"]):
            self.optim_mi.zero_grad()
            h = self.model_mi(features, adj)
            loss, noise_score = contrastive_loss_and_noise_score(
                h, self.edge_index, working_labels, self.n_nodes,
                margin=self.conf.training.get("margin", 0.5),
                node_degree=self.node_degree_1hop)
            loss.backward()
            self.optim_mi.step()
        return noise_score, loss.item()

    def _compute_flip_target(self, features, adj, working_labels):
        self.model.eval()
        with torch.no_grad():
            logp = F.log_softmax(self.model(features, adj), dim=1)
        self.model.train()
        idx = torch.arange(self.n_nodes, device=self.device)
        logp_masked = logp.clone()
        logp_masked[idx, working_labels] = float("inf")
        return logp_masked.argmin(dim=1)

    def _flip_labels(self, working_labels, noise_score, flip_target):
        budget = max(1, int(len(self.train_mask) * self.conf.training["flip_budget"] / 100))
        train_idx = torch.as_tensor(self.train_mask, dtype=torch.long).to(self.device)
        _, top_local = noise_score[train_idx].topk(budget)
        selected = train_idx[top_local]
        new_labels = working_labels.clone()
        new_labels[selected] = flip_target[selected]
        return new_labels

    def train(self):
        features, adj = self.feats, self.adj
        working_labels = self.noisy_label.clone()

        self.node_degree_1hop = torch.zeros(self.n_nodes, device=self.device)
        self.node_degree_1hop.scatter_add_(
            0, self.edge_index[0],
            torch.ones(self.edge_index.shape[1], device=self.device))

        for _ in range(self.conf.training["mi_pretrain_epochs"]):
            self._update_model_mi(features, adj, working_labels)
        noise_score, _ = self._update_model_mi(features, adj, working_labels)

        mi_loss = None
        for epoch in range(self.conf.training["n_epochs"]):
            improve = ""
            t0 = time.time()
            self.model.train()

            self.optim.zero_grad()
            output = F.log_softmax(self.model(features, adj), dim=1)
            loss_train = F.nll_loss(output[self.train_mask], working_labels[self.train_mask])
            loss_train.backward()
            self.optim.step()
            acc_train = self.metric(working_labels[self.train_mask].cpu().numpy(),
                                    output[self.train_mask].detach().cpu().numpy())

            if epoch >= self.conf.training["start_epoch"]:
                flip_target = self._compute_flip_target(features, adj, working_labels)
                working_labels = self._flip_labels(working_labels, noise_score, flip_target)

            if epoch % self.conf.training["mi_update_interval"] == 0:
                noise_score, mi_loss = self._update_model_mi(features, adj, working_labels)

            loss_val, acc_val = self.evaluate(self.noisy_label, self.val_mask)
            if _wandb is not None and _wandb.run is not None:
                log_dict = {
                    "epoch": epoch + 1,
                    "main/loss_train": loss_train.item(),
                    "main/acc_train": acc_train,
                    "main/loss_val": loss_val.item(),
                    "main/acc_val": acc_val,
                }
                if mi_loss is not None:
                    log_dict["mi/loss"] = mi_loss
                _wandb.log(log_dict)

            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag:
                improve = "*"
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result["valid"] = acc_val
                self.result["train"] = acc_train
                self.weights = deepcopy(self.model.state_dict())
            elif flag_earlystop:
                break

            if self.conf.training["debug"]:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                    epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

        loss_test, acc_test = self.test(self.test_mask)
        self.result["test"] = acc_test
        self.final_working_labels = working_labels
        if self.conf.training["debug"]:
            print("Optimization Finished!")
            print("Time(s): {:.4f}".format(self.total_time))
            print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result
