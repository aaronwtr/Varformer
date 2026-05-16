"""VarformerLightningModule: Lightning training wrapper for Varformer.

Renamed from MultiModalLightningTargetIdentifier (src/models/lightning.py).

Key changes from the original:
  - Metrics (Accuracy, AUROC, SpearmanCorrCoef, Recall, Precision, F1Score,
    AveragePrecision) have been moved from the nn.Module into this class.
  - _log() now references self.<metric> instead of self.model.<metric>.
  - model.varformer is now a VariantEncoder (no VarformerTargetIdentifier wrapper).

Deleted legacy classes (no replacement):
  - BaseLightningTargetIdentifier
  - MLPLightningTargetIdentifier
  - VarformerLightningTargetIdentifier
  - ShardedVarformerLightningTargetIdentifier
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, ExponentialLR, ReduceLROnPlateau
from torchmetrics import Accuracy, AUROC, SpearmanCorrCoef, Recall, Precision, F1Score, AveragePrecision

from varformer.models.varformer import Varformer


class VarformerLightningModule(pl.LightningModule):
    """Lightning module for Varformer training, validation, and inference."""

    def __init__(
        self,
        config,
        num_samples_per_class,
        num_features_gc: int,
        num_features_go: int,
        num_mutations: int,
        max_seq_len: int,
        num_genes: int,
        class_prior,
        use_pvc: bool = True,
    ):
        self.save_hyperparameters()
        super().__init__()

        self.hyperparams = config["hyperparameters"]
        self.use_pvc = use_pvc
        self.model = Varformer(
            config=config,
            num_features_gc=num_features_gc,
            num_features_go=num_features_go,
            num_mutations=num_mutations,
            max_seq_len=max_seq_len,
            num_genes=num_genes,
            use_pvc=use_pvc,
        )
        self.pi = class_prior
        self.val_step_probas = []

        # Metrics (moved from nn.Module into the LightningModule)
        threshold = float(self.hyperparams["threshold"])
        self.acc = Accuracy(task="binary", threshold=threshold)
        self.auroc = AUROC(task="binary")
        self.recall = Recall(task="binary", threshold=threshold)
        self.precision = Precision(task="binary", threshold=threshold)
        self.auprc = AveragePrecision(task="binary")
        self.f1 = F1Score(task="binary", threshold=threshold)
        self.spearman = SpearmanCorrCoef()

    def forward(self, x, mask=None):
        return self.model(x, mask)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        features = batch
        return self._common_step(features, batch_idx, "predict")

    def _common_step(self, batch, batch_idx, step_type):
        gene_names = None
        pvc_labels = None
        if self.trainer.sanity_checking:
            return None
        else:
            if self.use_pvc:
                pvc_labels = batch["pvc"]["labels"]
                test_source = batch["pvc"]["test_source"][0]
                gene_names = batch["pvc"]["gene_name"]
                mask = batch["pvc"]["mask"]
            else:
                pvc_labels = batch["gc"][1]
                test_source = batch["gc"][2] if len(batch["gc"]) > 2 else None
                gene_names = None
                mask = None

            model_input = {
                "gc": batch["gc"],
                "go": batch["go"],
            }
            if self.use_pvc:
                model_input["pvc"] = batch["pvc"]

            if self.hyperparams["return_attn"]:
                logits, probas, bin_preds, z_var, attn_weights = self.model(model_input, mask)
            else:
                logits, probas, bin_preds, z_var = self.model(model_input, mask)

            labels = pvc_labels
            eps = 1e-8

            if step_type == "train":
                if self.hyperparams.get("pusb", False):
                    pos_mask = labels == 1
                    unlabeled_mask = labels == 0

                    if pos_mask.sum() > 0:
                        pos_mean_log = torch.mean(torch.log(probas[pos_mask] + eps))
                        pos_mean_log_1 = torch.mean(torch.log(1 - probas[pos_mask] + eps))
                    else:
                        pos_mean_log = torch.tensor(0.0, device=self.device)
                        pos_mean_log_1 = torch.tensor(0.0, device=self.device)
                    if unlabeled_mask.sum() > 0:
                        unlabeled_mean_log_1 = torch.mean(torch.log(1 - probas[unlabeled_mask] + eps))
                    else:
                        unlabeled_mean_log_1 = torch.tensor(0.0, device=self.device)

                    loss_positive = -self.pi * pos_mean_log
                    loss_unlabeled_component = self.pi * pos_mean_log_1 - unlabeled_mean_log_1
                    loss = loss_positive + torch.relu(loss_unlabeled_component)
                else:
                    beta = self.beta
                    epsilon = 1e-8
                    n0 = self.num_samples_per_class[0]
                    n1 = self.num_samples_per_class[1]
                    w0 = (1.0 - beta) / (1.0 - beta ** n0 + epsilon)
                    w1 = (1.0 - beta) / (1.0 - beta ** n1 + epsilon)
                    norm_factor = 2.0 / (w0 + w1)
                    w0 *= norm_factor
                    w1 *= norm_factor
                    weight_val_0 = torch.tensor(w0, device=self.device, dtype=logits.dtype)
                    weight_val_1 = torch.tensor(w1, device=self.device, dtype=logits.dtype)
                    class_weight = torch.where(labels == 0, weight_val_0, weight_val_1)
                    loss = F.binary_cross_entropy_with_logits(logits, labels.float(), weight=class_weight)

                self._log(labels, step_type, loss, bin_preds, probas)

            elif step_type == "val":
                if logits.shape != labels.shape:
                    if logits.dim() == 0:
                        logits = logits.unsqueeze(0)
                    if labels.dim() == 0:
                        labels = labels.unsqueeze(0)
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())
                self._log(labels, step_type, loss, bin_preds, probas)
                self.val_step_probas.append(probas.detach().cpu())

            elif step_type == "test":
                if "best_threshold" in self.hyperparams:
                    bin_preds = (probas > self.hyperparams["best_threshold"]).float()
                loss = None
                self._log(labels, step_type, loss, bin_preds, probas, test_source=test_source)
                return loss

            elif step_type == "predict":
                output = {}
                if self.hyperparams["return_attn"]:
                    for i, gene_name in enumerate(gene_names):
                        output[gene_name] = {
                            "prediction": probas[i].detach().cpu().item(),
                            "classification": bin_preds[i].detach().cpu().item(),
                            "z_var": z_var[i].detach().cpu().to(torch.float32).numpy(),
                            "attn_weights": attn_weights[i].detach().cpu().to(torch.float32).numpy(),
                        }
                else:
                    for i, gene_name in enumerate(gene_names):
                        output[gene_name] = {
                            "prediction": probas[i].detach().cpu().item(),
                            "classification": bin_preds[i].detach().cpu().item(),
                            "z_var": z_var[i].detach().cpu().to(torch.float32).numpy(),
                        }
                return output
            return loss

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return None
        all_probs = torch.cat(self.val_step_probas, dim=0).to(torch.float32).numpy()
        new_threshold = np.quantile(all_probs, 1 - self.pi)
        self.model.hyperparams["threshold"] = new_threshold
        self.log("val_threshold", new_threshold)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")

    def _log(self, labels, step_type, loss, bin_preds, probas, pos_loss=None, neg_loss=None, test_source=None):
        if bin_preds.shape != labels.shape:
            if bin_preds.dim() == 0:
                bin_preds = bin_preds.unsqueeze(0)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
        if probas.shape != labels.shape:
            if probas.dim() == 0:
                probas = probas.unsqueeze(0)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
        if step_type in ["train", "val"]:
            if pos_loss is not None:
                self.log(f"{step_type}_pos_loss", pos_loss, batch_size=labels.shape[0])
            if neg_loss is not None:
                self.log(f"{step_type}_neg_loss", neg_loss, batch_size=labels.shape[0])

            self.log(f"{step_type}_loss", loss, batch_size=labels.shape[0])
            self.log(f"{step_type}_acc", self.acc(bin_preds, labels), batch_size=labels.shape[0])
            self.log(f"{step_type}_auroc", self.auroc(probas, labels.int()), batch_size=labels.shape[0])
            self.model_spearman = self.spearman(probas, labels.float())
            self.log(f"{step_type}_spearman", self.model_spearman, batch_size=labels.shape[0])
            self.log(f"{step_type}_recall", self.recall(bin_preds, labels.long()), batch_size=labels.shape[0])
            self.log(f"{step_type}_precision", self.precision(bin_preds, labels.long()), batch_size=labels.shape[0])
            self.log(f"{step_type}_f1", self.f1(bin_preds, labels.long()), batch_size=labels.shape[0])
            self.log(f"{step_type}_auprc", self.auprc(probas, labels.long()), batch_size=labels.shape[0])
        else:
            self.log(f"{step_type}_acc_{test_source}", self.acc(bin_preds, labels), batch_size=labels.shape[0])
            self.log(f"{step_type}_auroc_{test_source}", self.auroc(bin_preds, labels.int()), batch_size=labels.shape[0])
            self.log(f"{step_type}_spearman_{test_source}", self.spearman(probas, labels.float()), batch_size=labels.shape[0])
            self.log(f"{step_type}_recall_{test_source}", self.recall(bin_preds, labels.long()), batch_size=labels.shape[0])
            self.log(f"{step_type}_precision_{test_source}", self.precision(bin_preds, labels.long()), batch_size=labels.shape[0])
            self.log(f"{step_type}_f1_{test_source}", self.f1(bin_preds, labels.long()), batch_size=labels.shape[0])
            self.log(f"{step_type}_auprc_{test_source}", self.auprc(probas, labels.long()), batch_size=labels.shape[0])

            logger_type = self.logger.__class__.__name__
            if logger_type == "WandbLogger":
                self.logger.experiment.log({
                    f"{step_type}_{test_source}_predictions": bin_preds.detach().cpu().numpy(),
                    f"{step_type}_{test_source}_probas": probas.to(dtype=torch.float32).detach().cpu().numpy(),
                    f"{step_type}_{test_source}_labels": labels.to(dtype=torch.float32).detach().cpu().numpy(),
                })

    def configure_optimizers(self):
        weight_decay = float(self.hyperparams.get("weight_decay", 0))
        if self.hyperparams["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=float(self.hyperparams["lr_start"]), weight_decay=weight_decay
            )
        elif self.hyperparams["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=float(self.hyperparams["lr_start"]), weight_decay=weight_decay
            )
        elif self.hyperparams["optimizer"] == "RMSprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(), lr=float(self.hyperparams["lr_start"]), weight_decay=weight_decay
            )
        elif self.hyperparams["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=float(self.hyperparams["lr_start"]), weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Optimizer {self.hyperparams['optimizer']} not recognized.")

        if self.hyperparams["scheduler"] == "CosineAnnealingLR":
            lr_scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=int(self.hyperparams["T0"]), eta_min=float(self.hyperparams["lr_end"])
            )
        elif self.hyperparams["scheduler"] == "StepLR":
            lr_scheduler = StepLR(
                optimizer,
                step_size=int(self.hyperparams["step_size"]),
                gamma=float(self.hyperparams["gamma"]),
            )
        elif self.hyperparams["scheduler"] == "ExponentialLR":
            lr_scheduler = ExponentialLR(optimizer, gamma=float(self.hyperparams["gamma"]))
        elif self.hyperparams["scheduler"] == "ReduceLROnPlateau":
            lr_scheduler = ReduceLROnPlateau(
                optimizer,
                factor=float(self.hyperparams["factor"]),
                patience=int(self.hyperparams["patience"]),
            )
        else:
            raise ValueError(f"Scheduler {self.hyperparams['scheduler']} not recognized.")

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def initialize_weights(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        initial_weights = {}

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                initial_weights[name + ".weight"] = module.weight.detach().clone()
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                    initial_weights[name + ".bias"] = module.bias.detach().clone()

        return initial_weights
