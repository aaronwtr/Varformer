"""PyTorch Lightning wrapper around the Varformer architecture.

Owns training/val/test/predict steps, the PUL (nnPU) loss, metric logging,
optimizer + scheduler configuration. The underlying nn.Module is a `Varformer`
instance accessible as `self.model`.
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
                # Varformer uses nnPU (non-negative PU) loss; pusb must be enabled.
                if not self.hyperparams.get("pusb", False):
                    raise NotImplementedError(
                        "Varformer training requires hyperparameters.pusb=True (nnPU loss)."
                    )
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

                self._log_nnpu_diagnostics(
                    probas=probas,
                    logits=logits,
                    pos_mask=pos_mask,
                    unlabeled_mask=unlabeled_mask,
                    pos_mean_log=pos_mean_log,
                    pos_mean_log_1=pos_mean_log_1,
                    unlabeled_mean_log_1=unlabeled_mean_log_1,
                    loss_positive=loss_positive,
                    loss_unlabeled_component=loss_unlabeled_component,
                    loss=loss,
                    batch_size=labels.shape[0],
                )

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

    def _log(self, labels, step_type, loss, bin_preds, probas, test_source=None):
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
            self.log(f"{step_type}_loss", loss, batch_size=labels.shape[0])
            self.log(f"{step_type}_acc", self.acc(bin_preds, labels), batch_size=labels.shape[0])
            self.log(f"{step_type}_auroc", self.auroc(probas, labels.int()), batch_size=labels.shape[0])
            spearman_val = self.spearman(probas, labels.float())
            self.log(f"{step_type}_spearman", spearman_val, batch_size=labels.shape[0])
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

    def _log_nnpu_diagnostics(
        self,
        *,
        probas,
        logits,
        pos_mask,
        unlabeled_mask,
        pos_mean_log,
        pos_mean_log_1,
        unlabeled_mean_log_1,
        loss_positive,
        loss_unlabeled_component,
        loss,
        batch_size: int,
    ):
        """Log per-term scalars from the nnPU loss so NaN/Inf onset is locatable.

        Wandb traces show *where* the divergence starts (which intermediate
        first becomes non-finite), and the ranges of ``probas``/``logits`` tell
        us whether the underlying cause is a sigmoid saturating, an empty
        masked subset, or upstream gradient explosion. Cheap to compute and
        only adds scalar columns.
        """
        def _safe(x):
            return x.detach().float() if isinstance(x, torch.Tensor) else torch.tensor(float(x))

        def _isbad(x) -> float:
            t = _safe(x)
            return float((torch.isnan(t) | torch.isinf(t)).any().item())

        # Per-term non-finite flags (0.0 / 1.0).
        self.log("diag_nan_pos_mean_log",        _isbad(pos_mean_log),        batch_size=batch_size)
        self.log("diag_nan_pos_mean_log_1",      _isbad(pos_mean_log_1),      batch_size=batch_size)
        self.log("diag_nan_unl_mean_log_1",      _isbad(unlabeled_mean_log_1),batch_size=batch_size)
        self.log("diag_nan_loss_positive",       _isbad(loss_positive),       batch_size=batch_size)
        self.log("diag_nan_loss_unl_component",  _isbad(loss_unlabeled_component), batch_size=batch_size)
        self.log("diag_nan_total_loss",          _isbad(loss),                batch_size=batch_size)

        # Activation extremes (early-warning for sigmoid saturation).
        with torch.no_grad():
            self.log("diag_probas_min",  probas.detach().float().min().item(),  batch_size=batch_size)
            self.log("diag_probas_max",  probas.detach().float().max().item(),  batch_size=batch_size)
            self.log("diag_logits_absmax", logits.detach().float().abs().max().item(), batch_size=batch_size)

        # Mask occupancy — empty masks fall back to constant 0 tensors, which is
        # an instability vector worth seeing in wandb.
        self.log("diag_n_pos",  float(pos_mask.sum().item()),       batch_size=batch_size)
        self.log("diag_n_unl",  float(unlabeled_mask.sum().item()), batch_size=batch_size)

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

