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

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
    ExponentialLR,
    ReduceLROnPlateau,
)
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
        use_gc: bool = True,
        use_go: bool = True,
    ):
        self.save_hyperparameters()
        super().__init__()

        self.hyperparams = config["hyperparameters"]
        self.use_pvc = use_pvc
        self.use_gc = use_gc
        self.use_go = use_go
        self.model = Varformer(
            config=config,
            num_features_gc=num_features_gc,
            num_features_go=num_features_go,
            num_mutations=num_mutations,
            max_seq_len=max_seq_len,
            num_genes=num_genes,
            use_pvc=use_pvc,
            use_gc=use_gc,
            use_go=use_go,
        )
        self.observed_positive_rate = float(class_prior)
        self.pu_class_prior = float(
            self.hyperparams.get("pu_class_prior")
            if self.hyperparams.get("pu_class_prior") is not None
            else self.observed_positive_rate
        )
        self.threshold_positive_fraction = float(
            self.hyperparams.get("threshold_positive_fraction")
            if self.hyperparams.get("threshold_positive_fraction") is not None
            else self.observed_positive_rate
        )
        for name, value in (
            ("pu_class_prior", self.pu_class_prior),
            ("threshold_positive_fraction", self.threshold_positive_fraction),
        ):
            if not 0 < value < 1:
                raise ValueError(f"{name} must lie strictly between 0 and 1; got {value}.")
        # Backward-compatible attribute for old checkpoints/callers. New code
        # must use the semantically explicit names above.
        self.pi = self.pu_class_prior
        self.val_step_probas = []
        self.register_buffer(
            "_nnpu_active_batches", torch.zeros((), dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_nnpu_total_batches", torch.zeros((), dtype=torch.long), persistent=False
        )
        self.nnpu_correction_history = []
        self.variant_loss_history = []
        self._variant_active_batches = 0
        self._variant_total_batches = 0

        threshold = float(self.hyperparams["threshold"])
        self.register_buffer("decision_threshold", torch.tensor(threshold, dtype=torch.float32))
        self.acc = Accuracy(task="binary", threshold=threshold)
        self.auroc = AUROC(task="binary")
        self.recall = Recall(task="binary", threshold=threshold)
        self.precision = Precision(task="binary", threshold=threshold)
        self.auprc = AveragePrecision(task="binary")
        self.f1 = F1Score(task="binary", threshold=threshold)
        self.spearman = SpearmanCorrCoef()
        # The validation monitor must represent one correlation over every
        # validation gene. Logging per-batch scalar values from a stateful
        # metric causes Lightning to average running-prefix correlations and
        # can select the wrong checkpoint.
        self.val_spearman_epoch = SpearmanCorrCoef()

    def forward(self, x, mask=None):
        return self.model(x, mask)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        features = batch
        return self._common_step(features, batch_idx, "predict")

    def _variant_loss(self, variant_scores, batch):
        """Per-variant objective. None when the head or its labels are absent."""
        if variant_scores is None:
            return None
        pvc = batch.get("pvc")
        if pvc is None or "variant_labels" not in pvc:
            if float(self.hyperparams.get("variant_loss_weight", 0.0)) > 0:
                raise ValueError(
                    "variant_head is active with a non-zero loss weight but no "
                    "per-variant labels reached the batch; set variant_labels_path"
                )
            return None
        device = variant_scores.device
        targets = pvc["variant_labels"].to(device).float()
        observed = pvc["variant_label_mask"].to(device).bool()
        if targets.shape != variant_scores.shape or observed.shape != variant_scores.shape:
            raise ValueError("Misaligned per-variant labels")
        if not observed.any():
            return None

        objective = str(self.hyperparams.get("variant_objective", "pooled_bce"))
        if objective == "pooled_bce":
            per_token = F.binary_cross_entropy_with_logits(
                variant_scores, targets, reduction="none"
            )
            return (per_token * observed).sum() / observed.sum()
        if objective != "within_gene_ranking":
            raise ValueError(f"Unknown variant_objective: {objective}")

        strata = pvc["variant_strata"].to(device).long()
        cap = int(self.hyperparams.get("variant_pairs_per_gene", 256))
        per_gene = []
        for row in range(variant_scores.shape[0]):
            keep = observed[row]
            if not keep.any():
                continue
            scores, labels, types = variant_scores[row][keep], targets[row][keep], strata[row][keep]
            losses = []
            for stratum in torch.unique(types):
                if stratum.item() < 0:
                    continue
                selected = types == stratum
                rare = scores[selected & (labels == 1)]
                common = scores[selected & (labels == 0)]
                if rare.numel() == 0 or common.numel() == 0:
                    continue
                # Bound the outer product; a 1024-token gene would otherwise
                # materialise ~260k pairs per stratum.
                side = int(self.hyperparams.get("variant_pairs_per_side", 96))
                if rare.numel() > side:
                    rare = rare[torch.randperm(rare.numel(), device=rare.device)[:side]]
                if common.numel() > side:
                    common = common[torch.randperm(common.numel(), device=common.device)[:side]]
                losses.append(F.softplus(-(rare[:, None] - common[None, :])).reshape(-1))
            if not losses:
                continue
            pairs = torch.cat(losses)
            if pairs.numel() > cap:
                pairs = pairs[torch.randperm(pairs.numel(), device=pairs.device)[:cap]]
            per_gene.append(pairs.mean())
        if not per_gene:
            return None
        return torch.stack(per_gene).mean()

    def _nnpu_loss_terms(self, probas, labels, eps: float = 1e-6):
        """Return the implemented nnPU terms and clipping indicator.

        The unclipped negative-risk estimate is
        ``pi E_P[log(1-p)] - E_U[log(1-p)]``. The non-negative correction is
        active exactly when this quantity is below zero.
        """
        pos_mask = labels == 1
        unlabeled_mask = labels == 0
        if pos_mask.sum() > 0:
            pos_mean_log = torch.mean(torch.log(probas[pos_mask] + eps))
            pos_mean_log_1 = torch.mean(torch.log(1 - probas[pos_mask] + eps))
        else:
            pos_mean_log = torch.tensor(0.0, device=probas.device)
            pos_mean_log_1 = torch.tensor(0.0, device=probas.device)
        if unlabeled_mask.sum() > 0:
            unlabeled_mean_log_1 = torch.mean(torch.log(1 - probas[unlabeled_mask] + eps))
        else:
            unlabeled_mean_log_1 = torch.tensor(0.0, device=probas.device)
        positive_risk = -self.pu_class_prior * pos_mean_log
        negative_risk_unclipped = (
            self.pu_class_prior * pos_mean_log_1 - unlabeled_mean_log_1
        )
        correction_active = negative_risk_unclipped < 0
        loss = positive_risk + torch.relu(negative_risk_unclipped)
        return positive_risk, negative_risk_unclipped, correction_active, loss

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
                # Decisive GC/GO ablations still load the PVC dataset to keep
                # the eligible gene universe and batching identical. Use only
                # its gene identifiers for prediction export; PVC tensors are
                # not passed to the model when self.use_pvc is False.
                pvc_batch = batch.get("pvc")
                gene_names = pvc_batch.get("gene_name") if pvc_batch is not None else None
                mask = None

            model_input = {
                "gc": batch["gc"],
                "go": batch["go"],
            }
            if self.use_pvc:
                model_input["pvc"] = batch["pvc"]

            output = self.model(model_input, mask)
            variant_scores = output[-1] if self.model.variant_head is not None else None
            if self.hyperparams["return_attn"]:
                logits, probas, bin_preds, z_var, attn_weights = output[:5]
            else:
                logits, probas, bin_preds, z_var = output[:4]

            labels = pvc_labels
            if step_type == "train":
                if not self.hyperparams.get("pusb", False):
                    raise NotImplementedError(
                        "Varformer training requires hyperparameters.pusb=True (nnPU loss)."
                    )
                _, negative_risk_unclipped, correction_active, loss = self._nnpu_loss_terms(
                    probas, labels
                )
                self._nnpu_total_batches.add_(1)
                self._nnpu_active_batches.add_(correction_active.to(torch.long))
                self.log(
                    "train_nnpu_negative_risk_unclipped",
                    negative_risk_unclipped,
                    on_step=True,
                    on_epoch=True,
                    batch_size=labels.shape[0],
                )
                self.log(
                    "train_nnpu_correction_active",
                    correction_active.to(torch.float32),
                    on_step=True,
                    on_epoch=True,
                    batch_size=labels.shape[0],
                )

                variant_loss = self._variant_loss(variant_scores, batch)
                self._variant_total_batches += 1
                if variant_loss is not None:
                    self._variant_active_batches += 1
                    loss = loss + float(self.hyperparams["variant_loss_weight"]) * variant_loss
                    self.log(
                        "train_variant_bce",
                        variant_loss,
                        on_step=True,
                        on_epoch=True,
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
                bin_preds = (probas > self.decision_threshold).float()
                loss = None
                self._log(labels, step_type, loss, bin_preds, probas, test_source=test_source)
                return loss

            elif step_type == "predict":
                if gene_names is None:
                    raise ValueError(
                        "Prediction export requires aligned gene identifiers; "
                        "load the PVC dataset even when use_pvc=False."
                    )
                output = {}
                if self.hyperparams["return_attn"]:
                    for i, gene_name in enumerate(gene_names):
                        payload = {
                            "prediction": probas[i].detach().cpu().item(),
                            "classification": bin_preds[i].detach().cpu().item(),
                            "label": labels[i].detach().cpu().item(),
                            "test_source": test_source,
                        }
                        if z_var is not None:
                            payload["z_var"] = z_var[i].detach().cpu().to(torch.float32).numpy()
                        if attn_weights is not None:
                            payload["attn_weights"] = (
                                attn_weights[i].detach().cpu().to(torch.float32).numpy()
                            )
                        output[gene_name] = payload
                else:
                    for i, gene_name in enumerate(gene_names):
                        payload = {
                            "prediction": probas[i].detach().cpu().item(),
                            "classification": bin_preds[i].detach().cpu().item(),
                            "label": labels[i].detach().cpu().item(),
                            "test_source": test_source,
                        }
                        if z_var is not None:
                            payload["z_var"] = z_var[i].detach().cpu().to(torch.float32).numpy()
                        output[gene_name] = payload
                return output
            return loss

    def on_train_epoch_end(self):
        if float(self.hyperparams.get("variant_loss_weight", 0.0)) > 0:
            if self._variant_total_batches and not self._variant_active_batches:
                raise ValueError(
                    "variant_loss_weight is non-zero but no batch produced an "
                    "eligible variant loss this epoch; the objective is inactive"
                )
            self.variant_loss_history.append({
                "epoch": int(self.current_epoch),
                "active_batches": int(self._variant_active_batches),
                "total_batches": int(self._variant_total_batches),
            })
        self._variant_active_batches = 0
        self._variant_total_batches = 0
        total = int(self._nnpu_total_batches.item())
        active = int(self._nnpu_active_batches.item())
        fraction = active / total if total else 0.0
        self.nnpu_correction_history.append({
            "epoch": int(self.current_epoch),
            "active_batches": active,
            "total_batches": total,
            "active_fraction": fraction,
        })
        self.log("train_nnpu_correction_fraction", fraction, on_epoch=True)
        self._nnpu_active_batches.zero_()
        self._nnpu_total_batches.zero_()

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return None
        if not self.val_step_probas:
            return None
        all_probs = torch.cat(self.val_step_probas, dim=0).to(torch.float32).numpy()
        new_threshold = np.quantile(all_probs, 1 - self.threshold_positive_fraction)
        self.model.hyperparams["threshold"] = new_threshold
        self.decision_threshold.fill_(float(new_threshold))
        self.log("val_threshold", new_threshold)
        self.val_step_probas.clear()

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
            self.log(f"{step_type}_acc", self.acc(bin_preds, labels.long()), batch_size=labels.shape[0])
            self.log(f"{step_type}_auroc", self.auroc(probas, labels.int()), batch_size=labels.shape[0])
            if step_type == "val":
                # Mixed-precision training produces fp16 probabilities. Rank
                # correlation in fp16 can collapse to 0/NaN because its rank
                # accumulation exceeds the dtype's numerical range. Preserve
                # model precision but accumulate the selection metric in fp32.
                self.val_spearman_epoch.update(
                    probas.to(torch.float32),
                    labels.to(torch.float32),
                )
                self.log(
                    "val_spearman",
                    self.val_spearman_epoch,
                    on_step=False,
                    on_epoch=True,
                )
            else:
                spearman_val = self.spearman(probas, labels.float())
                self.log("train_spearman", spearman_val, batch_size=labels.shape[0])
            self.log(f"{step_type}_recall", self.recall(bin_preds, labels.long()), batch_size=labels.shape[0])
            self.log(f"{step_type}_precision", self.precision(bin_preds, labels.long()), batch_size=labels.shape[0])
            self.log(f"{step_type}_f1", self.f1(bin_preds, labels.long()), batch_size=labels.shape[0])
            self.log(f"{step_type}_auprc", self.auprc(probas, labels.long()), batch_size=labels.shape[0])
        else:
            self.log(f"{step_type}_acc_{test_source}", self.acc(bin_preds, labels.long()), batch_size=labels.shape[0])
            # probas, not bin_preds: AUROC is a ranking metric and is meaningless
            # over thresholded 0/1 calls. The train/val branch above always used
            # probas, and the LR/DrugnomeAI baselines score with
            # roc_auc_score(y, probs), so the previous form made the reported
            # test AUROC non-comparable with both.
            self.log(f"{step_type}_auroc_{test_source}", self.auroc(probas, labels.int()), batch_size=labels.shape[0])
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
            adamw_kwargs = {}
            optimizer_foreach = self.hyperparams.get("optimizer_foreach")
            if optimizer_foreach is not None:
                adamw_kwargs["foreach"] = optimizer_foreach
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=float(self.hyperparams["lr_start"]),
                weight_decay=weight_decay,
                **adamw_kwargs,
            )
        else:
            raise ValueError(f"Optimizer {self.hyperparams['optimizer']} not recognized.")

        if self.hyperparams["scheduler"] == "CosineAnnealingLR":
            # NOTE (2026-08-19): despite the name, this is WARM RESTARTS, not a
            # single cosine anneal. With T_mult unset (=1) and "interval": "step"
            # below, LR sawtooths between lr_start and lr_end every T0 optimizer
            # steps and the peak never decays. At T0=200 and 14646 train genes
            # that is ~68.7/34.5/17.4 restarts for batch 64/128/256 -- an almost
            # exact 2x halving that mirrors the historical "bs64 < bs128 < bs256"
            # HPO result, so that finding is likely a restart-frequency artifact
            # rather than a batch-size effect. Kept AS-IS so the pinned baseline
            # (private_repro/PINNED_BASELINE.md) stays byte-reproducible; use
            # "CosineAnnealingSingleCycle" below for the corrected behaviour.
            lr_scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=int(self.hyperparams["T0"]), eta_min=float(self.hyperparams["lr_end"])
            )
        elif self.hyperparams["scheduler"] == "CosineAnnealingSingleCycle":
            # True single-cycle cosine anneal over the whole run: LR decays
            # monotonically from lr_start to lr_end with no restarts, so the
            # schedule is identical in shape regardless of batch size. T_max is
            # derived from the actual total step count, which makes the schedule
            # batch-size invariant and lets batch size be tested on its own terms.
            total_steps = int(self.trainer.estimated_stepping_batches)
            lr_scheduler = CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=float(self.hyperparams["lr_end"])
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
