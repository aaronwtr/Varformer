import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, ExponentialLR, ReduceLROnPlateau

from models.target_identifier import MultiModalTargetIdentifier


class MultiModalLightningTargetIdentifier(pl.LightningModule):
    def __init__(self, config, num_samples_per_class, num_features_gc, num_features_go, num_mutations, max_seq_len,
                 num_genes, class_prior):
        self.save_hyperparameters()
        super().__init__()

        self.config = config['hyperparameters']
        self.model = MultiModalTargetIdentifier(
            config=config,
            num_features_gc=num_features_gc,
            num_features_go=num_features_go,
            num_mutations=num_mutations,
            max_seq_len=max_seq_len,
            num_genes=num_genes
        )
        self.pi = class_prior
        self.val_step_probas = []

    def forward(self, x, mask=None):
        return self.model(x, mask)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        features = batch
        mask = features['pvc']['mask']
        return self._common_step(features, batch_idx, 'predict', split_idx=dataloader_idx)

    def _common_step(self, batch, batch_idx, step_type, split_idx=None):
        if self.trainer.sanity_checking:
            return None
        else:
            # Extract labels (assuming the pvc branch holds the PU labels)
            for key, data in batch.items():
                if key == "pvc":
                    pvc_labels = batch[key]['labels']
                    test_source = batch[key]['test_source'][0]
                elif key == "go":
                    go_labels = batch[key][1]
                else:
                    gc_labels = batch[key][1]

            # Forward pass through the model (using the pvc mask for transformer inputs)
            if self.config['return_attn']:
                logits, probas, bin_preds, attn_weights = self.model(
                    {
                        'gc': batch['gc'],
                        'go': batch['go'],
                        'pvc': batch['pvc']
                    },
                    batch["pvc"]["mask"]
                )
            else:
                logits, probas, bin_preds = self.model(
                    {
                        'gc': batch['gc'],
                        'go': batch['go'],
                        'pvc': batch['pvc']
                    },
                    batch["pvc"]["mask"]
                )

            labels = pvc_labels  # Use the labels from pvc branch
            eps = 1e-8  # to avoid log(0)

            if step_type == 'train':
                if self.config.get('pusb', False):
                    pos_mask = (labels == 1)
                    unlabeled_mask = (labels == 0)

                    # Compute means safely (if no positive/unlabeled sample, use 0)
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

                    # Nonnegative PU risk:
                    # loss = -π * E[log f(x)]  + max(0, π * E[log(1 - f(x))] - E[log(1 - f(x))] over unlabeled)
                    loss_positive = - self.pi * pos_mean_log
                    loss_unlabeled_component = self.pi * pos_mean_log_1 - unlabeled_mean_log_1
                    loss = loss_positive + torch.relu(loss_unlabeled_component)
                else:
                    # Class-balanced BCE
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

            elif step_type == 'val':
                if logits.shape != labels.shape:
                    if logits.dim() == 0:
                        logits = logits.unsqueeze(0)

                    if labels.dim() == 0:
                        labels = labels.unsqueeze(0)
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())
                self._log(labels, step_type, loss, bin_preds, probas)
                self.val_step_probas.append(probas.detach().cpu())
            elif step_type == 'test':
                if 'best_threshold' in self.model.config:
                    bin_preds = (probas > self.model.config['best_threshold']).float()
                loss = None
                self._log(labels, step_type, loss, bin_preds, probas, test_source=test_source)
                return loss
            elif step_type == 'predict':
                if self.config['return_attn']:
                    return probas, attn_weights, split_idx if split_idx is not None else 0
                else:
                    return probas, split_idx if split_idx is not None else 0

            return loss

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return None
        all_probs = torch.cat(self.val_step_probas, dim=0).to(torch.float32).numpy()
        new_threshold = np.quantile(all_probs, 1 - self.pi)
        self.model.config['threshold'] = new_threshold
        self.log('val_threshold', new_threshold)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'test')

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
        if step_type in ['train', 'val']:
            if pos_loss is not None:
                self.log(f'{step_type}_pos_loss', pos_loss, batch_size=labels.shape[0])
            if neg_loss is not None:
                self.log(f'{step_type}_neg_loss', neg_loss, batch_size=labels.shape[0])

            self.log(f'{step_type}_loss', loss, batch_size=labels.shape[0])
            self.log(f'{step_type}_acc', self.model.acc(bin_preds, labels), batch_size=labels.shape[0])
            self.log(f'{step_type}_auroc', self.model.auroc(probas, labels.int()),
                     batch_size=labels.shape[0])
            self.model_spearman = self.model.spearman(probas, labels.float())
            self.log(f'{step_type}_spearman', self.model_spearman,
                     batch_size=labels.shape[0])
            self.log(f'{step_type}_recall', self.model.recall(bin_preds, labels.long()),
                     batch_size=labels.shape[0])
            self.log(f'{step_type}_precision', self.model.precision(bin_preds, labels.long()),
                     batch_size=labels.shape[0])
            self.log(f'{step_type}_f1', self.model.f1(bin_preds, labels.long()),
                     batch_size=labels.shape[0]),
            self.log(f'{step_type}_auprc', self.model.auprc(probas, labels.long()),
                     batch_size=labels.shape[0])
        else:
            self.log(f'{step_type}_acc_{test_source}', self.model.acc(bin_preds, labels),
                     batch_size=labels.shape[0])
            self.log(f'{step_type}_auroc_{test_source}', self.model.auroc(bin_preds, labels.int()),
                     batch_size=labels.shape[0])
            self.log(f'{step_type}_spearman_{test_source}', self.model.spearman(probas, labels.float()),
                     batch_size=labels.shape[0])
            self.log(f'{step_type}_recall_{test_source}', self.model.recall(bin_preds, labels.long()),
                     batch_size=labels.shape[0])
            self.log(f'{step_type}_precision_{test_source}', self.model.precision(bin_preds, labels.long()),
                     batch_size=labels.shape[0])
            self.log(f'{step_type}_f1_{test_source}', self.model.f1(bin_preds, labels.long()),
                     batch_size=labels.shape[0])
            self.log(f'{step_type}_auprc_{test_source}', self.model.auprc(probas, labels.long()),
                     batch_size=labels.shape[0])

            logger_type = self.logger.__class__.__name__
            if logger_type == "WandbLogger":
                self.logger.experiment.log({
                    f'{step_type}_{test_source}_predictions': bin_preds.detach().cpu().numpy(),
                    f'{step_type}_{test_source}_probas': probas.to(dtype=torch.float32).detach().cpu().numpy(),
                    f'{step_type}_{test_source}_labels': labels.to(dtype=torch.float32).detach().cpu().numpy()
                })

    def configure_optimizers(self):
        weight_decay = float(self.config.get('weight_decay', 0))
        if self.config['optimizer'] == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=float(self.config['lr_start']),
                                         weight_decay=weight_decay)
        elif self.config['optimizer'] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=float(self.config['lr_start']),
                                        weight_decay=weight_decay)
        elif self.config['optimizer'] == "RMSprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=float(self.config['lr_start']),
                                            weight_decay=weight_decay)
        elif self.config['optimizer'] == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.config['lr_start']),
                                          weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {self.config['optimizer']} not recognized.")

        if self.config['scheduler'] == "CosineAnnealingLR":
            lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(self.config['T0']),
                                                       eta_min=float(self.config['lr_end']))
        elif self.config['scheduler'] == "StepLR":
            lr_scheduler = StepLR(optimizer, step_size=int(self.config['step_size']),
                                  gamma=float(self.config['gamma']))
        elif self.config['scheduler'] == "ExponentialLR":
            lr_scheduler = ExponentialLR(optimizer, gamma=float(self.config['gamma']))
        elif self.config['scheduler'] == "ReduceLROnPlateau":
            lr_scheduler = ReduceLROnPlateau(optimizer, factor=float(self.config['factor']),
                                             patience=int(self.config['patience']))
        else:
            raise ValueError(f"Scheduler {self.config['scheduler']} not recognized.")
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def initialize_weights(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        initial_weights = {}

        gc_weights = self.model.gc_branch.initialise_weights(seed)
        initial_weights.update({'gc_branch.' + k: v for k, v in gc_weights.items()})

        go_weights = self.model.go_branch.initialise_weights(seed)
        initial_weights.update({'go_branch.' + k: v for k, v in go_weights.items()})

        pvc_weights = self.model.pvc_branch.initialise_weights(seed)
        initial_weights.update({'pvc_branch.' + k: v for k, v in pvc_weights.items()})

        for name, module in self.model.classification_branch.named_modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                initial_weights[f'classification_branch.{name}.weight'] = module.weight.detach().clone()
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                    initial_weights[f'classification_branch.{name}.bias'] = module.bias.detach().clone()
            elif isinstance(module, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(module.weight, 1)
                initial_weights[f'classification_branch.{name}.weight'] = module.weight.detach().clone()
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
                    initial_weights[f'classification_branch.{name}.bias'] = module.bias.detach().clone()
                module.running_mean.fill_(0)
                module.running_var.fill_(1)
                initial_weights[f'classification_branch.{name}.running_mean'] = module.running_mean.detach().clone()
                initial_weights[f'classification_branch.{name}.running_var'] = module.running_var.detach().clone()

        return initial_weights


# legacy
class BaseLightningTargetIdentifier(pl.LightningModule):
    def __init__(self, model, config, num_samples_per_class, beta=0.9999):
        super().__init__()
        self.beta = beta
        self.num_samples_per_class = num_samples_per_class
        self.config = config['hyperparameters']
        self.model = model

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
        if step_type in ['train', 'val']:
            if pos_loss is not None:
                self.log(f'{step_type}_pos_loss', pos_loss, batch_size=labels.shape[0])
            if neg_loss is not None:
                self.log(f'{step_type}_neg_loss', neg_loss, batch_size=labels.shape[0])

            self.log(f'{step_type}_loss', loss, batch_size=labels.shape[0])
            self.log(f'{step_type}_acc', self.model.acc(bin_preds, labels), batch_size=labels.shape[0])
            self.log(f'{step_type}_auroc', self.model.auroc(probas, labels.int()),
                     batch_size=labels.shape[0])
            self.log(f'{step_type}_spearman', self.model.spearman(probas, labels.float()),
                     batch_size=labels.shape[0])
            self.log(f'{step_type}_recall', self.model.recall(bin_preds, labels.long()),
                     batch_size=labels.shape[0])
            self.log(f'{step_type}_precision', self.model.precision(bin_preds, labels.long()),
                     batch_size=labels.shape[0])
            self.log(f'{step_type}_f1', self.model.f1(bin_preds, labels.long()),
                     batch_size=labels.shape[0]),
            self.log(f'{step_type}_auprc', self.model.auprc(probas, labels.long()),
                     batch_size=labels.shape[0])
        else:
            self.log(f'{step_type}_acc_{test_source}', self.model.acc(bin_preds, labels),
                     batch_size=labels.shape[0])
            self.log(f'{step_type}_auroc_{test_source}', self.model.auroc(bin_preds, labels.int()),
                     batch_size=labels.shape[0])
            self.log(f'{step_type}_spearman_{test_source}', self.model.spearman(probas, labels.float()),
                     batch_size=labels.shape[0])
            self.log(f'{step_type}_recall_{test_source}', self.model.recall(bin_preds, labels.long()),
                     batch_size=labels.shape[0])
            self.log(f'{step_type}_precision_{test_source}', self.model.precision(bin_preds, labels.long()),
                     batch_size=labels.shape[0])
            self.log(f'{step_type}_f1_{test_source}', self.model.f1(bin_preds, labels.long()),
                     batch_size=labels.shape[0])
            self.log(f'{step_type}_auprc_{test_source}', self.model.auprc(probas, labels.long()),
                     batch_size=labels.shape[0])

            logger_type = self.logger.__class__.__name__
            if logger_type == "WandbLogger":
                self.logger.experiment.log({
                    f'{step_type}_{test_source}_predictions': bin_preds.detach().cpu().numpy(),
                    f'{step_type}_{test_source}_probas': probas.to(dtype=torch.float32).detach().cpu().numpy(),
                    f'{step_type}_{test_source}_labels': labels.to(dtype=torch.float32).detach().cpu().numpy()
                })

    def _common_step(self, batch, batch_idx, step_type):
        if len(batch) > 2:  # For transformer
            features = {key: batch[key] for key in ['pathogenicity', 'position', 'mutation', 'gene']}
            labels = batch['labels']
            masks = batch['mask']
            test_source = batch['test_source'][0]

            logits, probas, bin_preds, _ = self(features, masks)
        else:  # For regular MLP
            features, labels = batch
            test_source = None
            logits, probas, bin_preds = self(features)

        labels = (labels > float(self.config['threshold'])).float()
        if step_type == 'train':
            class_weight = torch.tensor([1 if labels[i] == 0 else self.imbalance for i in range(len(labels))],
                                        device=self.device)
            loss = F.binary_cross_entropy_with_logits(logits, labels.float(), weight=class_weight)
            self._log(labels, step_type, loss, bin_preds, probas)
        elif step_type == 'val':
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            self._log(labels, step_type, loss, bin_preds, probas)
        else:
            loss = None
            self._log(labels, step_type, loss, bin_preds, probas, test_source=test_source)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        weight_decay = float(self.config.get('weight_decay', 0))
        if self.config['optimizer'] == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=float(self.config['lr_start']),
                                         weight_decay=weight_decay)
        elif self.config['optimizer'] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=float(self.config['lr_start']), weight_decay=weight_decay)
        elif self.config['optimizer'] == "RMSprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=float(self.config['lr_start']),
                                            weight_decay=weight_decay)
        elif self.config['optimizer'] == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.config['lr_start']),
                                          weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {self.config['optimizer']} not recognized.")

        return optimizer


class MLPLightningTargetIdentifier(BaseLightningTargetIdentifier):
    def __init__(self, model, config, imbalance):
        super().__init__(model, config, imbalance)

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        features = batch
        logits, probas, bin_preds = self(features)
        return probas


class VarformerLightningTargetIdentifier(BaseLightningTargetIdentifier):
    def __init__(self, model, config, imbalance):
        super().__init__(model, config, imbalance)

    def forward(self, x, mask):
        return self.model(x, mask)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        features, masks = batch
        logits, probas, bin_preds = self(features, masks)
        return probas


class ShardedVarformerLightningTargetIdentifier(BaseLightningTargetIdentifier):
    def __init__(self, model, config, imbalance):
        super().__init__(model, config, imbalance)
        self.model = model
        self.accumulated_shards = {}

    def forward(self, x, mask):
        return self.model(x, mask)

    def _log(self, labels, step_type, loss, bin_preds, probas, test_source=None):
        if step_type in ['train', 'val']:
            if bin_preds.shape != labels.shape:
                bin_preds = bin_preds.unsqueeze(0)
                probas = probas.unsqueeze(0)
            self.log(f'{step_type}_loss', loss, batch_size=labels.shape[0], sync_dist=True)
            self.log(f'{step_type}_acc', self.model.acc(bin_preds, labels), batch_size=labels.shape[0],
                     sync_dist=True)
            self.log(f'{step_type}_auroc', self.model.auroc(probas, labels.int()),
                     batch_size=labels.shape[0], sync_dist=True)
            self.log(f'{step_type}_spearman', self.model.spearman(probas, labels.float()),
                     batch_size=labels.shape[0], sync_dist=True)
            self.log(f'{step_type}_recall', self.model.recall(bin_preds, labels.long()),
                     batch_size=labels.shape[0], sync_dist=True)
            self.log(f'{step_type}_precision', self.model.precision(bin_preds, labels.long()),
                     batch_size=labels.shape[0], sync_dist=True)
            self.log(f'{step_type}_f1', self.model.f1(bin_preds, labels.long()),
                     batch_size=labels.shape[0], sync_dist=True)
        else:
            self.log(f'{step_type}_acc_{test_source}', self.model.acc(bin_preds, labels),
                     batch_size=labels.shape[0], sync_dist=True)
            self.log(f'{step_type}_auroc_{test_source}', self.model.auroc(probas, labels.int()),
                     batch_size=labels.shape[0], sync_dist=True)
            self.log(f'{step_type}_spearman_{test_source}', self.model.spearman(probas, labels.float()),
                     batch_size=labels.shape[0], sync_dist=True)
            self.log(f'{step_type}_recall', self.model.recall(bin_preds, labels.long()),
                     batch_size=labels.shape[0], sync_dist=True)
            self.log(f'{step_type}_precision', self.model.precision(bin_preds, labels.long()),
                     batch_size=labels.shape[0], sync_dist=True)
            self.log(f'{step_type}_f1', self.model.f1(bin_preds, labels.long()),
                     batch_size=labels.shape[0], sync_dist=True)
            self.logger.experiment.log({
                f'{step_type}_{test_source}_predictions': bin_preds.detach().cpu().numpy(),
                f'{step_type}_{test_source}_probas': probas.detach().cpu().numpy(),
                f'{step_type}_{test_source}_labels': labels.detach().cpu().numpy()
            })

    def _common_step(self, batch, batch_idx, step_type):
        features = {key: batch[key] for key in ['pathogenicity', 'position', 'mutation', 'gene']}
        labels = batch['labels']
        masks = batch['mask']
        test_source = batch['test_source'][0]

        logits, probas, bin_preds, _ = self(features, masks)

        labels = (labels > float(self.config['threshold'])).float()
        if step_type == 'train':
            class_weight = torch.tensor([1 if labels[i] == 0 else self.imbalance for i in range(len(labels))],
                                        device=self.device)
            loss = F.binary_cross_entropy_with_logits(logits, labels.float(), weight=class_weight)
            self._log(labels, step_type, loss, bin_preds, probas)
        elif step_type == 'val':
            if logits.shape != labels.shape:
                logits = logits.unsqueeze(0)  # For the case where the batch size is 1
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            self._log(labels, step_type, loss, bin_preds, probas)
        else:
            loss = None
            self._log(labels, step_type, loss, bin_preds, probas, test_source)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        weight_decay = float(self.config.get('weight_decay', 0))
        if self.config['optimizer'] == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=float(self.config['lr_start']),
                                         weight_decay=weight_decay)
        elif self.config['optimizer'] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=float(self.config['lr_start']), weight_decay=weight_decay)
        elif self.config['optimizer'] == "RMSprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=float(self.config['lr_start']),
                                            weight_decay=weight_decay)
        elif self.config['optimizer'] == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.config['lr_start']),
                                          weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {self.config['optimizer']} not recognized.")

        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(self.config['T0']),
                                                   eta_min=float(self.config['lr_end']))
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.zeros_(m.bias)
