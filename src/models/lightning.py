import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class BaseLightningTargetIdentifier(pl.LightningModule):
    def __init__(self, model, config, imbalance):
        super().__init__()
        self.imbalance = imbalance
        self.config = config['hyperparameters']
        self.model = model

    def _log(self, labels, step_type, loss, bin_preds, probas, test_source=None):
        if step_type in ['train', 'val']:
            self.log(f'{step_type}_loss', loss, batch_size=labels.shape[0])
            self.log(f'{step_type}_acc', self.model.acc(bin_preds, labels), batch_size=labels.shape[0])
            self.log(f'{step_type}_auroc', self.model.auroc(probas, labels.int()),
                     batch_size=labels.shape[0])
            self.log(f'{step_type}_spearman', self.model.spearman(probas, labels.float()),
                     batch_size=labels.shape[0])
            self.log(f'{step_type}_recall', self.model.recall(bin_preds, labels.long()),
                     batch_size=labels.shape[0])
            self.log(f'{step_type}_recall@10', self.model.recall_at_10(bin_preds, labels.long()),
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
            self.log(f'{step_type}_recall@10_{test_source}', self.model.recall_at_10(bin_preds, labels.long()),
                     batch_size=labels.shape[0])
            self.logger.experiment.log({
                f'{step_type}_{test_source}_predictions': bin_preds.detach().cpu().numpy(),
                f'{step_type}_{test_source}_probas': probas.detach().cpu().numpy(),
                f'{step_type}_{test_source}_labels': labels.detach().cpu().numpy()
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
            self.log(f'{step_type}_recall@10', self.model.recall_at_10(bin_preds, labels.long()),
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
            self.log(f'{step_type}_recall@10', self.model.recall_at_10(bin_preds, labels.long()),
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
                logits = logits.unsqueeze(0)    # For the case where the batch size is 1
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

        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(self.config['T_0']),
                                                   eta_min=float(self.config['lr_end']))
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.zeros_(m.bias)


class MultiModalLightningTargetIdentifier(BaseLightningTargetIdentifier):
    def __init__(self, model, config, imbalance):
        super().__init__(model, config, imbalance)
        self.   model = model

    def forward(self, x, mask=None):
        return self.model(x, mask)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        features, masks = batch
        logits, probas, bin_preds = self(features, masks)
        return probas

    def _common_step(self, batch, batch_idx, step_type):
        if self.trainer.sanity_checking:
            return None
        else:
            for key, data in batch.items():
                if key == "pvc":
                    pvc_labels = batch[key]['labels']
                    test_source = batch[key]['test_source'][0]
                elif key == "go":
                    go_labels = batch[key][1]
                else:
                    gc_labels = batch[key][1]

            assert torch.all(torch.eq(pvc_labels, go_labels)) and torch.all(torch.eq(go_labels, gc_labels))

            logits, probas, bin_preds = self(batch, batch["pvc"]["mask"])

            labels = pvc_labels     # we can pick any of the three label set
            if step_type == 'train':
                class_weight = torch.tensor([1 if labels[i] == 0 else self.imbalance for i in range(len(labels))], device=self.device)
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

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'test')

    def setup(self, stage):
        self.initialize_weights()

    def configure_optimizers(self):
        weight_decay = float(self.config.get('weight_decay', 0))
        if self.config['optimizer'] == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=float(self.config['lr_start']), weight_decay=weight_decay)
        elif self.config['optimizer'] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=float(self.config['lr_start']), weight_decay=weight_decay)
        elif self.config['optimizer'] == "RMSprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=float(self.config['lr_start']), weight_decay=weight_decay)
        elif self.config['optimizer'] == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.config['lr_start']), weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {self.config['optimizer']} not recognized.")

        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(self.config['T_0']), eta_min=float(self.config['lr_end']))
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
