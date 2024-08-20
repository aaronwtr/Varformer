import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np


class BaseLightningTargetIdentifier(pl.LightningModule):
    def __init__(self, model, config, imbalance, model_type):
        super().__init__()
        self.imbalance = imbalance
        self.mlp_config = config['hyperparameters']['mlp']
        self.config = config['hyperparameters'][model_type]
        self.model = model

    def _log(self, labels, step_type, loss, bin_preds, probas, test_source=None):
        if step_type in ['train', 'val']:
            self.log(f'{step_type}_loss', loss)
            self.log(f'{step_type}_acc', self.model.acc(bin_preds, labels))
            self.log(f'{step_type}_auroc', self.model.auroc(bin_preds, labels.int()))
            self.log(f'{step_type}_spearman', self.model.spearman(probas, labels.float()))
            self.log(f'{step_type}_f1', self.model.f1(bin_preds, labels.long()))
        else:
            self.log(f'{step_type}_{test_source}_acc', self.model.acc(bin_preds, labels))
            self.log(f'{step_type}_{test_source}_auroc', self.model.auroc(bin_preds, labels.int()))
            self.log(f'{step_type}_{test_source}_spearman', self.model.spearman(probas, labels.float()))
            self.log(f'{step_type}_{test_source}_f1', self.model.f1(bin_preds, labels.long()))

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

        labels = (labels > float(self.mlp_config['threshold'])).float()
        if step_type == 'train':
            class_weight = torch.tensor([1 if labels[i] == 0 else self.imbalance for i in range(len(labels))],
                                        device=self.device)
            loss = F.binary_cross_entropy_with_logits(logits, labels.float(), weight=class_weight)
            self._log(labels, step_type, loss, bin_preds, probas)
        elif step_type == 'val':
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            self._log(labels, step_type, loss, bin_preds, probas)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
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
    def __init__(self, model, config, imbalance, model_type="mlp"):
        super().__init__(model, config, imbalance, model_type)

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        features = batch
        logits, probas, bin_preds = self(features)
        return probas


class VarformerLightningTargetIdentifier(BaseLightningTargetIdentifier):
    def __init__(self, model, config, imbalance, model_type="varformer"):
        super().__init__(model, config, imbalance, model_type)

    def forward(self, x, mask):
        return self.model(x, mask)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        features, masks = batch
        logits, probas, bin_preds = self(features, masks)
        return probas


class ShardedVarformerLightningTargetIdentifier(BaseLightningTargetIdentifier):
    def __init__(self, model, config, imbalance, model_type="varformer"):
        super().__init__(model, config, imbalance, model_type)
        self.model = model
        self.accumulated_shards = {}

    def forward(self, x, mask):
        return self.model(x, mask)

    def training_step(self, batch, batch_idx):
        _, _, _, shard_embeds = self(batch, mask=batch['mask'])

        labels = torch.tensor([int(label) for label in batch['labels']], dtype=torch.float32, device=self.device)
        # Feed embeddings through the TargetID MLP
        logits = self.model.layers(shard_embeds).squeeze()
        class_weight = torch.tensor([1 if label == 0 else self.imbalance for label in labels],
                                    device=self.device)
        loss = F.binary_cross_entropy_with_logits(logits, labels, weight=class_weight)
        self.log('train_loss', loss)
        return loss
        # else:
        #     return None  # Skip the optimizer step when no complete genes are available

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
