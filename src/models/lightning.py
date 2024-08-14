import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class BaseLightningTargetIdentifier(pl.LightningModule):
    def __init__(self, model, config, imbalance, model_type):
        super().__init__()
        self.imbalance = imbalance
        self.mlp_config = config['hyperparameters']['mlp']
        self.config = config['hyperparameters'][model_type]
        self.model = model

    def _common_step(self, batch, batch_idx, step_type):
        if len(batch) > 2:  # For transformer
            features = {key: batch[key] for key in ['pathogenicity', 'position', 'mutation']}
            labels = batch['labels']
            masks = batch['mask']
            gene_ids = batch['gene_id']
            shard_id = batch['shard_id']
            total_shards = batch['total_shards']

            logits, probas, bin_preds = self(features, masks)
        else:  # For regular MLP
            features, labels = batch
            logits, probas, bin_preds = self(features)

        if step_type == 'train':
            class_weight = torch.tensor([1 if labels[i] == 0 else self.imbalance for i in range(len(labels))],
                                        device=self.device)
            loss = F.binary_cross_entropy_with_logits(logits, labels.float(), weight=class_weight)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        labels = (labels > float(self.mlp_config['threshold'])).float()

        self.log(f'{step_type}_loss', loss)
        self.log(f'{step_type}_acc', self.model.acc(bin_preds, labels))
        self.log(f'{step_type}_auroc', self.model.auroc(bin_preds, labels.int()))
        self.log(f'{step_type}_spearman', self.model.spearman(probas, labels.float()))
        self.log(f'{step_type}_f1', self.model.f1(bin_preds, labels.long()))

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        if len(batch) == 4:  # For varformer
            features, masks, labels, test_source = batch
            logits, probas, bin_preds = self(features, masks)
        else:  # For regular MLP
            features, labels, test_source = batch
            logits, probas, bin_preds = self(features)

        labels = (labels > float(self.config['threshold'])).float()
        test_source = test_source[0]

        self.log(f'test_acc_{test_source}', self.model.acc(bin_preds, labels))
        self.log(f'test_auroc_{test_source}', self.model.auroc(bin_preds, labels.int()))
        self.log(f'test_spearman_{test_source}', self.model.spearman(probas, labels.float()))
        self.log(f'test_f1_{test_source}', self.model.f1(bin_preds, labels.long()))

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
        shard_embeds = self(batch, mask=batch['mask'])

        # Accumulate shard embeddings
        for i, (gene_id, shard_id) in enumerate(zip(batch['gene_id'], batch['shard_id'])):
            if gene_id not in self.accumulated_shards:
                self.accumulated_shards[gene_id] = {}
            self.accumulated_shards[gene_id][shard_id] = shard_embeds[i]

        # Process genes with all shards present
        complete_genes = []
        for gene_id, shards in self.accumulated_shards.items():
            if len(shards) == batch['total_shards'][batch['gene_id'] == gene_id][0]:
                gene_embed = self.model.aggregator(torch.stack(list(shards.values())))
                complete_genes.append((gene_id, gene_embed))
                del self.accumulated_shards[gene_id]

        if complete_genes:
            gene_ids, gene_embeds = zip(*complete_genes)
            gene_embeds = torch.stack(gene_embeds)

            # Compute loss based on your specific task
            # TODO: make sure the below is the targetidentifier
            logits, probas, bin_preds = self.model.layers(gene_embeds).squeeze()
            class_weight = torch.tensor([1 if batch['labels'][i] == 0 else self.imbalance for i in range(len(batch['labels']))],
                                        device=self.device)
            loss = F.binary_cross_entropy_with_logits(logits, batch['labels'].float(), weight=class_weight)
            self.log('train_loss', loss)
            return loss
        else:
            return None  # Skip the optimizer step when no complete genes are available

    def on_train_epoch_end(self):
        # Clear accumulated shards at the end of each epoch
        self.accumulated_shards.clear()

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
