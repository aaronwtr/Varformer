"""Varformer: multi-modal gene tractability model.

Renamed from MultiModalTargetIdentifier (src/models/target_identifier.py).

Deleted classes (no replacement — they were dead or superseded):
  - BaseTargetIdentifier       (dead wrapper with unused classifier)
  - VarformerTargetIdentifier  (useless extra wrapper; its .varformer attr
                                caused the state_dict double-prefix)
  - MultiModalTargetIdentifierV1 (legacy architecture)

Metrics (Accuracy, AUROC, ...) have been moved to VarformerLightningModule.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from varformer.models.attention import GeneVariantAttention
from varformer.models.variant_encoder import VariantEncoder


class Varformer(nn.Module):
    """Multi-modal gene tractability predictor.

    Combines GC (genome context) features, GO (gene ontology) features, and
    per-variant transformer embeddings to predict gene tractability.

    Args:
        config:              Dict or Config object supporting
                             config['hyperparameters'] access.
        num_features_gc:     Width of the GC feature vector per gene.
        num_features_go:     Width of the GO feature vector per gene.
        num_mutations:       Vocabulary size of mutation encodings.
        max_seq_len:         Maximum number of variants per gene.
        num_genes:           Total number of genes (used for informational
                             purposes; not tied to any parameter shape).
        use_pvc:             Whether the variant-context (PVC) branch is active.
    """

    def __init__(
        self,
        config: Any,
        num_features_gc: int,
        num_features_go: int,
        num_mutations: int,
        max_seq_len: int,
        num_genes: int,
        use_pvc: bool = True,
    ):
        super().__init__()

        self.num_features_gc = num_features_gc
        self.num_features_go = num_features_go
        self.num_mutations = num_mutations
        self.max_seq_len = max_seq_len
        self.num_genes = num_genes
        self.use_pvc = use_pvc
        self.config = config
        self.hyperparams = self.config["hyperparameters"]
        self.dropout = nn.Dropout(float(self.hyperparams["dropout"]))

        # 1. GC projection branch
        gc_width = int(self.hyperparams["gc_width"])
        self.gc_projection = nn.Sequential(
            nn.Linear(self.num_features_gc, gc_width),
            nn.LayerNorm(gc_width),
            nn.ReLU(),
            nn.Dropout(p=float(self.hyperparams["dropout"])),
        )

        # 2. GO projection branch
        go_width = int(self.hyperparams["go_width"])
        self.go_projection = nn.Sequential(
            nn.Linear(num_features_go, go_width),
            nn.LayerNorm(go_width),
            nn.ReLU(),
            nn.Dropout(p=float(self.hyperparams["dropout"])),
        )

        # 3. PVC branch (VariantEncoder + cross-attention) — optional
        gene_feature_dim = gc_width + go_width

        if self.use_pvc:
            # Attribute name `self.varformer` is kept so checkpoint keys like
            # model.varformer.mutation_embedding.weight continue to match
            # (after the load_legacy_checkpoint wrapper-prefix collapse).
            self.varformer = VariantEncoder(
                max_seq_len=max_seq_len,
                num_muts=num_mutations,
                dropout=float(self.hyperparams["dropout"]),
                d_model=int(self.hyperparams["d_model"]),
                dim_feedforward=int(self.hyperparams["dim_feedforward"]),
                nhead=int(self.hyperparams["nhead"]),
                num_encoder_layers=int(self.hyperparams["num_encoder_layers"]),
            )

            variant_feature_dim = int(self.hyperparams["d_model"])
            attention_dim = int(self.hyperparams["gv_attn_dim"])

            self.gene_variant_attention = GeneVariantAttention(
                gene_feature_dim=gene_feature_dim,
                variant_feature_dim=variant_feature_dim,
                attention_dim=attention_dim,
                nhead=int(self.hyperparams["nhead"]),
            )
        else:
            self.varformer = None
            self.gene_variant_attention = None
            attention_dim = 0

        # 4. Classification head
        cls_head_layers = []
        depth_cls_head = int(self.hyperparams["depth_cls_head"])
        inp_dim_classifier = gene_feature_dim + attention_dim
        current_dim = inp_dim_classifier
        for _ in range(depth_cls_head):
            hidden_dim = current_dim // 2
            cls_head_layers += [
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=float(self.hyperparams["dropout"])),
            ]
            current_dim = hidden_dim
        cls_head_layers += [nn.Linear(current_dim, 1)]
        self.classification_head = nn.Sequential(*cls_head_layers)

    def forward(
        self,
        x: dict,
        mask: torch.Tensor | None = None,
    ):
        """Forward pass.

        Args:
            x:    Dict with keys 'gc', 'go', and (if use_pvc) 'pvc'.
                  - x['gc']: tuple/list whose first element is [B, num_features_gc].
                  - x['go']: tuple/list whose first element is [B, num_features_go].
                  - x['pvc']: dict with 'pathogenicity', 'position', 'mutation'.
            mask: [B, max_seq_len] bool padding mask (True = padding).

        Returns:
            If return_attn is True:
                (logits, probas, bin_preds, z_var, attn_weights)
            Else:
                (logits, probas, bin_preds, z_var)
        """
        device = next(self.parameters()).device

        x_gc = x["gc"][0].to(device)
        x_go = x["go"][0].to(device)

        z_gc = self.gc_projection(x_gc)
        z_go = self.go_projection(x_go)
        z_gene = torch.cat([z_gc, z_go], dim=-1)

        if self.use_pvc:
            pvc_input = {
                "pathogenicity": x["pvc"]["pathogenicity"].to(device),
                "position": x["pvc"]["position"].to(device),
                "mutation": x["pvc"]["mutation"].to(device),
            }

            z_pvc = self.varformer(
                pvc_input["pathogenicity"],
                pvc_input["position"],
                pvc_input["mutation"],
                mask,
            )
            z_var, variant_attn_weights = self.gene_variant_attention(z_gene, z_pvc)
            concatenated_features = torch.cat([z_gene, z_var], dim=-1)
        else:
            z_var = None
            variant_attn_weights = None
            concatenated_features = z_gene

        logits = self.classification_head(concatenated_features).squeeze()
        probabilities = torch.sigmoid(logits)
        binary_predictions = (probabilities > float(self.hyperparams["threshold"])).float()

        if self.hyperparams["return_attn"]:
            return logits, probabilities, binary_predictions, z_var, variant_attn_weights
        else:
            return logits, probabilities, binary_predictions, z_var

    # ------------------------------------------------------------------
    # Public SDK class methods (Phase 7)
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, population: str, seed=42):
        """Load a published checkpoint for (population, seed).

        seed:
          - int: load that specific seed.
          - "best": pick checkpoint with highest val_spearman parsed from filename.
          - "ensemble": load all 5 seeds, return a VarformerEnsemble wrapper.
        """
        from varformer.config import Config
        from varformer.checkpoints import find_checkpoint, best_seed

        if seed == "ensemble":
            from varformer.models.ensemble import VarformerEnsemble
            return VarformerEnsemble.from_pretrained(population)

        config = Config.load()
        ckpt_root = config.paths.ckpt_root
        if seed == "best":
            seed = best_seed(ckpt_root, population)
        ckpt_path = find_checkpoint(ckpt_root, population, int(seed))
        return cls._build_and_load(config, population, ckpt_path)

    @classmethod
    def from_checkpoint(cls, path):
        """Load any checkpoint by path."""
        from varformer.config import Config
        from pathlib import Path
        config = Config.load()
        p = Path(path)
        population = p.parent.name if p.parent.name in ("nfe", "elgh", "afr", "amr") else "nfe"
        return cls._build_and_load(config, population, p)

    @classmethod
    def _build_and_load(cls, config, population, ckpt_path):
        """Internal: instantiate LightningModule, load legacy checkpoint, return nn.Module instance with metadata."""
        from varformer.checkpoints import load_legacy_checkpoint
        from varformer.training.lightning_module import VarformerLightningModule
        import pickle

        with open(str(config.paths["MISSENSE_MAP"]), "rb") as f:
            missense_map = pickle.load(f)
        num_mutations = len(missense_map)

        raw_ckpt = load_legacy_checkpoint(ckpt_path)
        hp = raw_ckpt.get("hyper_parameters", {})
        num_features_gc = hp.get("num_features_gc")
        num_features_go = hp.get("num_features_go")
        max_seq_len = hp.get("max_seq_len", config.hyperparameters.max_seq_len)
        num_genes = hp.get("num_genes", 0)

        lm = VarformerLightningModule(
            config={"hyperparameters": config.hyperparameters.model_dump()},
            num_samples_per_class=None,
            num_features_gc=num_features_gc,
            num_features_go=num_features_go,
            num_mutations=num_mutations,
            max_seq_len=max_seq_len,
            num_genes=num_genes,
            class_prior=None,
            use_pvc=config.hyperparameters.use_pvc,
        )
        lm.load_state_dict(raw_ckpt["state_dict"], strict=False)
        # Return the inner nn.Module with metadata. Use object.__setattr__ to bypass
        # nn.Module's __setattr__ hook — if we used regular assignment, lm (an
        # nn.Module) would be registered as a child of `instance` (its own .model),
        # creating a cycle that breaks .apply()/.to() recursion.
        instance = lm.model
        object.__setattr__(instance, "_population", population)
        object.__setattr__(instance, "_lightning_module", lm)
        object.__setattr__(instance, "_config", config)
        return instance

    @classmethod
    def trainer(cls, population, config_overrides=None, output_dir=None):
        """Return a VarformerTrainer for the given population."""
        from varformer.training.train import VarformerTrainer
        return VarformerTrainer(population=population, config_overrides=config_overrides, output_dir=output_dir)

    @classmethod
    def tune(cls, population, n_trials=100, **kwargs):
        """Run hyperparameter search via Optuna; return best params + metric."""
        from varformer.training.tune import tune as _tune
        import os
        os.environ["VARFORMER_POPULATION"] = population
        result = _tune()
        return {
            "best_params": getattr(result, "best_params", {}),
            "best_metric": getattr(result, "best_value", float("nan")),
        }

    def predict(self, genes, return_attention=False):
        """Run inference on the given gene list using local features.

        Returns {gene_id: {"prediction", "classification", "z_var", "attn_weights"?}}.
        Outputs are JSON-serializable (Python primitives + numpy arrays).
        """
        from varformer.inference.predict import predict_subset
        return predict_subset(self, genes, return_attention=return_attention)

    def evaluate(self, test_set):
        """Run on labelled holdout (pfam | rcnt | pharos), return metric dict."""
        from varformer.inference.evaluate import evaluate_subset
        return evaluate_subset(self, test_set)
