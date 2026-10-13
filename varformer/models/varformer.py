"""Varformer: multimodal architecture for drug-target prioritisation."""
from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from varformer.models.g2v_attention import GeneVariantAttention
from varformer.models.variant_encoder import VariantEncoder


class Varformer(nn.Module):
    """Multimodal model for drug-target identification and prioritisation.

    Combines three input branches — genome-context (GC) features, gene-ontology
    (GO) features, and a per-variant transformer (PVC) — to produce a
    target-prioritization score. The score is interpreted by its within-model
    ordering, not as a calibrated probability of programme success. The PVC
    branch encodes a padded sequence of
    population variants through a transformer encoder and then attends over
    those variant embeddings conditioned on the gene-level representation
    (GeneVariantAttention).  Active branches are fused and passed through a
    prioritisation head. Ablation runs may disable any branch, provided at
    least one remains active.

    Most users should construct instances via the class-method factory rather
    than calling ``__init__`` directly:

    Example:
        >>> model = Varformer.from_pretrained("nfe", seed=42)
        >>> predictions = model.predict(["ENSG00000141510", "ENSG00000012048"])
        >>> print(predictions["ENSG00000141510"]["prediction"])
        0.73

    Args:
        config: Dict or Config object supporting ``config['hyperparameters']``
            key access.  Typically built by ``varformer.config.Config.load()``.
        num_features_gc: Width of the GC feature vector per gene.
        num_features_go: Width of the GO feature vector per gene.
        num_mutations: Vocabulary size of mutation encodings (number of unique
            missense mutation types in the missense map).
        max_seq_len: Maximum number of variants per gene; sequences are padded
            or truncated to this length.
        num_genes: Total number of genes in the dataset.  Used for logging
            only; not tied to any learnable parameter shape.
        use_pvc: Whether the variant-context (PVC) branch is active.  When
            ``False`` the model behaves as a GC+GO-only classifier and
            ``VariantEncoder`` / ``GeneVariantAttention`` are not instantiated.
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
        use_gc: bool = True,
        use_go: bool = True,
    ):
        super().__init__()

        self.num_features_gc = num_features_gc
        self.num_features_go = num_features_go
        self.num_mutations = num_mutations
        self.max_seq_len = max_seq_len
        self.num_genes = num_genes
        self.use_gc = use_gc
        self.use_go = use_go
        self.use_pvc = use_pvc
        if not any((use_gc, use_go, use_pvc)):
            raise ValueError("At least one of GC, GO, or PVC must be active.")
        self.config = config
        self.hyperparams = self.config["hyperparameters"]

        # 1. GC projection branch
        gc_width = int(self.hyperparams["gc_width"])
        self.gc_projection = (
            nn.Sequential(
                nn.Linear(self.num_features_gc, gc_width),
                nn.LayerNorm(gc_width),
                nn.ReLU(),
                nn.Dropout(p=float(self.hyperparams["dropout"])),
            )
            if self.use_gc else None
        )

        # 2. GO projection branch
        go_width = int(self.hyperparams["go_width"])
        self.go_projection = (
            nn.Sequential(
                nn.Linear(num_features_go, go_width),
                nn.LayerNorm(go_width),
                nn.ReLU(),
                nn.Dropout(p=float(self.hyperparams["dropout"])),
            )
            if self.use_go else None
        )

        # 3. PVC branch (VariantEncoder + cross-attention) — optional
        gene_feature_dim = (gc_width if self.use_gc else 0) + (go_width if self.use_go else 0)

        if self.use_pvc:
            # Attribute name `self.varformer` is required for state_dict key compatibility
            # (model.varformer.mutation_embedding.weight, etc.).
            mm_max_norm = self.hyperparams.get("mutation_embedding_max_norm", None)
            mm_max_norm = float(mm_max_norm) if mm_max_norm is not None else None
            self.varformer = VariantEncoder(
                max_seq_len=max_seq_len,
                num_muts=num_mutations,
                dropout=float(self.hyperparams["dropout"]),
                d_model=int(self.hyperparams["d_model"]),
                dim_feedforward=int(self.hyperparams["dim_feedforward"]),
                nhead=int(self.hyperparams["nhead"]),
                num_encoder_layers=int(self.hyperparams["num_encoder_layers"]),
                mutation_embedding_max_norm=mm_max_norm,
                variant_token_schema=self.hyperparams.get("variant_token_schema", "original"),
                token_ablate=tuple(self.hyperparams.get("variant_token_ablate", ()) or ()),
            )

            variant_feature_dim = int(self.hyperparams["d_model"])
            attention_dim = int(self.hyperparams["gv_attn_dim"])

            if gene_feature_dim:
                self.gene_variant_attention = GeneVariantAttention(
                    gene_feature_dim=gene_feature_dim,
                    variant_feature_dim=variant_feature_dim,
                    attention_dim=attention_dim,
                    nhead=int(self.hyperparams["nhead"]),
                )
                self.pvc_pool_projection = None
            else:
                # PVC-only control: no gene query is available, so use masked
                # mean pooling followed by a learned projection to the same
                # fusion width as the cross-attention branch.
                self.gene_variant_attention = None
                self.pvc_pool_projection = nn.Linear(variant_feature_dim, attention_dim)
        else:
            self.varformer = None
            self.gene_variant_attention = None
            self.pvc_pool_projection = None
            attention_dim = 0

        # 4. Classification head
        cls_head_layers = []
        depth_cls_head = int(self.hyperparams["depth_cls_head"])
        current_dim = gene_feature_dim + attention_dim
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

        # Allocated last so enabling it cannot perturb the shared RNG stream.
        self.variant_head = (
            nn.Linear(int(self.hyperparams["d_model"]), 1)
            if self.use_pvc and bool(self.hyperparams.get("variant_head", False))
            else None
        )

    def forward(
        self,
        x: dict,
        mask: Optional[torch.Tensor] = None,
    ):
        """Low-level nn.Module forward pass.

        Most users should call ``predict()`` instead; this method is exposed for
        fine-grained control inside Lightning training loops.

        Args:
            x: Feature dict with the following keys:

                * ``"gc"`` — tuple/list whose first element is a float tensor of
                  shape ``[B, num_features_gc]``.
                * ``"go"`` — tuple/list whose first element is a float tensor of
                  shape ``[B, num_features_go]``.
                * ``"pvc"`` — (only when ``use_pvc=True``) dict with keys
                  ``"pathogenicity"``, ``"position"``, and ``"mutation"``, each
                  a tensor of shape ``[B, max_seq_len]``.

            mask: Boolean padding mask of shape ``[B, max_seq_len]`` where
                ``True`` marks padding positions.  Ignored when ``use_pvc=False``.

        Returns:
            A tuple whose length depends on ``config["hyperparameters"]["return_attn"]``:

            * ``return_attn=True``:
              ``(logits, scores, binary_predictions, z_var, attn_weights)``
            * ``return_attn=False``:
              ``(logits, scores, binary_predictions, z_var)``

            Where ``logits`` are raw pre-sigmoid values of shape ``[B]``,
            ``scores`` are sigmoid-activated values in ``[0, 1]``,
            ``binary_predictions`` are thresholded 0/1 floats,
            ``z_var`` is the attended variant embedding of shape ``[B, attention_dim]``
            (or ``None`` when ``use_pvc=False``), and ``attn_weights`` is the
            per-variant attention weight vector of shape ``[B, max_seq_len]``
            (or ``None`` when ``use_pvc=False``).
        """
        device = next(self.parameters()).device

        if mask is not None:
            mask = mask.to(device=device, dtype=torch.bool)

        gene_parts = []
        if self.use_gc:
            gene_parts.append(self.gc_projection(x["gc"][0].to(device)))
        if self.use_go:
            gene_parts.append(self.go_projection(x["go"][0].to(device)))
        z_gene = torch.cat(gene_parts, dim=-1) if gene_parts else None

        variant_scores = None
        if self.use_pvc:
            z_pvc = self.varformer(
                x["pvc"]["pathogenicity"].to(device),
                x["pvc"]["position"].to(device),
                x["pvc"]["mutation"].to(device),
                mask,
                allele_frequency=x["pvc"].get("allele_frequency", None).to(device)
                if x["pvc"].get("allele_frequency", None) is not None else None,
                relative_position=x["pvc"].get("relative_position", None).to(device)
                if x["pvc"].get("relative_position", None) is not None else None,
                frequency_confidence=x["pvc"].get("frequency_confidence", None).to(device)
                if x["pvc"].get("frequency_confidence", None) is not None else None,
            )
            if self.variant_head is not None:
                variant_scores = self.variant_head(z_pvc).squeeze(-1)

            if z_gene is not None:
                z_var, variant_attn_weights = self.gene_variant_attention(
                    z_gene,
                    z_pvc,
                    key_padding_mask=(
                        mask if self.hyperparams.get("mask_cross_attention_padding", True) else None
                    ),
                )
                concatenated_features = torch.cat([z_gene, z_var], dim=-1)
            else:
                valid = (~mask).unsqueeze(-1).to(z_pvc.dtype)
                pooled = (z_pvc * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
                z_var = self.pvc_pool_projection(pooled)
                variant_attn_weights = valid.squeeze(-1) / valid.sum(dim=1).clamp_min(1)
                concatenated_features = z_var
        else:
            z_var = None
            variant_attn_weights = None
            concatenated_features = z_gene

        logits = self.classification_head(concatenated_features).squeeze()
        scores = torch.sigmoid(logits)
        binary_predictions = (scores > float(self.hyperparams["threshold"])).float()

        if self.hyperparams["return_attn"]:
            output = (logits, scores, binary_predictions, z_var, variant_attn_weights)
        else:
            output = (logits, scores, binary_predictions, z_var)
        if self.variant_head is not None:
            output = output + (variant_scores,)
        return output

    # ------------------------------------------------------------------
    # Public SDK class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, population: str, seed=42):
        """Load a locally available checkpoint for a given population and seed.

        Resolves the checkpoint path from the configured checkpoint root
        (``Config.paths.ckpt_root``), builds the model with data-derived
        dimensions, and loads the weights via ``load_checkpoint``.

        Args:
            population: Population identifier.  One of ``"nfe"``, ``"sas"``,
                ``"afr"``, ``"amr"``.
            seed: Which model seed to load.  Accepts two forms:

                * ``int`` — loads the checkpoint for exactly that seed (e.g.
                  ``seed=42``).
                * ``"best"`` — selects the seed whose checkpoint filename
                  encodes the highest ``val_spearman`` score.

        Returns:
            A ``Varformer`` nn.Module instance ready for ``predict()`` and
            ``evaluate()`` calls.

        Raises:
            FileNotFoundError: if no checkpoint matching the (population, seed)
                pair is found under the checkpoint root. Checkpoint availability
                depends on the local installation and applicable data licences.

        Example:
            >>> model = Varformer.from_pretrained("nfe", seed=42)
            >>> predictions = model.predict(["ENSG00000141510"])
            >>> best_model = Varformer.from_pretrained("nfe", seed="best")
        """
        from varformer.config import Config
        from varformer.checkpoints import find_checkpoint, best_seed

        config = Config.load()
        ckpt_root = config.paths.ckpt_root
        if seed == "best":
            seed = best_seed(ckpt_root, population)
        ckpt_path = find_checkpoint(ckpt_root, population, int(seed))
        return cls._build_and_load(config, population, ckpt_path)

    @classmethod
    def from_checkpoint(cls, path):
        """Load a model from an arbitrary checkpoint file path.

        Infers the population from the parent directory name when that name is
        one of ``"nfe"``, ``"sas"``, ``"afr"``, ``"amr"``; falls back to
        ``"nfe"`` otherwise.  Useful when working with checkpoints saved by
        ``VarformerTrainer.fit()`` or custom training runs.

        Args:
            path: Path to a ``.ckpt`` file (string or path-like).  The parent
                directory name is used to infer the population if possible.

        Returns:
            A ``Varformer`` nn.Module instance ready for ``predict()`` and
            ``evaluate()`` calls, with ``_population``, ``_lightning_module``,
            ``_config``, ``_cfg_dict``, and ``_test_loaders`` set as
            non-module attributes.

        Raises:
            FileNotFoundError: if ``path`` does not exist or cannot be loaded
                by PyTorch Lightning.

        Example:
            >>> model = Varformer.from_checkpoint("checkpoints/nfe/seed42-epoch=99-val_spearman=0.61.ckpt")
            >>> predictions = model.predict(["ENSG00000141510"])
        """
        from varformer.config import Config
        from pathlib import Path
        config = Config.load()
        p = Path(path)
        population = p.parent.name if p.parent.name in ("nfe", "sas", "afr", "amr") else "nfe"
        return cls._build_and_load(config, population, p)

    @classmethod
    def _build_and_load(cls, config, population, ckpt_path):
        """Build the LightningModule with data-derived dimensions and load a checkpoint.

        Returns the inner ``Varformer`` nn.Module, with the LightningModule and
        config attached as plain attributes (see the ``object.__setattr__``
        block below).
        """
        import pickle
        import pandas as pd

        from varformer.checkpoints import load_checkpoint, saved_gc_exclusions
        from varformer.training.lightning_module import VarformerLightningModule
        from varformer.data.pipeline import ModuleDataProcessor
        from varformer.data.loaders import ModelPreprocessorInference

        # Downstream code uses mapping access (cfg['hyperparameters']['X'],
        # cfg['paths']['Y']); build that dict view from the Config object.
        cfg = {
            "hyperparameters": {
                **config.hyperparameters.model_dump(),
                "population": population,
                "return_attn": True,
                "mode": "inference",
                "mutation_embedding_max_norm": None,
                "precision": "16-mixed",
            },
            "paths": config.paths.as_dict,
        }

        # The gene inputs must match those the checkpoint was trained with.
        raw_ckpt = load_checkpoint(ckpt_path)
        cfg["hyperparameters"]["gc_exclude_features"] = saved_gc_exclusions(
            raw_ckpt, cfg["hyperparameters"]["gc_exclude_features"]
        )

        # Run the data pipeline to derive the model's input dimensions.
        data = ModuleDataProcessor(gc=True, go=True, pvc=True, config=cfg).process()
        splits = data if isinstance(data, list) else [data]
        first = splits[0]
        num_features_gc = first["train"]["gc"].shape[1] - (1 if "target" in first["train"]["gc"].columns else 0)
        num_features_go = first["train"]["go"].shape[1] - (1 if "target" in first["train"]["go"].columns else 0)
        num_genes = len(first["labels"])

        with open(cfg["paths"]["MISSENSE_MAP"], "rb") as f:
            missense_map = pickle.load(f)
        num_mutations = len(missense_map)

        lm = VarformerLightningModule(
            config=cfg,
            num_samples_per_class=None,
            num_features_gc=num_features_gc,
            num_features_go=num_features_go,
            num_mutations=num_mutations,
            max_seq_len=cfg["hyperparameters"]["max_seq_len"],
            num_genes=num_genes,
            class_prior=first["class_prior"],
            use_pvc=cfg["hyperparameters"].get("use_pvc", True),
            use_gc=cfg["hyperparameters"].get("use_gc", True),
            use_go=cfg["hyperparameters"].get("use_go", True),
        )
        lm.load_state_dict(raw_ckpt["state_dict"], strict=False)

        # Cache the test loaders so predict_subset can reuse them.
        consolidated_data = {
            "gc": pd.concat([first["train"]["gc"], first["test_data"]["gc"]]),
            "go": pd.concat([first["train"]["go"], first["test_data"]["go"]]),
        }
        consolidated_pvc = {**first["train"]["pvc"], **first["test_data"]["pvc"]}
        consolidated_pvc.pop("labels", None)
        test_loaders = ModelPreprocessorInference.create_test_loaders(
            config=cfg,
            consolidated_data=consolidated_data,
            pvc_data=consolidated_pvc,
            torch_dtype=cfg["hyperparameters"]["precision"],
        )

        instance = lm.model
        # We attach the LightningModule and some bookkeeping onto the returned
        # Varformer instance so the SDK (predict/evaluate) can reach them later.
        #
        # nn.Module overrides __setattr__ so that assigning any nn.Module-valued
        # attribute auto-registers it as a submodule.  ``lm`` is a LightningModule
        # (an nn.Module) whose ``lm.model`` *is* this same instance.  A normal
        # ``instance._lightning_module = lm`` would therefore register lm as a
        # child of instance, creating an instance -> lm -> instance reference
        # cycle that makes recursive traversals (.to(), .apply(), .state_dict())
        # infinite-loop.  ``object.__setattr__`` stores the values as plain
        # Python attributes, bypassing that registration, so no cycle forms.
        object.__setattr__(instance, "_population", population)
        object.__setattr__(instance, "_lightning_module", lm)
        object.__setattr__(instance, "_config", config)
        object.__setattr__(instance, "_cfg_dict", cfg)
        object.__setattr__(instance, "_test_loaders", test_loaders)
        return instance

    @classmethod
    def trainer(
        cls,
        population,
        config_overrides=None,
        output_dir=None,
        frozen_split_manifest=None,
    ):
        """Create a ``VarformerTrainer`` configured for the given population.

        Convenience factory that constructs a ``VarformerTrainer`` instance.
        Call ``.fit(seeds=[...])`` on the returned object to start training.

        Args:
            population: Population identifier.  One of ``"nfe"``, ``"sas"``,
                ``"afr"``, ``"amr"``.
            config_overrides: Optional dict of hyperparameter key-value pairs
                that override the defaults loaded from ``Config``.  For example,
                ``{"epochs": 50, "lr_start": 1e-4}``.
            output_dir: Optional directory path where checkpoints will be
                written.  When ``None`` the trainer uses the default checkpoint
                path from ``Config``.
            frozen_split_manifest: Optional exact split manifest. When supplied,
                training fails unless its population, seed, train/validation
                universe, and benchmark genes match the materialized data.

        Returns:
            A ``VarformerTrainer`` instance.

        Example:
            >>> trainer = Varformer.trainer("sas", config_overrides={"epochs": 50})
            >>> ckpt_paths = trainer.fit(seeds=[7, 42, 85])
        """
        from varformer.training.train import VarformerTrainer
        return VarformerTrainer(
            population=population,
            config_overrides=config_overrides,
            output_dir=output_dir,
            frozen_split_manifest=frozen_split_manifest,
        )


    def predict(self, genes, return_attention=False):
        """Run inference on a list of genes using locally cached features.

        Delegates to ``varformer.inference.predict.predict_subset``, which
        reuses the data loaders and Lightning module cached at load time.

        Args:
            genes: List of Ensembl gene IDs (e.g. ``"ENSG00000141510"``) to
                return predictions for.  Genes not present in the cached
                loaders are silently omitted from the result.
            return_attention: When ``True``, include per-variant attention
                weights in each gene's result dict.

        Returns:
            A dict mapping each recognised gene ID to a payload dict with the
            following keys:

            * ``"prediction"`` (``float`` in ``[0, 1]``) — the model's
              sigmoid score for the gene.
            * ``"classification"`` (``int``, ``0`` or ``1``) — binarised
              prediction using the trained decision threshold.
            * ``"z_var"`` (``numpy.ndarray`` of shape ``(d_model,)``) — attended
              variant embedding for the gene.
            * ``"attn_weights"`` (``numpy.ndarray`` of shape ``(max_seq_len,)``)
              — per-variant attention weights.  **Only present when**
              ``return_attention=True``.

            All values are JSON-serialisable (numpy arrays can be converted with
            ``.tolist()``).

        Example:
            >>> model = Varformer.from_pretrained("nfe", seed=42)
            >>> preds = model.predict(["ENSG00000141510", "ENSG00000012048"])
            >>> print(preds["ENSG00000141510"]["prediction"])
            0.73
            >>> preds_attn = model.predict(["ENSG00000141510"], return_attention=True)
            >>> print(preds_attn["ENSG00000141510"]["attn_weights"].shape)
            (512,)
        """
        from varformer.inference.predict import predict_subset
        return predict_subset(self, genes, return_attention=return_attention)

    def score_variants(self, genes=None):
        """Score individual missense variants with the variant head.

        Requires a model trained with ``variant_head=True``.

        Args:
            genes: List of Ensembl gene IDs to score, or ``None`` to score
                every gene in the cached loaders.

        Returns:
            A dict mapping each recognised gene ID to a payload dict with the
            following keys, each a ``numpy.ndarray`` with one entry per
            retained variant in the model's token order (descending
            pathogenicity):

            * ``"variant_scores"`` — the variant-head score; higher values rank
              a variant higher.
            * ``"pathogenicity"``, ``"position"``, ``"mutation"`` — the token
              inputs, for alignment with the source variant table.
        """
        from varformer.inference.predict import score_variants_subset
        return score_variants_subset(self, genes)

    def evaluate(self, test_set: str) -> dict:
        """Evaluate the model on a labelled holdout set and return metrics.

        Runs the model in eval mode over the test partition and computes the
        standard binary-classification metric suite. The model must already
        be loaded with ``from_pretrained()`` or ``from_checkpoint()``.

        Args:
            test_set: Which labelled test partition to score against. One of:
                ``"pfam"``   — Pfam-derived holdout genes.
                ``"rcnt"``   — Recent FDA-approval holdout genes.
                ``"pharos"`` — Pharos chemoinformatics holdout genes.

        Returns:
            Metrics dict with keys ``{"auroc", "auprc", "spearman", "accuracy",
            "recall", "precision", "f1"}``. Values are floats in ``[0, 1]``.

        Raises:
            KeyError: if ``test_set`` is not one of the three supported names.
            FileNotFoundError: if the test-labels pickle file is missing.

        Example:
            >>> model = Varformer.from_pretrained("nfe", seed="best")
            >>> metrics = model.evaluate(test_set="pfam")
            >>> print(metrics["auroc"])
            0.87
        """
        from varformer.inference.evaluate import evaluate_subset
        return evaluate_subset(self, test_set)
