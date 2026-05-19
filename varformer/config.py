"""Pydantic config models for Varformer.

Single source of truth:
  - ``configs/default.yml`` — hyperparameters
  - ``configs/paths/{hpc,local}.yml`` — path roots

Path resolution: ``data_root`` and ``ckpt_root`` are top-level in the paths YAML;
per-file paths (``OT_PATH``, ``MISSENSE_MAP``, ...) are derived in this module.
The active profile is selected by the ``VARFORMER_PROFILE`` env var (default
``"local"``).

Both ``Config`` and its nested models support dict-style key access
(``config['hyperparameters']['d_model']``, ``config['paths']['MISSENSE_MAP']``)
in addition to attribute access.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"

Population = Literal["sas", "nfe", "afr", "amr"]


class Hyperparameters(BaseModel):
    # training & optimization
    optimizer: str = "AdamW"
    pusb: bool = True
    precision: str = "16-mixed"
    epochs: int = 100
    gradient_clip_val: Optional[float] = None
    batch_size: int = 128
    grad_accum: Optional[int] = None
    lr_start: float = 1e-4
    lr_end: float = 1e-5
    scheduler: str = "CosineAnnealingLR"
    T0: int = 200
    weight_decay: float = 3e-4
    use_pvc: bool = True

    # architecture
    gc_width: int = 32
    go_width: int = 512
    max_seq_len: int = 1024
    num_encoder_layers: int = 3
    d_model: int = 256
    dim_feedforward: int = 4096
    gv_attn_dim: int = 256
    nhead: int = 8
    depth_cls_head: int = 4
    dropout: float = 0.3
    threshold: float = 0.5

    # logistic regression baseline
    C: float = 1.0
    penalty: str = "l2"
    solver: str = "liblinear"
    max_iter: int = 1000
    class_weight: str = "balanced"

    # misc
    wandb: bool = True
    return_attn: bool = True
    num_workers: int = 0
    seed: int = 57
    mode: Literal["eval", "inference"] = "inference"

    # Dict-style access so callers can use ``config['hyperparameters']['d_model']``.
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return key in self.__class__.model_fields

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class Paths(BaseModel):
    data_root: Path
    ckpt_root: Path

    model_config = {"arbitrary_types_allowed": True}

    @property
    def as_dict(self) -> dict[str, str]:
        """Return paths as a dict keyed by the derived path identifiers.

        All values are strings — callers often concatenate or append to paths
        via f-string formatting, which is awkward with ``pathlib.Path``.
        """
        d = self.data_root
        return {
            "DATA_DIR": str(d),
            "FEATURES_DIR": str(d / "features"),
            "GH_CSQ": str(d / "sas" / "gh_parts" / "processed_gh_data" / "all_csqs_non_filtered.pkl"),
            "POP_DATA": str(d / "processed_pop_data") + "/",
            "GNOMAD_DATA": str(d / "gnomad_data") + "/",
            "VAR_MAP": str(d / "sas" / "gh_parts" / "processed_gh_data" / "variant_to_rs_dict.pkl"),
            "CKPT_PATH": str(self.ckpt_root) + "/",
            "TEST_LABELS_FILE": str(d / "test_data" / "full_test_labels_per_source.pkl"),
            "CITELINE_LABELS": str(d / "labels" / "citeline_manual_labels.pkl"),
            "MISSENSE_MAP": str(d / "sas" / "missense_mutation_map.pkl"),
            "GENE_VAR_MAP": str(d / "sas" / "gene_var_map.pkl"),
            "AM_PATH_ISO": str(d / "alphamissense" / "AlphaMissense_isoforms_hg38.tsv"),
            "AM_PATH_CAN": str(d / "alphamissense" / "AlphaMissense_hg38.tsv"),
            "OT_PATH": str(d / "targetPrioritisation" / "processed" / "merged_opentargets_data.pkl"),
            "PROTEIN_ATLAS_FEATURES": str(d / "hpa" / "proteinatlas.tsv"),
            "TEST_GENES_PATH": str(d / "test_data" / "holdout_genes.xlsx"),
            "TISSUE_EXPRESSION_HPA": str(d / "hpa" / "normal_tissue.tsv"),
            "TISSUE_EXPRESSION_GTEX": str(d / "gtex" / "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_median_tpm.gct.gz"),
        }

    def __getitem__(self, key: str) -> str:
        return self.as_dict[key]

    def __contains__(self, key: str) -> bool:
        return key in self.as_dict


class Config(BaseModel):
    hyperparameters: Hyperparameters
    paths: Paths

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    @classmethod
    def load(
        cls,
        profile: Optional[str] = None,
        hyperparams_override: Optional[dict] = None,
    ) -> "Config":
        if profile is None:
            profile = os.environ.get("VARFORMER_PROFILE", "local")

        with (CONFIGS / "default.yml").open() as f:
            hp_dict = yaml.safe_load(f)["hyperparameters"]
        if hyperparams_override:
            hp_dict.update(hyperparams_override)
        hp = Hyperparameters(**hp_dict)

        with (CONFIGS / "paths" / f"{profile}.yml").open() as f:
            paths_dict = yaml.safe_load(f)
        paths = Paths(
            data_root=Path(paths_dict["data_root"]),
            ckpt_root=Path(paths_dict["ckpt_root"]),
        )

        return cls(hyperparameters=hp, paths=paths)
