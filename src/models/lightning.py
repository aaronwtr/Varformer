"""Shim — re-exports the new VarformerLightningModule under the legacy class name.

The .load_from_checkpoint classmethod is patched to use varformer.checkpoints.load_legacy_checkpoint
+ strict=False so pre-refactor checkpoints load correctly.

DELETED legacy classes (no replacement):
  - BaseLightningTargetIdentifier
  - MLPLightningTargetIdentifier
  - VarformerLightningTargetIdentifier
  - ShardedVarformerLightningTargetIdentifier
"""
from pathlib import Path
from varformer.training.lightning_module import VarformerLightningModule
from varformer.checkpoints import load_legacy_checkpoint as _load_legacy_checkpoint


class MultiModalLightningTargetIdentifier(VarformerLightningModule):
    """Legacy alias. Existing callers do `.load_from_checkpoint(path, **kwargs)`."""

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
        # Resolve the checkpoint and remap the legacy state_dict
        ckpt = _load_legacy_checkpoint(checkpoint_path)
        # Pop Lightning-consumed kwargs that are not constructor args
        strict = kwargs.pop("strict", False)
        kwargs.pop("map_location", None)
        kwargs.pop("hparams_file", None)
        # Instantiate with all constructor args (config, num_features_gc, ...)
        instance = cls(**kwargs)
        instance.load_state_dict(ckpt["state_dict"], strict=strict)
        return instance
