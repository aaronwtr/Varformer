"""Training callbacks for the Varformer model."""
import math

import torch
from pytorch_lightning.callbacks import Callback


def _is_bad(x) -> bool:
    """Return True if x is None, NaN, or +/- inf."""
    if x is None:
        return False
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return False
        x = x.detach().float().item() if x.numel() == 1 else x.detach().float()
        if isinstance(x, torch.Tensor):
            return bool(torch.isnan(x).any() or torch.isinf(x).any())
    try:
        return math.isnan(float(x)) or math.isinf(float(x))
    except (TypeError, ValueError):
        return False


class NaNDiagnosticsCallback(Callback):
    """Watch for NaN/Inf in losses, gradients, and weights during training.

    On the first NaN or Inf observed, dumps a one-time diagnostic block to
    stdout and to the active logger (epoch, global step, per-tensor weight
    and grad norms, and which monitored metrics were affected) so a
    post-mortem doesn't depend on memorising the run.  Subsequent NaN events
    in the same run are counted but not re-dumped.

    Args:
        stop_on_nan: When True (default), call ``trainer.should_stop = True``
            after the first NaN dump so the bad run terminates with the
            diagnostic intact rather than continuing into a NaN floor that
            poisons the best-checkpoint and ``BestThresholdCallback`` state.
    """

    def __init__(self, stop_on_nan: bool = True):
        super().__init__()
        self.stop_on_nan = stop_on_nan
        self._first_nan_logged = False
        self._nan_event_count = 0

    def on_after_backward(self, trainer, pl_module):
        if self._first_nan_logged:
            return

        bad_grad_norms = {}
        for name, p in pl_module.named_parameters():
            if p.grad is None:
                continue
            gnorm = p.grad.detach().float().norm().item()
            if not math.isfinite(gnorm):
                bad_grad_norms[name] = gnorm
        if bad_grad_norms:
            self._dump(trainer, pl_module, source="backward", details={
                "non_finite_grad_norms": bad_grad_norms,
            })

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._first_nan_logged:
            return

        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if _is_bad(loss):
            self._dump(trainer, pl_module, source="train_batch_end", details={
                "train_loss": float(loss) if loss is not None else None,
            })

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or self._first_nan_logged:
            return

        # Only watch metrics whose NaN reliably signals a real training divergence.
        # val_auprc / val_threshold can be NaN as edge-case artefacts of specific
        # validation batches (constant predictions, single-class batches, etc.)
        # without the model itself being broken — treating those as fatal would
        # spuriously kill otherwise-healthy runs.
        bad = {}
        for key in ("val_loss", "val_spearman"):
            v = trainer.callback_metrics.get(key)
            if _is_bad(v):
                bad[key] = float(v) if v is not None else None
        if bad:
            self._dump(trainer, pl_module, source="validation_end", details={
                "non_finite_metrics": bad,
            })

    def _dump(self, trainer, pl_module, source: str, details: dict):
        self._nan_event_count += 1
        if self._first_nan_logged:
            return
        self._first_nan_logged = True

        weight_norms = {
            name: p.detach().float().norm().item()
            for name, p in pl_module.named_parameters()
        }
        max_w = max(weight_norms.values()) if weight_norms else float("nan")
        min_w = min(weight_norms.values()) if weight_norms else float("nan")
        top_norms = sorted(weight_norms.items(), key=lambda kv: -kv[1])[:5]

        msg_lines = [
            "",
            "=" * 78,
            "NaN/Inf detected during training",
            "=" * 78,
            f"  source         : {source}",
            f"  epoch          : {trainer.current_epoch}",
            f"  global_step    : {trainer.global_step}",
            f"  details        : {details}",
            f"  weight norms   : min={min_w:.4g}  max={max_w:.4g}",
            "  top-5 weight norms:",
            *[f"    {n:60s} {v:.4g}" for n, v in top_norms],
            "=" * 78,
            "",
        ]
        print("\n".join(msg_lines))

        try:
            pl_module.log("nan_event_first_epoch", float(trainer.current_epoch))
            pl_module.log("nan_event_first_step", float(trainer.global_step))
            pl_module.log("nan_event_max_weight_norm", float(max_w))
            pl_module.log("nan_event_count", float(self._nan_event_count))
        except Exception:
            # Logging hook may not be available (e.g. inside backward) — best-effort only.
            pass

        if self.stop_on_nan:
            print("NaNDiagnosticsCallback: setting trainer.should_stop = True")
            trainer.should_stop = True


class EmbeddingNormClipCallback(Callback):
    """Clamp mutation-embedding vector norms after each optimizer step.

    Replaces ``nn.Embedding(max_norm=...)`` which does in-place renormalisation
    during ``forward()`` — under bf16 autocast that silently casts the master
    weight from fp32 to bf16, crashing the fused AdamW kernel when it
    encounters mixed parameter dtypes.

    This callback achieves the same per-vector L2 cap but operates on the
    fp32 master weight tensor *after* the optimizer step, outside any autocast
    context, so the dtype stays homogeneous throughout the training loop.

    Args:
        max_norm: Maximum L2 norm per embedding vector.  Vectors exceeding
            this norm are scaled down; vectors within it are left unchanged.
    """

    def __init__(self, max_norm: float):
        super().__init__()
        self.max_norm = max_norm

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        """Clamp embedding norms after optimizer.step(), before zero_grad()."""
        encoder = pl_module.model.varformer
        if encoder is None:
            return
        with torch.no_grad():
            # renorm_ rescales each row (dim=0) so its L2 norm (p=2) does not
            # exceed max_norm.  Equivalent to what nn.Embedding(max_norm=...)
            # does internally, but safe under mixed-precision because we
            # operate on the fp32 master weight outside autocast.
            encoder.mutation_embedding.weight.data.renorm_(2, 0, self.max_norm)


class BestThresholdCallback(Callback):
    """Tracks the classification threshold from the epoch with the best val_spearman score."""

    def __init__(self, monitor='val_spearman', mode='max'):
        self.metric = monitor
        self.best_threshold = None
        if mode == 'max':
            self.best_metric = float('-inf')
        elif mode == 'min':
            self.best_metric = float('inf')
        else:
            raise ValueError("Mode must be either 'max' or 'min'.")

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip during sanity checks
        if trainer.sanity_checking:
            return

        # Get current metrics
        current_metric = trainer.callback_metrics.get(self.metric, 0)
        current_threshold = trainer.callback_metrics.get('val_threshold', None)

        if current_threshold is not None and current_metric > self.best_metric:
            self.best_metric = current_metric
            self.best_threshold = current_threshold

            # Store the best threshold as a hyperparameter in the model
            pl_module.hparams['best_threshold'] = current_threshold

            # Save it also in the model configuration for easy access
            pl_module.model.config['best_threshold'] = current_threshold

            # Log it
            pl_module.log('best_threshold', current_threshold)
