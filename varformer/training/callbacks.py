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
    """Stop training on genuine divergence and dump a one-time diagnostic.

    Divergence means a model parameter or the training loss has become
    non-finite.  A non-finite *gradient* is not divergence: under fp16-mixed
    AMP the GradScaler skips such a step and lowers the scale, so gradient
    overflow is recorded as a diagnostic note and training continues.

    Args:
        stop_on_nan: When True (default), set ``trainer.should_stop`` after the
            first divergence so the run ends with the diagnostic intact rather
            than continuing into a non-finite floor.
    """

    def __init__(self, stop_on_nan: bool = True):
        super().__init__()
        self.stop_on_nan = stop_on_nan
        self._first_nan_logged = False
        self._grad_overflow_logged = False
        self._nan_event_count = 0

    def on_after_backward(self, trainer, pl_module):
        if self._grad_overflow_logged:
            return
        for name, p in pl_module.named_parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad.detach()).all():
                self._grad_overflow_logged = True
                print(
                    f"[NaNDiagnostics] Scaled-gradient overflow first seen at "
                    f"epoch {trainer.current_epoch}, step {trainer.global_step} "
                    f"(origin: {name}).  Handled by the fp16 GradScaler "
                    f"(step skipped, scale lowered) — not fatal."
                )
                return

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        # Genuine divergence: a model PARAMETER is non-finite after the
        # optimizer step.  If the scaler skipped the step the params are
        # unchanged and finite, so this never false-positives on overflow.
        if self._first_nan_logged:
            return
        for name, p in pl_module.named_parameters():
            if not torch.isfinite(p.detach()).all():
                self._dump(trainer, pl_module, source="parameters", details={
                    "first_nonfinite_param": name,
                })
                return

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
