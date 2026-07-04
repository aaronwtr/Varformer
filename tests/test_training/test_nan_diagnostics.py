"""Behavioural tests for NaNDiagnosticsCallback.

The callback must distinguish a *recoverable* fp16 scaled-gradient overflow
(handled by the GradScaler, which skips the step) from a *genuine*
unrecoverable divergence (NaN loss or NaN model parameters).  Only the latter
should stop training.
"""
import torch
import torch.nn as nn

from varformer.training.callbacks import NaNDiagnosticsCallback


class _FakeTrainer:
    def __init__(self):
        self.should_stop = False
        self.current_epoch = 3
        self.global_step = 100
        self.sanity_checking = False
        self.callback_metrics = {}


class _TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2)

    def log(self, *args, **kwargs):  # pl_module.log stub
        pass


def test_scaled_gradient_overflow_does_not_stop_training():
    """inf gradients in on_after_backward are a normal fp16 GradScaler event."""
    cb = NaNDiagnosticsCallback(stop_on_nan=True)
    trainer = _FakeTrainer()
    module = _TinyModule()
    # Simulate a scaled-gradient overflow: one parameter has inf grad.
    for p in module.parameters():
        p.grad = torch.full_like(p, float("inf"))
        break

    cb.on_after_backward(trainer, module)

    assert trainer.should_stop is False, (
        "scaled-gradient overflow must not stop training — the GradScaler "
        "skips the step and lowers the scale"
    )


def test_nan_loss_stops_training():
    """A NaN training loss is genuine, unrecoverable divergence."""
    cb = NaNDiagnosticsCallback(stop_on_nan=True)
    trainer = _FakeTrainer()
    module = _TinyModule()

    cb.on_train_batch_end(
        trainer, module, {"loss": torch.tensor(float("nan"))}, batch=None, batch_idx=0
    )

    assert trainer.should_stop is True


def test_nan_parameter_stops_training():
    """A NaN model parameter after the optimizer step is genuine divergence."""
    cb = NaNDiagnosticsCallback(stop_on_nan=True)
    trainer = _FakeTrainer()
    module = _TinyModule()
    with torch.no_grad():
        next(module.parameters()).fill_(float("nan"))

    cb.on_before_zero_grad(trainer, module, optimizer=None)

    assert trainer.should_stop is True


def test_healthy_step_does_not_stop_training():
    """Finite grads, loss, and params leave training running."""
    cb = NaNDiagnosticsCallback(stop_on_nan=True)
    trainer = _FakeTrainer()
    module = _TinyModule()
    for p in module.parameters():
        p.grad = torch.zeros_like(p)

    cb.on_after_backward(trainer, module)
    cb.on_before_zero_grad(trainer, module, optimizer=None)
    cb.on_train_batch_end(
        trainer, module, {"loss": torch.tensor(0.5)}, batch=None, batch_idx=0
    )

    assert trainer.should_stop is False
