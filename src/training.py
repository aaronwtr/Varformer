"""Shim — most content moved to varformer.training / varformer.inference / paper.baselines.
Delete in Phase 8.
"""
from varformer.training.train import train_model  # noqa: F401
from varformer.training.tune import tune, objective  # noqa: F401
from varformer.inference.predict import run_inference  # noqa: F401


def setup_training(**modules):
    """Legacy dispatcher kept for src/main.py."""
    import torch
    torch.set_float32_matmul_precision('medium')
    config = modules.get('config', None)
    if not config:
        raise ValueError("Config is None. Please provide a valid configuration dictionary.")
    mode = config['hyperparameters']['mode']
    from varformer.data.pipeline import ModuleDataProcessor
    data = ModuleDataProcessor(
        gc=modules.get('gc', False),
        go=modules.get('go', False),
        pvc=modules.get('pvc', False),
        psc=modules.get('psc', False),
        config=config,
    ).process()
    import pytorch_lightning as pl
    pl.seed_everything(config['hyperparameters']['seed'])
    if mode == 'eval':
        return train_model(data)
    elif mode == 'inference':
        return run_inference(data)
    else:
        raise ValueError(f"Invalid mode '{mode}' in config. Expected 'eval' or 'inference'.")
