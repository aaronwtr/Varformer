"""Training callbacks for the Varformer model."""
from pytorch_lightning.callbacks import Callback


class BestThresholdCallback(Callback):
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
