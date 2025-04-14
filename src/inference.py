import torch
import yaml
from models.lightning import MultiModalLightningTargetIdentifier
from dataloader import ModuleDataProcessor
from pytorch_lightning import Trainer


def load_model(checkpoint_path, config_path):
    # Load configuration
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    # Initialize model
    model = MultiModalLightningTargetIdentifier.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        num_samples_per_class=None,  # Replace with actual values if needed
        num_features_gc=config['hyperparameters']['gc_width'],
        num_features_go=config['hyperparameters']['go_width'],
        num_mutations=config['hyperparameters']['num_mutations'],
        max_seq_len=config['hyperparameters']['max_seq_len'],
        num_genes=config['hyperparameters']['num_genes'],
        num_iters=config['hyperparameters']['num_iters'],
        class_prior=config['hyperparameters']['class_prior']
    )
    return model, config


def prepare_data(config):
    # Process data using ModuleDataProcessor
    data_processor = ModuleDataProcessor(
        gc=True, go=True, pvc=True, psc=False, config=config
    )
    data = data_processor.process()
    return data['test_data']


def run_inference(model, test_data, batch_size=32):
    # Create a PyTorch Lightning Trainer
    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)

    # Run inference
    predictions = trainer.predict(model, dataloaders=test_data, batch_size=batch_size)
    return predictions


def run_inference_pipeline(config, checkpoint, output):
    # Load model and configuration
    model, config = load_model(checkpoint, config)

    # Prepare test data
    test_data = prepare_data(config)

    # Run inference
    predictions = run_inference(model, test_data)

    # Save predictions
    torch.save(predictions, output)
    print(f"Predictions saved to {output}")
