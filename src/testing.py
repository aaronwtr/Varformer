import torch
import yaml
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from preprocessing import GeneCharacterisationPreprocessor
from models.target_identifier import MultiModalTargetIdentifier
from dataloader import ModuleDataProcessor


def load_test_data(go, gc, pvc, psc):
    data = ModuleDataProcessor(gc, go, pvc, psc).process()
    return data['test_data']


def test_model(test_data, model_checkpoint_path, config):
    # TODO: Figure out how to get the feature dimensions here
    model_init = MultiModalTargetIdentifier(
        config=config,
        num_features_gc=gc_features_dim,
        num_features_go=go_features_dim,
        num_mutations=num_mutations,
        max_seq_len=hyperparams['max_seq_len'],
        num_genes=max_genes_pvc
    )
    model = model_init.load_from_checkpoint(model_checkpoint_path, config=config)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)
    trainer = Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu')
    test_results = trainer.test(model, test_loader)


def run_test(**modules):
    gc = modules.get('gc', False)
    go = modules.get('go', False)
    pvc = modules.get('pvc', False)
    psc = modules.get('psc', False)
    test_data = load_test_data(gc, go, pvc, psc)
    model_ckpt_path = 'checkpoints/seed42-epochepoch=76-val_aurocval_auroc=0.90.ckpt'
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    test_model(test_data, model_ckpt_path, config)
