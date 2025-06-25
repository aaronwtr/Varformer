import torch
import yaml
import pickle as pkl

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary
from torch.utils.data import DataLoader

from preprocessing import GeneCharacterisationPreprocessor, ModelPreprocessorEval
from models.lightning import MultiModalLightningTargetIdentifier
from dataloader import ModuleDataProcessor, MultiModalDataLoader


def load_test_data(go, gc, pvc, psc, config):
    data = ModuleDataProcessor(gc, go, pvc, psc).process()
    return preprocess_test_data(data, config)


def preprocess_test_data(data, config):
    preprocessor = ModelPreprocessorEval(config, data)
    _, _, _, test_combined, _, _ = preprocessor.model_init()
    return test_combined


def load_model(config, data):
    preprocessor = ModelPreprocessorEval(config, data)
    model, train_combined, val_combined, test_combined, hyperparameters, accelerator = preprocessor.model_init()
    return model


def test_model(test_data, model_checkpoint_path, config):
    model = MultiModalLightningTargetIdentifier.load_from_checkpoint(model_checkpoint_path, config=config)
    summary = ModelSummary(model, max_depth=-1)
    print(summary)
    trainer = Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu')
    trainer.test(model=model, dataloaders=test_data["pfam"])
    trainer.test(model=model, dataloaders=test_data["rcnt"])
    trainer.test(model=model, dataloaders=test_data["pharos"])


def run_test(**modules):
    # TODO: Wandb integration
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    gc = modules.get('gc', False)
    go = modules.get('go', False)
    pvc = modules.get('pvc', False)
    psc = modules.get('psc', False)

    test_data = load_test_data(gc, go, pvc, psc, config)

    model_ckpt_path = 'checkpoints/seed42-epoch=98-val_auroc=0.87.ckpt'
    test_model(test_data, model_ckpt_path, config)
