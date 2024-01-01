import train


def main(mode="training"):
    if mode == "training":
        train.training(tag="Standard Training")
    elif mode == "puupl":
        train.training(tag="PUUPL Training")
    elif mode == "distillation":
        train.distillation()
    elif mode == "tuning":
        train.tuning()
    elif mode == "kfold_training":
        train.kfold_training()
    else:
        raise ValueError("Invalid mode. Pick from 'training' or 'tuning'")


if __name__ == "__main__":
    main(mode="distillation")

    # TODO:
    #  MLP model
    #  [X] Set up checkpointing to save the model with the best validation auroc
    #  [X] Make sure to save hyperparameters per model in wandb or locally
    #  [X] Find out how many epochs to train for (100)
    #  [X] Add biological process and molecular function features from HPA (think about implications)
    #  [X] Add cellular target localization features from HPA
    #  [X] Check normalization after introduction of new features
    #  [X] Hold-out test set (ACMG genes)
    #  [X] Slot features into the feature types for spider plots
    #  [X] Set up cross validation
    #  [X] Hold out golden standard 32 drug targets
    #  [X] Check overlap between golden standard and ACMG genes
    #  [X] Check hold out drug targets for common essential genes
    #  [X] Train baseline neural network model
    #  [FAILED] Setup PUUPL (To unstable to use, need a properly trained model to do this. --> Self-distillation)
    #  [ ] Hyperparameter tuning run where we add logging of width/depth ratio and total number of parameters
    #  [ ] Setup self-distillation
    #  [ ] Self-distillation such that probas around the threshold are kept as unlabeled data and in this way
    #      iteratively gather pseudolabels
    #  [ ] Train final models
    #  [ ] Evaluate on held out test set
    #  [ ] Add SHAP interpretability
    #  -
    #  XGBoost baseline:
    #  [ ] Set up training loops
    #  [ ] Hyperparameter tuning
    #  [ ] Set up cross validation
    #  [ ] Make training data different degrees of class imbalance and evaluate. Keep val data as is
    #  [ ] Train final model
    #  [ ] Evaluate on held out test set
    #  [ ] Add SHAP interpretability
    #  -
    #  Autoencoder deep learning model:
    #  [ ] Set up autoencoders for variant-level and categorical features
    #  [ ] Hyperparameter tuning
    #  [ ] Train autoencoders
    #  [ ] Train final model
    #  [ ] Evaluate on held out test set
    #  [ ] Add SHAP interpretability
    #  -
    #  Add tissue-specific / disease-specific data
