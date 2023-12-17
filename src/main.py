import train


def main(mode="training"):
    if mode == "training":
        train.training()
    elif mode == "tuning":
        train.tuning()
    elif mode == "kfold_training":
        train.kfold_training()
    else:
        raise ValueError("Invalid mode. Pick from 'training' or 'tuning'")


if __name__ == "__main__":
    main(mode="training")

    # TODO:
    #  MLP model
    #  [X] Set up checkpointing to save the model with the best validation auroc
    #  [X] Make sure to save hyperparameters per model in wandb or locally
    #  [X] Hyperparameter tuning
    #  [X] Find out how many epochs to train for (500)
    #  [X] Add gene essentiality feature
    #  [X] Add biological process and molecular function features from HPA (think about implications)
    #  [X] Add cellular target localization features from HPA
    #  [X] Check normalization after introduction of new features
    #  [X] Hold-out test set (ACMG genes)
    #  [X] Slot features into the feature types for spider plots
    #  [ ] Set up cross validation
    #  [ ] Train baseline neural network model
    #  [ ] Introduce self-destillation / self-supervision (see SSL cookbook)
    #  [ ] Add autoencoder (for the variant-level and categorical features)
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
    #  [ ] Add SHAP interpretability
