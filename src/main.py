import training
import testing


def main(mode="training"):
    if mode == "training":
        training.train(tag="Standard Training")
    elif mode == "puupl":
        training.train(tag="PUUPL Training")
    elif mode == "tuning":
        training.tune()
    elif mode == "kfold_teacher":
        training.kfold_teacher()
    elif mode == "kfold_student":
        training.kfold_student()
    elif mode == "testing":
        testing.test()
    else:
        raise ValueError("Invalid mode. Pick from 'training' or 'tuning', 'kfold_student',"
                         "'kfold_teacher, 'testing', or 'puupl'")


if __name__ == "__main__":
    main(mode="kfold_student")

    # TODO:
    #  MLP model
    #  [ ] Evaluate on held out test set
    #  [ ] Compare performance with missense variant pathogenicity autoencoding and without autoencoding
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
    #       [ ] Pathogenicity autoencoder
    #  [ ] Hyperparameter tuning
    #  [ ] Train autoencoders
    #  [ ] Train final model
    #  [ ] Evaluate on held out test set
    #  [ ] Add SHAP interpretability
    #  -
    #  Add tissue-specific / disease-specific data
