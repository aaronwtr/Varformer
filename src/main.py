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
    #  [X] Fix preprocessing into three modules
    #      [X] Global gene characterisation module
    #      [X] Protein variant and structure module
    #      [X] Gene ontology module
    #  Note: Each module should output a data object that can be used to train the model
    #  [ ] Setup feature loading for pathogenicity feaatures from training.py to preprocessing.py
    #  [ ] Test the feature preprocessing pipeline
    #  [ ] Get baseline evaluation metrics at train time and test time for each module
    #  [ ] Combine modules into ensemble model and evaluate
    #  -
    #  XGBoost baseline:
    #  [ ] Set up training loops
    #  [ ] Hyperparameter tuning
    #  [ ] Set up cross validation
    #  [ ] Make training data different degrees of class imbalance and evaluate. Keep val data as is
    #  [ ] Train final model
    #  [ ] Evaluate on held out test set
    #  [ ] Add SHAP interpretability
