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
        training.kfold_teacher(gc=True)
    elif mode == "kfold_student":
        training.kfold_student()
    elif mode == "testing":
        testing.test()
    else:
        raise ValueError("Invalid mode. Pick from 'training' or 'tuning', 'kfold_student',"
                         "'kfold_teacher, 'testing', or 'puupl'")


if __name__ == "__main__":
    main(mode="kfold_teacher")

    # TODO:
    #  MLP model
    #  [X] Fix preprocessing into four modules
    #      [X] Gene characterisation module
    #      [X] Gene ontology module
    #      [ ] Population variant characterisation module
    #           [X] Generate AM embeddings
    #           [X] Evaluate AM embeddings
    #           [ ] Generate ESM AA seq embeddings
    #           [ ] Evaluate ESM AA seq embeddings
    #           [ ] Evaluate PVC embeddings (AM + ESM)
    #      [ ] Protein structure characterisation module
    #           [ ] Generate AF protein structure embeddings
    #           [ ] Evaluate AF protein structure embeddings
    #  [X] Incorporate citeline data and make label distribution plot
    #  [ ] Swap out GH datafreeze for all the GH data and make sure to handle LoF and Missense properly
    #       [ ] Go through data peculiarities and select columns to keep.
    #       [ ] Check if current data parsing works for all data (make unit tests for all columns)
    #  [ ] Implement usage of genotype information
    #       [ ] Prepare AaronWenteler data on the TRE
    #       [ ] Check if allele freqs are calculated using GH genotype data (this might be what we can use)
    #       [ ] Individualise predictions based on genotype data during inference time
    #  [ ] Implement all the test datasets
    #  -
    #  [ ] Train and test each module separately
    #      [ ] Gene characterisation module
    #      [ ] Gene ontology module
    #      [ ] Protein structure characterisation module
    #      [ ] Population variant characterisation module
    #  -
    #  Note: Each module should output a data object that can be used to train the model
    #  [X] Setup feature loading for pathogenicity features from training.py to preprocessing.py
    #  [X] Test the feature preprocessing pipeline
    #  [ ] Get baseline evaluation metrics at train time and test time for each module
    #  [ ] Combine modules into ensemble model and evaluate
    #  -
    #  XGBoost baseline:
    #  [ ] Set up training loops
    #  [ ] Set up cross validation
    #  [ ] Make training data different degrees of class imbalance and evaluate. Keep val data as is
    #  [ ] Train final model
    #  [ ] Evaluate on held out test set
    #  [ ] Add SHAP interpretability
