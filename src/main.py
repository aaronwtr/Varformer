import training
import testing
import argparse


def main(mode="training", config=None):
    if mode == "training":
        training.train(tag="Standard Training")
    elif mode == "puupl":
        training.train(tag="PUUPL Training")
    elif mode == "tuning":
        training.tune()
    elif mode == "kfold_teacher":
        training.kfold_teacher(pvc=True, go=True, gc=True, config=config)
    elif mode == "kfold_student":
        training.kfold_student()
    elif mode == "logistic_regression":
        training.logistic_regression(pvc=True, go=True, gc=True, config=config)
    elif mode == "testing":
        testing.run_test(pvc=True, go=True, gc=True)
    else:
        raise ValueError("Invalid mode. Pick from 'training' or 'tuning', 'kfold_student',"
                         "'kfold_teacher, 'testing', or 'puupl'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Varformer.")
    parser.add_argument("--mode", type=str, default="kfold_teacher", help="Mode to run the script in.")
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(mode=args.mode, config=args.config)

    # TODO:
    #  MLP model
    #  [X] ! Integrate variant-to-gene mapping into the model end-to-end
    #  [X] Fix preprocessing into four modules
    #      [X] Gene characterisation module (eval)
    #      [X] Gene ontology module (eval)
    #      [X] Population variant characterisation module
    #           [X] Generate AM embeddings
    #           [X] ! Evaluate AM embeddings
    #           [X] Generate ESM AA seq embeddings
    #           [X] Evaluate ESM AA seq embeddings
    #           [X] Generate AF embeddings
    #           [X] Evaluate AF embeddings
    #           [X] Evaluate PVC embeddings (AM + ESM)
    #  [X] Incorporate citeline data and make label distribution plot
    #  [X] Swap out GH datafreeze for all the GH data and make sure to handle LoF and Missense properly
    #       [X] Go through data peculiarities and select columns to keep.
    #       [X] Check if current data parsing works for all data (make unit tests for all columns)
    #  [ ] Implement usage of genotype information
    #       [ ] Prepare AaronWenteler data on the TRE
    #       [ ] Select samples for analysis on TRE
    #       [ ] Upload results for the variants associated with samples to TRE
    #       [ ] Check if allele freqs are calculated using GH genotype data (this might be what we can use)
    #       [ ] Individualise predictions based on genotype data during inference time
    #  [X] Generate test datasets
    #       [X] Get positive samples
    #       [X] Balance the positive samples with randomly sampled negatives. Make sure this works in all modalities
    #  -
    #  -
    #  [ ] Combine modules into ensemble model and evaluate
    #  -
    #  XGBoost baseline:
    #  [ ] Set up training loops
    #  [ ] Set up cross validation
    #  [ ] Make training data different degrees of class imbalance and evaluate. Keep val data as is
    #  [ ] Train final model
    #  [ ] Evaluate on held out test set
    #  [ ] Add SHAP interpretability
