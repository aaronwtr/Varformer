import training
import testing
import argparse

from inference import run_inference_pipeline
from utils import utils


def main(mode="training", config=None, output=None):
    if config:
        config = utils.load_config(config)
    else:
        print("No config file provided, using default config!")
        config = utils.load_default_config()
    if mode == "training":
        if config['hyperparameters']['multiseed']:
            seeds = [7, 32, 42, 85, 482]
            for seed in seeds:
                print(f"Training model with seed: {seed}")
                config["hyperparameters"]["seed"] = seed
                training.setup_training(pvc=True, go=True, gc=True, config=config)
        else:
            training.setup_training(pvc=True, go=True, gc=True, config=config)
    elif mode == "tuning":
        training.tune()
    elif mode == "testing":
        testing.run_test(pvc=True, go=True, gc=True, config=config, extract_genes_only=True)
    elif mode == "inference":
        if not output:
            raise ValueError("For inference mode --output arguments is required.")
        run_inference_pipeline(output=output)
    elif mode == "logistic_regression":
        if config['hyperparameters']['multiseed']:
            seeds = [7, 32, 42, 85, 482]
            for seed in seeds:
                print(f"Training logistic regression model with seed: {seed}")
                config["hyperparameters"]["seed"] = seed
                training.logistic_regression(pvc=True, go=True, gc=True, config=config)
        else:
            training.logistic_regression(pvc=True, go=True, gc=True, config=config)
    elif mode == "random":
        training.random(pvc=True, go=True, gc=True, config=config)
    elif mode == "drugnome_ai":
        training.drugnome_ai(pvc=True, go=True, gc=True, config=config)
    else:
        raise ValueError("Invalid mode. Pick from 'training', 'tuning', 'testing', 'inference', 'random',"
                         "logistic_regression, 'drugnome_ai or others.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Varformer.")
    parser.add_argument("--mode", type=str, default="kfold_teacher", help="Mode to run the script in.")
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    parser.add_argument("--output", type=str, help="Path to save the predictions (required for inference).")
    args = parser.parse_args()
    main(mode=args.mode, config=args.config, output=args.output)

