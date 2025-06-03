import training
import testing
import argparse

from inference import run_inference_pipeline
from utils import utils


def main(mode="training", config=None, checkpoint=None, output=None):
    if config:
        config = utils.load_config(config)
    else:
        print("No config file provided, using default config!")
        config = utils.load_default_config()
    if mode == "training":
        training.setup_training(pvc=True, go=True, gc=True, config=config)
    elif mode == "tuning":
        training.tune()
    elif mode == "testing":
        testing.run_test(pvc=True, go=True, gc=True)
    elif mode == "inference":
        if not checkpoint or not output:
            raise ValueError("For inference mode, both --checkpoint and --output arguments are required.")
        run_inference_pipeline(checkpoint=checkpoint, output=output)
    elif mode == "logistic_regression":
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
    parser.add_argument("--checkpoint", type=str, help="Path to the model checkpoint (required for inference).")
    parser.add_argument("--output", type=str, help="Path to save the predictions (required for inference).")
    args = parser.parse_args()
    main(mode=args.mode, config=args.config, checkpoint=args.checkpoint, output=args.output)
