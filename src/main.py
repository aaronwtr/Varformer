import training
import testing
import argparse
from inference import run_inference_pipeline


def main(mode="training", config=None, checkpoint=None, output=None):
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
    elif mode == "random":
        training.random(pvc=True, go=True, gc=True, config=config)
    elif mode == "drugnome_ai":
        training.drugnome_ai(pvc=True, go=True, gc=True, config=config)
    elif mode == "inference":
        if not checkpoint or not output:
            raise ValueError("For inference mode, both --checkpoint and --output arguments are required.")
        run_inference_pipeline(checkpoint=checkpoint, output=output)
    else:
        raise ValueError("Invalid mode. Pick from 'training', 'tuning', 'kfold_student', 'kfold_teacher', "
                         "'testing', 'puupl', 'inference', or others.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Varformer.")
    parser.add_argument("--mode", type=str, default="kfold_teacher", help="Mode to run the script in.")
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    parser.add_argument("--checkpoint", type=str, help="Path to the model checkpoint (required for inference).")
    parser.add_argument("--output", type=str, help="Path to save the predictions (required for inference).")
    args = parser.parse_args()
    main(mode=args.mode, config=args.config, checkpoint=args.checkpoint, output=args.output)
