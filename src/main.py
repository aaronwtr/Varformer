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
