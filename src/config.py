# TODO: port to yaml
ELGH_PATH = "../data/elgh/"
MIVA_PATH = f"{ELGH_PATH}all_functional.gatk_PASS.FS_30.DP_0.GQ_20.AB_0.01.functional.missingness_lt_0.genotype_" \
                f"counts.present_in_ELGH.n_transcripts_corrected.txt"
GENOME_PATH = "../data/hg38.fasta"
VP_INFERENCE_PATH = "../models/VariPred/VariPred/predict.sh"
VP_TRAINING_PATH = "../models/VariPred/VariPred/train_VariPred.sh"
VP_OUTPUT_PATH = "../data/VariPred/finetuned_output_1/"
AF_PATH = "../data/alphafold/alphafold_cifs/"


NUM_VP_BATCHES = 1000
