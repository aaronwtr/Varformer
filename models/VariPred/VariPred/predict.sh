# Replace the following variables based on the paths where you store models and data
# df_path: folder name, where you store your data
# target_ds: dataframe name, the name of the data set you would like to get prediction (in .csv format)
# output_path: file name for the predicted results (in .txt format)

# In this example, we are using the training set named as "target.csv" under the directory "../example/dataset"
# kFOLD VALIDATION PREDICTION
#df_path="../data/VariPred"
#target_ds="test_downsample_fold_5"
#output_name="varipred_output_finetuned_fold_5"

# REGULAR PREDICTION
df_path="../data/VariPred/input"
target_ds=$1
output_name="../data/VariPred/output/VariPred_output_$1"


python3 ../models/VariPred/VariPred/main.py \
                -p ${df_path} \
                -i ${target_ds} \
                -o ${output_name}
