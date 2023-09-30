# Replace the following variables based on the paths where you store models and data
# df_path: folder name, where you store your data
# target_ds: dataframe name, the name of the data set you would like to get prediction (in .csv format)
# output_path: file name for the predicted results (in .txt format)

# In this example, we are using the training set named as "target.csv" under the directory "../example/dataset"
#df_path="../data/VariPred/embeds/"
#target_ds="train_downsample_5k_clean.pt"
#output_name="output/VariPred_output_finetuned_5k_clean"

df_path="../data/VariPred"
target_ds="test_downsample_fold_3"
output_name="varipred_output_finetuned_fold_3"


python3 ../models/VariPred/VariPred/main.py \
                -p ${df_path} \
                -i ${target_ds} \
                -o ${output_name}
