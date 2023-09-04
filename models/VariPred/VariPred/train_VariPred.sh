# Replace the following variables based on the paths where you store models and data
# df_path: folder name, where you store your data
# train_ds: dataframe name, the name of your training set (in .csv format)
# test_ds: dataframe name, the name of your test set (in .csv format)

# In this example, we are using the training set named as "train.csv" and the test set named as "test.csv" under the directory "../example/dataset"
df_path="../data/VariPred"
train_ds="train_downsample"
test_ds="test_downsample"

# flag -t means set the model to the training mode

python3 ../models/VariPred/VariPred/main.py \
                -p ${df_path}\
                -tr ${train_ds} \
                -ts ${test_ds} \
                -t
