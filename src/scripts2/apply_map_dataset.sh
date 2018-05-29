python_file='train/apply_map_dataset.py'
dataset="anzhen"
dataset_dir="/home/jiangtian/code/gp/data/anzhen/merged2"
apply_method_name="apply_loss4D"
map_dir="/home/jiangtian/code/gp/res_anzhen/gened_train_map/Deeplab_anzhen_zero_superpixel_quantity_100_validate_l4370.h5"
train="-train 1"
limit=4370


python -u ${python_file} \
    -dataset ${dataset}  \
    -dataset_dir ${dataset_dir}  \
    -apply_method_name ${apply_method_name} \
    -map_dir ${map_dir}  \
    ${train}  \
    -limit ${limit}
