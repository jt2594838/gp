python_file='train/apply_map_h5.py'
dataset=""
dataset_dir=""
apply_method_name="apply_loss4D"
map_dir=""
output_path=""
train="-train 1"
limit=0


python -u ${python_file} \
    -dataset ${dataset}  \
    -dateset_dir ${dataset_dir}  \
    -apply_method_name ${apply_method_name} \
    -map_dir ${map_dir}  \
    -output_path ${output_path}  \
    ${train}  \
    -limit ${limit}