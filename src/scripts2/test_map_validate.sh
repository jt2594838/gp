python_file='train/test_map_validate.py'
batch_size=1
classes=3
dataset="anzhen"
# pretrained="-pretrained 1"
model="ResNet101"
model_path="/home/jiangtian/code/gp/res_anzhen2/merged_model/ResNet101_anzhen_3_200_79.8767945134909.pkl"
weak_model_path="/home/jiangtian/code/gp/res_anzhen2/original_model/ResNet101_anzhen_3_100_78.89908029398786.pkl"
map_dir="/home/jiangtian/code/gp/res_anzhen2/val_map2/"
# preprocess="-preprocess 1"
# threshold="0,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1.0"
threshold="1.0"
apply_method="apply_loss4D"
gpu_no="5"
output="/home/jiangtian/code/gp/res_anzhen2/val_rst3/"
repeat=1
criterion="prec"
# binary_threshold="-binary_threshold 1"

python -u ${python_file} \
        -batch_size ${batch_size} \
        -classes ${classes}  \
        -dataset ${dataset} \
        ${pretrained} \
        -model ${model} \
        -model_path ${model_path} \
	-weak_model_path ${weak_model_path}  \
        -map_dir ${map_dir} \
        ${preprocess}   \
        -threshold ${threshold} \
        -apply_method ${apply_method} \
        -gpu_no ${gpu_no}   \
        -repeat ${repeat}   \
	-output ${output}   \
        -criterion  ${criterion}  \
        ${binary_threshold}
