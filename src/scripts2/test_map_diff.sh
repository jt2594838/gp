python_file='train/test_map_diff.py'
classes=3
dataset="anzhen"
# pretrained="-pretrained 1"
model="ResNet101"
model_path="/home/jiangtian/code/gp/res_anzhen/original_model/ResNet101_anzhen_3_200_98.35051569987819.pkl"
map_dir="/home/jiangtian/code/gp/res_anzhen/val_map/Deeplab_anzhen_zero_sp100_quality_validate_l4370.h5"
# preprocess="-preprocess 1"
threshold=0.7
apply_method="apply_loss4D"
gpu_no="3"
output="/home/jiangtian/code/gp/res_anzhen/val_diff/Deeplab_anzhen_zero_sp100_quality_validate_l4370.h5"
repeat=1

python -u ${python_file} \
        -classes ${classes}  \
        -dataset ${dataset} \
        ${pretrained} \
        -model ${model} \
        -model_path ${model_path} \
        -map_dir ${map_dir} \
        ${preprocess}   \
        -threshold ${threshold} \
        -apply_method ${apply_method} \
        -gpu_no ${gpu_no}   \
        -repeat ${repeat}   \
	-output ${output}
