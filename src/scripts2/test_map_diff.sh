python_file='train/test_map_diff.py'
classes=3
dataset="H5"
# pretrained="-pretrained 1"
model="ResNet101"
model_path="/home/jiangtian/code/gp/res_anzhen2/original_model/ResNet101_anzhen_3_200_81.19265792566702.pkl"
# model_path="/home/jiangtian/code/gp/res_anzhen2/original_model/ResNet101_anzhen_3_100_78.89908029398786.pkl"
map_dir="/home/jiangtian/code/gp/res_anzhen2/val_map/Deeplab_anzhen_zero_rect8_quality_validate_l487.h5"
# preprocess="-preprocess 1"
threshold=0.01
apply_method="apply_loss4D"
gpu_no="3"
output="/home/jiangtian/code/gp/res_anzhen2/val_diff/Deeplab_anzhen_zero_rect8_quality_validate_l487.h5"
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
