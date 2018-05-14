python_file='train/test_map_validate.py'
batch_size=50
classes=3
val_dir="/home/jt/codes/bs/gp/data/val_data/"
dataset="CIFAR_10"
# pretrained="-pretrained 1"
model="ResNet101"
model_path="/home/jt/codes/bs/gp/res/models/VGG16_CIFAR_10_10_10_78.84_98.48.pkl"
map_dir="/home/jt/codes/bs/gp/res/maps/Deeplab_CIFAR_10_unpreprocessed_VGG16_validate.h5"
description="l5000"
# preprocess="-preprocess 1"
threshold=0.9,1.0
apply_method="apply_loss4D"
gpu_no="0"
output="./output"
repeat=10

python -u ${python_file} \
        -batch_size ${batch_size} \
        -classes ${classes}  \
        -dataset ${dataset} \
        ${pretrained} \
        -model ${model} \
        -model_path ${model_path} \
        -map_dir ${map_dir} \
        -val_dir ${val_dir} \
        -description ${description} \
        ${preprocess}   \
        -threshold ${threshold} \
        -apply_method ${apply_method} \
        -gpu_no ${gpu_no}  \
        -output ${output}   \
        -repeat ${repeat}