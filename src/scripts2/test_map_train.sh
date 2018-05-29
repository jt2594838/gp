python_file='train/test_map_train.py'
batch_size=50
epoch=200
classes=3
train_dir="/home/jiangtian/code/gp/res_anzhen/gened_train_map/Deeplab_anzhen_zero_superpixel_quantity_100_validate_l4370.h5.applied"
dataset="anzhen"
in_channels=1
# pretrained="-pretrained 1"
model="ResNet101"
model_path="/home/jiangtian/code/gp/res_anzhen/model"
val_map_dir="/home/jiangtian/code/gp/res_anzhen/val_map/Deeplab_anzhen_zero_super_pixel_quantity_100_validate_l4370.h5"
description="l5000"
# preprocess="-preprocess 1"
threshold=0.5
apply_method="apply_loss4D"
gpu_no="1"
output="/home/jiangtian/code/gp/res_anzhen/train_rst/Deeplab_anzhen_zero_super_pixel_quantity_100_validate_l4370.rst"

python -u ${python_file} \
        -batch_size ${batch_size} \
        -epoch ${epoch}  \
        -classes ${classes}  \
        -train_dir ${train_dir} \
        -dataset ${dataset} \
        -in_channels ${in_channels} \
        ${pretrained} \
        -model ${model} \
        -model_path ${model_path} \
        -val_map_dir ${val_map_dir} \
        -description ${description} \
        $(preprocess)   \
        -threshold ${threshold} \
        -apply_method ${apply_method} \
        -gpu_no ${gpu_no}  \
        -output ${output}
