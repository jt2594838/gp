python_file='train/train_generator.py'
batch_size=20
epoch=50
classes=3
dataset="anzhen"
in_channels=1
# pretrained="-pretrained 1"
model="ConvDeconvV2"
model_path="/home/jiangtian/code/gp/res_anzhen/generator_model"
train_dir="/home/jiangtian/code/gp/res_anzhen/train_map/ResNet_anzhen_0_4370_zero_super_pixel_zero_super_pixel_greed_quantity.h5"
val_dir="/home/jiangtian/code/gp/res_anzhen/train_map/ResNet_anzhen_0_4370_zero_super_pixel_zero_super_pixel_greed_quantity.h5"
description="zero_super_pixel_quality"
preprocess="-preprocess 1"
gpu_no="3"

python -u ${python_file} \
        -batch_size ${batch_size} \
        -epoch ${epoch}  \
        -classes ${classes}  \
        -dataset ${dataset} \
        -in_channels ${in_channels} \
        ${pretrained} \
        -model ${model} \
        -model_path ${model_path} \
        -train_dir ${train_dir} \
        -val_dir ${val_dir} \
        -description ${description} \
        ${preprocess}   \
        -gpu_no ${gpu_no}
