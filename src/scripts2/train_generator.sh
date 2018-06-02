python_file='train/train_generator.py'
batch_size=20
epoch=500
classes=3
dataset="anzhen"
in_channels=1
# pretrained="-pretrained 1"
model="ConvDeconvV2"
model_path="/home/jiangtian/code/gp/res_anzhen2/generator_model"
train_dir="/home/jiangtian/code/gp/res_anzhen2/train_map_strong/ResNet_anzhen_0_485_zero_super_pixel_zero_zero_sp100_quality.h5"
val_dir="/home/jiangtian/code/gp/res_anzhen2/train_map_strong/ResNet_anzhen_0_485_zero_super_pixel_zero_zero_sp100_quality.h5"
description="zeroo_sp100_quality"
preprocess="-preprocess 1"
gpu_no="5"
learn_rate=0.005

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
	-lr ${learn_rate}  \
        -gpu_no ${gpu_no}
