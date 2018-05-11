python_file='train/train_generator.py'
batch_size=15
epoch=500
classes=3
val_dir="/home/jt/codes/bs/gp/data/anzhen/merged2"
dataset="anzhen"
in_channels=1
# pretrained="-pretrained 1"
model="Deeplab"
model_path="/home/jt/codes/bs/gp/res_anzhen/generator_model"
train_dir="/home/jt/codes/bs/gp/res_anzhen/train_map/ResNet_anzhen_0_4300_zero_greed_rect_quantity.h5"
val_dir="/home/jt/codes/bs/gp/res_anzhen/train_map/ResNet_anzhen_0_4300_zero_greed_rect_quantity.h5"
description="zero_greed_rect_quantity"
# preprocess="-preprocess 1"
gpu_no="0"
learn_rate=0.005

python -u ${python_file} \
        -batch_size ${batch_size} \
        -lr ${learn_rate}  \
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
