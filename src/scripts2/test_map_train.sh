python_file='train/test_map_train.py'
batch_size=50
epoch=50
classes=3
train_dir="/home/jt/codes/bs/gp/data/anzhen/merged2"
val_dir="/home/jt/codes/bs/gp/data/anzhen/merged2"
dataset="anzhen"
in_channels=1
# pretrained="-pretrained 1"
model="ResNet101"
model_path="/home/jt/codes/bs/gp/res_anzhen/model"
val_map_dir="/home/jt/codes/bs/nb/src/train/maps/DeeplabS_CIFAR_10_unpreprocessed_0.09455910949409008_VGG16_0.9_79.11_98.59_validate.h5"
description="l5000"
# preprocess="-preprocess 1"
threshold=0.9
apply_method="apply_loss4D"
gpu_no="0"

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
        -gpu_no ${gpu_no}