python_file='train/train_generator.py'
batch_size=50
epoch=50
classes=3
val_dir="/home/jt/codes/bs/gp/data/anzhen/merged2"
dataset="anzhen"
in_channels=1
# pretrained="-pretrained 1"
model="ResNet101"
model_path="/home/jt/codes/bs/nb/src/train/models/VGG16_CIFAR_10_10_10_78.84_98.48.pkl"
train_dir="/home/jt/codes/bs/nb/src/train/maps/DeeplabS_CIFAR_10_unpreprocessed_0.09455910949409008_VGG16_0.9_79.11_98.59_validate.h5"
val_dir="./data/val_data/"
description="l5000"
# preprocess="-preprocess 1"
gpu_no="0"

python ${python_file} \
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