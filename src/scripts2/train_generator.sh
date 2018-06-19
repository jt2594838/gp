python_file='train/train_generator.py'
batch_size=20
epoch=500
classes=3
dataset="anzhen"
in_channels=1
# pretrained="-pretrained 1"
model="ConvDeconvV2"
model_path="/home/jiangtian/code/gp/res_anzhen2/generator_model2"
train_dir="/home/jiangtian/code/gp/res_anzhen2/train_map2/ResNet_anzhen_0_485_zero_rect_rnd_zero_rect21_quantity.h5"
val_dir="/home/jiangtian/code/gp/res_anzhen2/train_map2/ResNet_anzhen_0_485_zero_rect_rnd_zero_rect21_quantity.h5"
description="zero_rect21_quantity"
preprocess="-preprocess 1"
gpu_no="2"
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
