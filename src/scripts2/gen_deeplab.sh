python_file='train/gen_deeplab.py'
classes=3
data_dir="/home/jiangtian/code/gp/data/anzhen/sample_0.2"
dataset="anzhen"
model="Deeplab"
model_path="/home/jiangtian/code/gp/res_anzhen/generator_model/Deeplab_anzhen_3_500_0.02958169884263651_zero_superpixel100_quality_weak.pkl"
map_path="/home/jiangtian/code/gp/res_anzhen/val_map"
limit=4370
# train="-train 1"
gpu_no="1"
description="zero_sp100_quality_weak"

python -u ${python_file} \
        -classes ${classes}  \
        -data_dir ${data_dir} \
        -dataset ${dataset} \
        ${pretrained} \
        -model ${model} \
        -model_path ${model_path} \
        -map_path ${map_path}  \
        -limit  ${limit} \
      	-description ${description} \
        ${train}  \
        -gpu_no ${gpu_no}
