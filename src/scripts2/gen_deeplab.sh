python_file='train/gen_deeplab.py'
classes=3
data_dir="/home/jiangtian/code/gp/data/anzhen/merged2"
dataset="anzhen"
model="Deeplab"
model_path="/home/jiangtian/code/gp/res_anzhen/generator_model/ConvDeconvV2_anzhen_3_500_0.0009759001193293189_zero_rect_greed_quality.pkl"
map_path="/home/jiangtian/code/gp/res_anzhen/val_map"
limit=4370
# train="-train 1"
gpu_no="5"
description="zero_rect_quality"

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
