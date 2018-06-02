python_file='train/gen_deeplab.py'
classes=3
data_dir="/home/jiangtian/code/gp/data/anzhen/az_split.test"
dataset="anzhen"
model="ConvDeconvV2"
model_path="/home/jiangtian/code/gp/res_anzhen2/generator_model/ConvDeconvV2_anzhen_3_500_0.21889667121731504_zeroo_sp100_quality.pkl"
map_path="/home/jiangtian/code/gp/res_anzhen2/val_map"
limit=487
# train="-train 1"
gpu_no="5"
description="zero_sp10_qualityT"

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
