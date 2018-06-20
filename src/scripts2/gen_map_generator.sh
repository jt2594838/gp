python_file='train/gen_map_generataor.py'
classes=3
data_dir="/home/jiangtian/code/gp/data/anzhen/az_split.test"
dataset="H5"
model_name="ConvDeconvV2"
model_path="/home/jiangtian/code/gp/res_anzhen2/generator_model2/ConvDeconvV2_anzhen_3_500_0.23465156768049514_zero_sp100_quality_weak.pkl"
map_path="/home/jiangtian/code/gp/res_anzhen2/val_map2"
limit=487
gpu_no="5"
description="zero_sp100_quality_weak"

python -u ${python_file} \
        -classes ${classes}  \
        -data_dir ${data_dir} \
        -dataset ${dataset} \
        ${pretrained} \
        -model_name ${model_name} \
        -model_path ${model_path} \
        -map_path ${map_path}  \
        -limit  ${limit} \
      	-description ${description} \
        -gpu_no ${gpu_no}
