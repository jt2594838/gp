python_file='train/gen_map.py'
batch_size=1
train_dir="/home/jiangtian/code/gp/data/anzhen/az_split.val"
dataset="anzhen"
# pretrained="-pretrained 0"
model_path="/home/jiangtian/code/gp/res_anzhen2/original_model/ResNet101_anzhen_3_100_78.89908029398786.pkl"
model_name="ResNet"
apply_method="apply_loss4D"
size=100
processor_name="zero"
gen_method_name="super_pixel_zero"
output_dir="/home/jiangtian/code/gp/res_anzhen2/train_map"
offset=0
length=485
update_err="-update_err 1"
gpu_no="3"
description="zero_sp100_quality"

python -u ${python_file} \
        -batch_size ${batch_size} \
        -train_dir ${train_dir} \
        -dataset ${dataset} \
        ${pretrained} \
        -model_path ${model_path} \
        -model_name ${model_name} \
        -apply_method ${apply_method} \
        -size ${size}  \
        -processor_name ${processor_name}  \
        -gen_method_name ${gen_method_name}  \
        -output_dir ${output_dir} \
        -offset  ${offset}  \
        -length  ${length}  \
        -description ${description}  \
        ${update_err}  \
        -gpu_no ${gpu_no}
