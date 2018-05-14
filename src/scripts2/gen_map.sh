python_file='train/gen_map.py'
batch_size=1
train_dir="/home/jiangtian/code/gp/data/anzhen/merged2"
val_dir="/home/jiangtian/code/gp/data/anzhen/merged2"
dataset="anzhen"
# pretrained="-pretrained 0"
model_path="/home/jiangtian/code/gp/res_anzhen/original_model/ResNet101_anzhen_3_200_98.35051569987819.pkl"
model_name="ResNet"
apply_method="apply_loss4D"
size=200
processor_name="zero"
gen_method_name="rect_greed"
output_dir="/home/jiangtian/code/gp/res_anzhen/train_map"
offset=0
length=4370
# update_err="-update_err 1"
gpu_no="0"
description="rect_greed_quantity"

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
