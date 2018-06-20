python_file='train/gen_map_classifier.py'
data_dir="/home/jiangtian/code/gp/data/anzhen/az_split.val"
dataset="H5"
model_path="/home/jiangtian/code/gp/res_anzhen2/original_model/ResNet101_anzhen_3_200_81.19265792566702.pkl"
model_name="ResNet"
apply_method="apply_loss4D"
size=21
processor_name="zero"
gen_method_name="rect_rnd"
output_dir="/home/jiangtian/code/gp/res_anzhen2/train_map2"
offset=0
length=485
# update_err="-update_err 1"
gpu_no="2"
description="zero_rect21_quantity"

python -u ${python_file} \
        -data_dir ${data_dir} \
        -dataset ${dataset} \
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
