python_file='train/gen_map.py'
batch_size=50
train_dir="/home/jt/codes/bs/gp/data/anzhen/merged2"
val_dir="/home/jt/codes/bs/gp/data/anzhen/merged2"
dataset="anzhen"
# pretrained="-pretrained 0"
model_path="/home/jt/codes/bs/gp/res_anzhen/model"
model_name="ResNet"
apply_method="apply_loss4D"
size=4
processor_name="zero"
gen_method_name="rect_greed"
output_dir="output/"
offset=0
length=0
update_err="-update_err 1"
gpu_no="0"

python ${python_file} \
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
        ${update_err}  \
        -gpu_no ${gpu_no}