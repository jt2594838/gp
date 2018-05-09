python_file='train/gen_deeplab.py'
epoch=50
classes=3
data_dir="/home/jt/codes/bs/gp/data/anzhen/merged2"
dataset="anzhen"
model="ResNet101"
model_path="/home/jt/codes/bs/gp/res_anzhen/model"
map_path="./"
limit=0
train="-train 1"
gpu_no="0"

python ${python_file} \
        -classes ${classes}  \
        -data_dir ${data_dir} \
        -dataset ${dataset} \
        ${pretrained} \
        -model ${model} \
        -model_path ${model_path} \
        -map_path ${map_path}  \
        -limit  ${limit}  \
        ${train}  \
        -gpu_no ${gpu_no}