python_file='train/pretrain_classifier.py'
batch_size=50
epoch=50
classes=3
train_dir="/home/jt/codes/bs/gp/data/anzhen/merged2"
val_dir="/home/jt/codes/bs/gp/data/anzhen/merged2"
dataset="anzhen"
in_channels=1
# pretrained="-pretrained 0"
model="ResNet101"
model_path="/home/jt/codes/bs/gp/res_anzhen/model"
gpu_no="0"

python ${python_file} \
        -batch_size ${batch_size} \
        -epoch ${epoch}  \
        -classes ${classes}  \
        -train_dir ${train_dir} \
        -dataset ${dataset} \
        -in_channels ${in_channels} \
        ${pretrained} \
        -model ${model} \
        -model_path ${model_path} \
        -gpu_no ${gpu_no}