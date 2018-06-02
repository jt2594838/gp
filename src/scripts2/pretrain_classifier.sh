python_file='train/pretrain_classifier.py'
batch_size=30
epoch=200
classes=3
train_dir="/home/jiangtian/code/gp/data/anzhen/az_split.merged_train"
val_dir="/home/jiangtian/code/gp/data/anzhen/az_split.test"
dataset="anzhen"
in_channels=1
# pretrained="-pretrained 0"
model="ResNet101"
model_path="/home/jiangtian/code/gp/res_anzhen2/merged_model"
gpu_no="2"
python -u ${python_file} \
        -batch_size ${batch_size} \
        -epoch ${epoch}  \
        -classes ${classes}  \
        -train_dir ${train_dir} \
	-val_dir ${val_dir}  \
        -dataset ${dataset} \
        -in_channels ${in_channels} \
        ${pretrained} \
        -model ${model} \
        -model_path ${model_path} \
        -gpu_no ${gpu_no}
