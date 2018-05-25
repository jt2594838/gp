python_file='train/test_map_validate.py'
batch_size=1
classes=3
dataset="anzhen"
# pretrained="-pretrained 1"
model="ResNet101"
model_path="/home/jiangtian/code/gp/res_anzhen/original_model/ResNet101_anzhen_3_200_98.35051569987819.pkl"
weak_model_path="/home/jiangtian/code/gp/res_anzhen/original_model/ResNet101_anzhen_3_10_82.68041237113403.pkl"
map_dir="/home/jiangtian/code/gp/res_anzhen/val_map/"
# preprocess="-preprocess 1"
threshold="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1.0"
apply_method="apply_loss4D"
gpu_no="2"
output="/home/jiangtian/code/gp/res_anzhen/val_rst_recall2/"
repeat=1
criterion="prec"
binary_threshold = "-binary_threshold 1"

python -u ${python_file} \
        -batch_size ${batch_size} \
        -classes ${classes}  \
        -dataset ${dataset} \
        ${pretrained} \
        -model ${model} \
        -model_path ${model_path} \
	-weak_model_path ${weak_model_path}  \
        -map_dir ${map_dir} \
        ${preprocess}   \
        -threshold ${threshold} \
        -apply_method ${apply_method} \
        -gpu_no ${gpu_no}   \
        -repeat ${repeat}   \
	-output ${output}   \
        -criterion  ${criterion}  \
        ${binary_threshold}
