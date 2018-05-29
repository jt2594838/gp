python_file='process/datasetToH5.py'
input="/home/jiangtian/code/gp/res_anzhen/val_map/"
output="/home/jiangtian/code/gp/res_anzhen/val_map_export"
train="-train 1"
dataset="CIFAR_10"

python -u ${python_file} \
    -dataset ${dataset} \
    -input ${input}  \
    -output ${output}  \
    ${train}
