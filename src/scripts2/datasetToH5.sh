python_file='process/datasetToH5.py'
input="/home/jiangtian/code/gp/data/val_data"
output="/home/jiangtian/code/gp/data/cifar10_val.h5"
# train="-train 1"
dataset="CIFAR_10"

python -u ${python_file} \
    -dataset ${dataset} \
    -input ${input}  \
    -output ${output}  \
    ${train}
