python_file='process/split_data.py'
input='/home/jiangtian/code/gp/data/anzhen/merged2'
train_rate=0.8
val_rate=0.1
test_rate=0.1
classes=3
output='/home/jiangtian/code/gp/data/anzhen/az_split'

python -u ${python_file} \
        -input ${input} \
        -train_rate ${train_rate}  \
        -val_rate ${val_rate}  \
        -test_rate ${test_rate}  \
        -classes ${classes}  \
        -output ${output} 
