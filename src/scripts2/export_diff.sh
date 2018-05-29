python_file='visualization/export_diff.py'
input="/home/jiangtian/code/gp/res_anzhen/val_diff/"
output="/home/jiangtian/code/gp/res_anzhen/val_diff_export"

python -u ${python_file} \
    -input ${input}  \
    -output ${output}
