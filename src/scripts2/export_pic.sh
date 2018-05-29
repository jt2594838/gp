python_file='visualization/export_pic.py'
apply_method_name="apply_loss"
input="/home/jiangtian/code/gp/res_anzhen/val_map/"
output="/home/jiangtian/code/gp/res_anzhen/val_map_export"
use_map="-use_map 1"

python -u ${python_file} \
    -apply_method ${apply_method_name} \
    -input ${input}  \
    -output ${output}  \
    ${use_map}
