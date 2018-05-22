python_file='visualization/export_pic.py'
apply_method_name="apply_loss"
input=""
output=""
use_map="-use_map 1"

python -u ${python_file} \
    -apply_method_name ${apply_method_name} \
    -input ${input}  \
    -output ${output}  \
    ${use_map}