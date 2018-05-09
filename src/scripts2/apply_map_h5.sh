python_file='train/apply_map_h5.py'
apply_method_name="zero"
map_path=""
output_path=""

python -u ${python_file} \
    -apply_method_name ${apply_method_name} \
    -map_path ${map_path}  \
    -output_path ${output_path}