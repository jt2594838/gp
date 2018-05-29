python_file='process/sample_anzhen.py'
input='/home/jiangtian/code/gp/data/anzhen/merged2'
sample_rate=0.2
output='/home/jiangtian/code/gp/data/anzhen/sample_0.2'

python -u ${python_file} \
        -input ${input} \
        -sample_rate ${sample_rate}  \
        -output ${output} 
