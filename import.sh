#!/bin/bash

usage()
    {
        echo "Usage: $0 <hostname_of_master_node> <path_to_current_repo> <path_to_anaconda>"
        echo "For example: $0 worker0 /root/repo/tftuner /root/miniconda3"
    }

    if [ $# -ne 3 ];
    then
        usage
        exit
    fi

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels conda-forge
conda env create -f conda_requirement.yaml

source activate tftuner
which python|xargs -i sed -i '1i\export SCRIPT_PYTHON_EXECUTABLE="{}"' bin/common.sh
sed -i "1i\export SELFTF_MASTER_NODE=\"$1\"" bin/common.sh
sed -i "1i\export SELFTF_HOME=\"$2\"" bin/common.sh
echo "source $3/bin/activate \${SELFTF_CONDA_ENV}" >> bin/common.sh
