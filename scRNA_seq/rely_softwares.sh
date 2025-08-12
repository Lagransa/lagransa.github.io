#!/bin/bash

now=$(date "+%Y-%d-%m-%s")

error_check(){
    echo "Error met on line $1, check it." |tee -a "error_$now.log"
    exit 1
}

trap "error_check $LINENO" ERR

download_path=$1
conda_version=$2
conda_install_path=$3
env_name=$4

cd $download_path
echo "Conda install started."

apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
curl "https://repo.anaconda.com/archive/Anaconda3-$conda_version-Linux-x86_64.sh" -o ~/anaconda.sh
./anaconda.sh -b -p $conda_install_path
eval "$($conda_install_path/anaconda/bin/conda shell.bash hook)"
conda init

echo "Setting conda environment."
conda create $env_name
conda activate $env_name

echo "All steps finished."