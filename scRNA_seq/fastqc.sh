#!/bin/bash

set -euo pipefail

now=$(date "+%Y-%d-%m-%s")

error_check(){
    echo "Error met on line $1, check it." |tee -a "error_$now.log"
    exit 1
}

trap "error_check $LINENO" ERR

echo "Step FastQC&MultiQC start."

project_path=$1
output_fastqc=$2
output_multiqc=$3


echo "Implementing FastQC..."
cd $project_path
mkdir $output_fastqc
output_path="$project_path/$output_fastqc"

fastqc ./*.fastq.gz -o $output_path > "$output_path/fastqc.log" 2>&1


echo "Implementing MultiQC..."
cd $output_fastqc
multiqc ./* -o "$output_path/$output_multiqc" > "$output_path/$output_multiqc/multiqc.log" 2>&1

echo "Step FastQC&MultiQC done."