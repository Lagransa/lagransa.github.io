#!/bin/bash

now=$(date "+%Y-%d-%m-%s")

error_check(){
    echo "Error met on line $1, check it." |tee -a "error_$now.log"
    exit 1
}

trap "error_check $LINENO" ERR

project_path=$1
output_path=$2
cores=$3
3_adapter=$4
5_adapter=$5
error_limit=$6
min_overlap=$7

is_pair_end=false

usage(){
    echo "$0 -s <single end reads>|-p <front reads> <reverse reads>"
    exit 1
}

while getopts "s:p" opt; do
    case $opt in
        s)
            is_pair_end=false
            single_file=$OPTARG
        ;;
        p)
            is_pair_end=true
            head_file=$OPTARG
            rev_file=${!OPTIND}
        ;;
        *)
            usage
    esac
done

if $is_pair_end