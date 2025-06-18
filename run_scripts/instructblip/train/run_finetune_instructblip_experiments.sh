#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Please provide a benchmark and an experiment number as arguments."
    echo "Usage: bash script_name.sh <benchmark> <experiment>"
    exit 1
fi

benchmark=$1
experiment=$2

export TORCH_HOME=/output/torch
export HUGGINGFACE_HUB_CACHE=/output/huggingface

# The directory to copy from
dir=/output/results/${benchmark}/${benchmark}_${experiment}
# The directory to copy to
dest=/input/results/${benchmark}/${benchmark}_${experiment}

mkdir -p $dir
mkdir -p $dest
touch $dir/${benchmark}_${experiment}.log
export TOKENIZERS_PARALLELISM=false
python3 -m torch.distributed.run --nproc_per_node=2 --master_port=29501 train.py --cfg-path lavis/projects/instructblip/train/${benchmark}/finetune_instructblip_${benchmark}_${experiment}.yaml 2>&1 | tee $dir/${benchmark}_$experiment.log

rsync -av --no-o --no-g --chmod=777 --exclude='*.pth' $dir/ $dest/
