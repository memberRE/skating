#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}
MKL_SERVICE_FORCE_INTEL=1
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
srun --gres=gpu:a100:1 --time 2800 \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
# Any arguments from the third one are captured by ${@:3}
