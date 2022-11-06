#!/bin/bash
# set -x
if [ $# -lt 3 ]; then
    echo "usage: $0 num_servers num_workers role root_url "
    exit -1;
fi

export USE_BYTESCHEDULER=1
# export BYTESCHEDULER_TUNING=1
# export BYTESCHEDULER_PARTITION=512000
# export BYTESCHEDULER_CREDIT=4096000
# export BYTESCHEDULER_TIMELINE=timeline.json
# export BYTESCHEDULER_DEBUG=1

RUNFILE="/users/duanqing/byteps/bytescheduler/examples/mxnet-image-classification/train_imagenet.py"

export COMMAND="python ${RUNFILE} --network resnet --num-layers 18 --benchmark 1 \
    --kv-store dist_sync --batch-size 32 --disp-batches 10 --num-examples 1000 --num-epochs 1" 


export DMLC_NUM_SERVER=$1
shift
export DMLC_NUM_WORKER=$1
shift
export DMLC_ROLE=$1
shift
export DMLC_PS_ROOT_URI=$1


export DMLC_PS_ROOT_PORT=8000
# export DMLC_PS_ROOT_URI='127.0.0.1'
# export DMLC_ROLE='scheduler'
$COMMAND &

wait
