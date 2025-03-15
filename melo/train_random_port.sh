#!/bin/bash

# ランダムポート使用バージョン
CONFIG=$1
GPUS=$2
MODEL_NAME=pretrain_vtuber

# ランダムポートを生成 (10000-65535の範囲)
PORT=$(( RANDOM % 55535 + 10000 ))

# 仮想環境のパス
VENV_PATH="/home/nagashimadaichi/dev/melo-tts/.venv"

while : # auto-resume: the code sometimes crash due to bug of gloo on some gpus
do
# CUDAのマルチプロセス設定を追加
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# マルチプロセッシング方式をspawnに設定
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

echo "Using port: $PORT"

# 仮想環境内のtorchrunを使用
$VENV_PATH/bin/torchrun --nproc_per_node=$GPUS \
        --master_port=$PORT \
    train.py --c $CONFIG --model $MODEL_NAME 

for PID in $(ps -aux | grep $CONFIG | grep python | awk '{print $2}')
do
    echo $PID
    kill -9 $PID
done
sleep 30
done 