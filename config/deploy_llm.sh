#! /bin/bash
PORT=$1
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MODEL_NAME="model"
docker run --gpus all --shm-size 1g -p $PORT:80 -v $CURRENT_DIR:/data ghcr.io/huggingface/text-generation-inference --model-id $MODEL_NAME
