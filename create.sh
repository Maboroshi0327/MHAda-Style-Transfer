#!/bin/bash

echo -n "Datasets mount path: "
read -r DATASETS_PATH
echo -n "Image tag: "
read -r tag

docker create --name savstr --ipc host -it --gpus all \
    -v $DATASETS_PATH:/root/datasets \
    -v ./SaVSTr:/root/SaVSTr \
    $tag