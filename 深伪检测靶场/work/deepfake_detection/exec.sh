#!/bin/sh

docker run -it --rm --gpus all --shm-size 8G \
    -v ./:/df_detect/app/ \
    df_detect:v1.0 bash
