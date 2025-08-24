#!/bin/bash

docker pull skim743/piper_sim_mujoco:latest

docker run -it \
           --rm \
           --privileged \
           --net=host \
           --runtime=nvidia \
           --gpus all \
           --name piper_sim_mujoco \
           -e NVIDIA_DRIVER_CAPABILITIES=all \
           -e DISPLAY=$DISPLAY \
           -v /tmp/.X11-unix/:/tmp/.X11-unix/:rw \
           -v "$(pwd)/../user_data:/CS4803ARM_Fall2025/user_data" \
           skim743/piper_sim_mujoco:latest