#!/bin/bash

docker run -it \
           --rm \
           --net=host \
           --name piper_sim_mujoco \
           -e DISPLAY=$DISPLAY \
	       -v /tmp/.X11-unix/:/tmp/.X11-unix/:rw \
           -v "$(pwd)/../user_data:/CS4803_ARM/user_data" \
           skim743/piper_sim_mujoco:latest