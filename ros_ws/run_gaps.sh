#!/bin/bash

roslaunch crazyswarm hover_swarm.launch prefix:=$1 --dump-params > ~/.ros/$1_params.yaml
roslaunch crazyswarm hover_swarm.launch prefix:=$1

