#!/bin/bash

roslaunch crazyswarm hover_swarm.launch gaps:=$1 ada:=$2 --dump-params > ~/.ros/gaps_$1_$2_params.yaml
roslaunch crazyswarm hover_swarm.launch gaps:=$1 ada:=$2

