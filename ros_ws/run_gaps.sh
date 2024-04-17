#!/bin/bash

roslaunch crazyswarm hover_swarm.launch gaps:=$1 --dump-params > ~/.ros/gaps_$1_params.yaml
roslaunch crazyswarm hover_swarm.launch gaps:=$1

