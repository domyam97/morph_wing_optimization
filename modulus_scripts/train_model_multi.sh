#!/bin/bash

pip3 install nvidia-modulus-sym --user -q

host=$HOSTNAME
echo $host

addr="${host:0:${#host}-4}"
echo $addr

echo "addr"
echo $addr
export SLURM_LAUNCH_NODE_IPADDR=$addr
export HOSTNAME=$addr
python3 morph_wing_multi.py

 
