#!/bin/bash

pip3 install nvidia-modulus-sym --user -q

host=$HOSTNAME
echo $host

addr="${host:0:${#host}-4}"
echo $addr

mkdir /tmp/data

echo "addr"
echo $addr
export SLURM_LAUNCH_NODE_IPADDR=$addr
export HOSTNAME=$addr

# Change to morph_wing_surface_vec.py if needed
python3 morph_wing_surface.py

 
