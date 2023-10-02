#!/bin/bash
mkdir /tmp/stls
mkdir /tmp/data

rm $SCRATCH/modulus/outputs/morph-wing_surf_big_results/inferencers/*

host=$HOSTNAME
echo $host

addr="${host:0:${#host}-4}"

echo "addr"
echo $addr
export SLURM_LAUNCH_NODE_IPADDR=$addr
export HOSTNAME=$addr
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6

python3 cfd_optimizer.py

