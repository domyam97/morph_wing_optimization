#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -G 1
#SBATCH --mem=8GB
#SBATCH --time=1:00:00
#SBATCH --output=opt.%j.out
#SBATCH --error=opt.%j.err
#SBATCH --gpu_cmode=shared
#SBATCH -C GPU_GEN:VLT
#SBATCH -C GPU_MEM:32GB

rm $SCRATCH/modulus/outputs/morph-wing_surf_big_results/monitors/*

ml system nvtop
cd direct_opt
singularity exec --nv --pwd ~/fly-by-feel/morph_wing/direct_opt $GROUP_HOME/$USER/simg/modulus_22.09.sif ./optimizer.sh
