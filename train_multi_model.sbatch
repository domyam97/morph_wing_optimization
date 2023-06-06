#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -G 1
#SBATCH --mem=16GB
#SBATCH --time=12:00:00
#SBATCH --output=train.%j.out
#SBATCH --error=train.%j.err
#SBATCH --gpu_cmode=shared
#SBATCH -C GPU_BRD:TESLA

echo $SLURM_NPROCS
echo "host"
echo $HOSTNAME
echo "proc IDs"
echo $SLURM_PROCIDS
echo "nodes"
echo $SLURM_JOB_NODELIST

cd modulus

singularity exec --nv --pwd ~/fly-by-feel/morph_wing/modulus $GROUP_HOME/$USER/simg/modulus_22.09.sif ./train_model_multi.sh

