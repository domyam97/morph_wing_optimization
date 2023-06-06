#!/usr/bin/bash
#SBATCH --job-name=run_sims
#SBATCH --output=sims.%j.out
#SBATCH --error=sims.%j.err
#SBATCH -p normal
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -n 4 
#SBATCH --mem=16GB
#SBATCH --time=1:00:00


cd ~/fly-by-feel/morph-wing
export AOA=4
touch OpenFoam/aoa.txt
echo $AOA > OpenFoam/aoa.txt

export Vinf=10
touch OpenFoam/vinf.txt
echo $Vinf > OpenFoam/vinf.txt

for i in -3 -1 0 1 3
do
    for j in -3 -1 0 1 3
    do
        mkdir ${GROUP_HOME}/${USER}/vtk_res/v_${Vinf}_aoa_${AOA}_d1_${i}_d2_${j}
        
        rm -r OpenFoam/constant/triSurface/*

        cp disk.stl OpenFoam/constant/triSurface/disk.stl
 
        cp def-morph/def_${i}_${j}_morphWing.stl OpenFoam/constant/triSurface/def-morphWing.stl
        
        singularity  exec --pwd ~/fly-by-feel/morph_wing/OpenFoam $GROUP_HOME/$USER/simg/openfoam-run_latest.sif ./prep_test.sh 

        singularity  exec --pwd ~/fly-by-feel/morph_wing/OpenFoam $GROUP_HOME/$USER/simg/openfoam-run_latest.sif ./run_parallel.sh 
    
        cp OpenFoam/VTK/smallVolume_500/internal.vtu ${GROUP_HOME}/${USER}/vtk_res/v_${Vinf}_aoa_${AOA}_d1_${i}_d2_${j}/internal.vtu

        cp OpenFoam/VTK/smallVolume_500/boundary/wing.vtp ${GROUP_HOME}/${USER}/vtk_res/v_${Vinf}_aoa_${AOA}_d1_${i}_d2_${j}/wing.vtp

    done
done

rm OpenFoam/aoa.txt
rm OpenFoam/vinf.txt
