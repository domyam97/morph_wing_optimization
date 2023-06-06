#!/bin/bash

source /usr/lib/openfoam/openfoam2212/etc/bashrc

echo "Meshing"
mpirun snappyHexMesh -overwrite -parallel > log.snappyHexMesh

mpirun topoSet -parallel >> log.snappyHexMesh

mpirun patchSummary -parallel > log.patchSummary

mpirun potentialFoam -writephi -parallel  > log.potentialFoam

echo "Checking Mesh"
mpirun checkMesh -parallel > log.checkMesh

echo "Running Solver"
export SOLVER_APP=$(foamDictionary -entry application -value system/controlDict) > /dev/null
mpirun $SOLVER_APP -parallel > log.solver

echo "reconstructing mesh"
reconstructParMesh -latestTime -constant > log.reconstruct
reconstructPar -latestTime -constant >> log.reconstruct

echo "making VTK for export"
foamToVTK -ascii -latestTime -overwrite -cellSet smallVolume

echo "DONE"

