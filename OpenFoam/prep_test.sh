#!/bin/bash

export AOA=$(cat aoa.txt)
export Vinf=$(cat vinf.txt)

source /usr/lib/openfoam/openfoam2212/etc/bashrc

./Allclean

echo "Running blockMesh"
blockMesh > log.blockMesh

surfaceTransformPoints  -rotate-y $AOA constant/triSurface/def-morphWing.stl constant/triSurface/morphWingTran.stl > log.surfaceFeature

echo "Extracting Features"
surfaceFeatureExtract >> log.surfaceFeature

decomposePar > log.decomposePar

echo "Creating 0 dirs"
rm -rf processor*/0
ls -d processor* | xargs -I{} cp -r 0.orig {}/0
