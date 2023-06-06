# morph_wing_optimization
A collection of code for optimizing the deflection of a morphing wing

## General Workflow
1) Generate deflected airfoil geometry using ``modulus/airfoilgen.py``
2) Run OpenCFD Simulations and generate VTK results. 
    See ``OpenFoam/prep_test`` and ``OpenFoam/run_parallel``
3) Train Modulus Neural Networks. DeepONet examples in ``modulus``
4) run ``direct_opt/cfd_optimizer.py`` ``optimize_nn_batch()`` methods to optimize on trained NN
