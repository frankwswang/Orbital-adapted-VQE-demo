<<<<<<< HEAD
# Orbital-adapted-VQE-demo

We demonstate three different ways to tackle the problem of finding the minimial energy of BeH2 using a variational quantum eigensolve (VQE) with respect to the STO-3G basis. First is a "vanilla" application of the PennyLane library, just using the default STO-3G basis set parameters, the default.qubit simulator, and the AllSinglesDoubles circuit template for our VQE ansatz. This Hamilotian has 666 Pauli strings to measure and requires 14 qubits. 

For our other two experiments we use a Julia package called [Quiqbox](https://github.com/frankwswang/Quiqbox.jl) to varitionally optimize the basis set parameters of BeH2 in the STO-3G basis with respect to the Hartree-Fock energy. Quiqbox was created by one of our group members, Weishi Wang. We provide functions to read in the optimized basis set parameters into a python enviroment. We believe that due to current limitations of NISQ era devices it is important to fully optimize basis set parameters before running a quantum algorithm. Quiqbox also has the ability to create very flexable basis sets such as floating basis sets and mixed-contracted GTO (linear combination of GTOs with mixed centers or orbital angular momentum). Due to the current limitations of PennyLane's basis set options we were only able to optimize the parameters with respect the the STO-3G basis (larger basis sets that pennylane provides are currently unfeasable for BeH2) but in the future we hope that PennyLane gives users the ability to import more flexible 

=======
# Orbital-adapted-VQE-demo
A demonstration of a variational quantum eigensolver for molecular electronic structure with top-level basis set (orbital) optimization.

## Contributors: [Casey Dowdle](https://github.com/CaseyLeeDowdle), [Weishi Wang](https://github.com/frankwswang)
>>>>>>> 42489f8 (Added different VQEs and formatted the code.)
