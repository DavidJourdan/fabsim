# fabsim - Tools for fabrication & simulation

Fabsim is a small library comprising different material models for simulating rods, membranes and shells
Implemented models include:
- Discrete Elastic Rods (both for individual rods and rod networks)
- Discrete Shells
- Saint-Venant Kirchhoff, neo-hookean and incompressible neo-hookean membrane energies
- Mass-spring system
These implementations include computation of the energy, gradients and hessian and are all performed using [Eigen](http://eigen.tuxfamily.org/),
the only dependency of this library. If available, fabsim will also make use of OpenMP to assemble the hessian in parallel.

The goal of this library is to provide a simple interface to manipulate material models that might be difficult to implement from scratch without introducing bugs in the derivatives (e.g. the DER model). All models have an ```energy```, ```gradient``` and ```hessian``` method that take as input a 1D vector stacking all degrees of freedom. These methods can then be used in any optimization library of your choice, see [this project](https://github.com/DavidJourdan/fabsim-example-project) for an example.

## How to build

If your project uses CMake, simply add 
```
add_subdirectory("path/to/fabsim")
// ...
target_link_libraries(YOUR_TARGET fabsim)
```
to your ```CMakeLists.txt```

To get started, you can try the [example project](https://github.com/DavidJourdan/fabsim-example-project)

## Acknowledgements

The code for the neo-hookean membrane energy comes from Etienne Vouga's [libshell](https://github.com/evouga/libshell) library