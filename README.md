# fabsim - Tools for fabrication & simulation

[![](https://github.com/DavidJourdan/fabsim/workflows/Build/badge.svg)](https://github.com/DavidJourdan/fabsim/actions)
[![](https://github.com/DavidJourdan/fabsim/workflows/Test/badge.svg)](https://github.com/DavidJourdan/fabsim/actions)
[![codecov](https://codecov.io/gh/DavidJourdan/fabsim/branch/master/graph/badge.svg)](https://codecov.io/gh/DavidJourdan/fabsim)

Fabsim is a small c++14 library comprising different material models for simulating rods, membranes and shells
Implemented models include:
- Discrete Elastic Rods (both for individual rods and rod networks)
- Discrete Shells
- Saint-Venant Kirchhoff, neo-hookean and incompressible neo-hookean membrane energies
- Mass-spring system

These implementations include computation of the energy, gradients and hessian and are all performed using [Eigen](http://eigen.tuxfamily.org/),
the only dependency of this library. If available, fabsim will also make use of OpenMP to assemble the hessian in parallel.

The goal of this library is to provide a simple interface to manipulate material models that might be difficult to implement from scratch without introducing bugs in the derivatives (e.g. the DER model). All models have an ```energy```, ```gradient``` and ```hessian``` method that take as input a 1D vector stacking all degrees of freedom. These methods can then be used in any optimization library of your choice, see [this project](https://github.com/DavidJourdan/fabsim-example-project) for an example.

Due to the stencil-based approach, it is also simple to implement your own material models by deriving from the ElementBase class. Debugging the derivatives is made easy thanks to the finite difference-based tools available in the test suite

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