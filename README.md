# fabsim - Tools for fabrication & simulation

fabsim is organized in 3 different parts:
* the main classes for physics simulation (mass-spring, discrete rods, ...) are in the root fsim folder
* the optimization files (newton and augmented lagrangian solvers) are in fsim/solver
* all the other helper functions are in fsim/util

It depends on [Eigen](http://eigen.tuxfamily.org/), and optionally on OpenMP and Cholmod

## How to build

If your project uses CMake, simply add 
```
add_subdirectory("path/to/fabsim")
// ...
target_link_libraries(YOUR_TARGET fabsim)
```
to your ```CMakeLists.txt```

To get started, you can try the [sample project]()