cmake_minimum_required(VERSION 3.1)

################################################################################

### Configuration
set(FABSIM_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
set(FABSIM_SOURCE_DIR "${FABSIM_ROOT}/src")
set(FABSIM_INCLUDE_DIR "${FABSIM_ROOT}/include")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

################################################################################

### Eigen
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
set (EXTRA_LIBS ${EXTRA_LIBS} Eigen3::Eigen)

### fabsim
file(GLOB BASE ${FABSIM_SOURCE_DIR}/ElasticMembrane.cpp
               ${FABSIM_SOURCE_DIR}/ElasticRod.cpp
               ${FABSIM_SOURCE_DIR}/ElasticShell.cpp
               ${FABSIM_SOURCE_DIR}/IncompressibleNeoHookeanElement.cpp
               ${FABSIM_SOURCE_DIR}/HingeElement.cpp
               ${FABSIM_SOURCE_DIR}/LocalFrame.cpp
               ${FABSIM_SOURCE_DIR}/MassSpring.cpp
               ${FABSIM_SOURCE_DIR}/NeoHookeanElement.cpp
               ${FABSIM_SOURCE_DIR}/OrthotropicStVKMembrane.cpp
               ${FABSIM_SOURCE_DIR}/OrthotropicStVKElement.cpp
               ${FABSIM_SOURCE_DIR}/RodCollection.cpp
               ${FABSIM_SOURCE_DIR}/RodStencil.cpp
               ${FABSIM_SOURCE_DIR}/Spring.cpp
               ${FABSIM_SOURCE_DIR}/StVKElement.cpp)
file(GLOB UTIL ${FABSIM_SOURCE_DIR}/util/filter_var.cpp 
               ${FABSIM_SOURCE_DIR}/util/finite_differences.cpp
               ${FABSIM_SOURCE_DIR}/util/first_fundamental_form.cpp
               ${FABSIM_SOURCE_DIR}/util/principal_directions_and_eigenvalues.cpp
               ${FABSIM_SOURCE_DIR}/util/geometry.cpp
               ${FABSIM_SOURCE_DIR}/util/io.cpp)
add_library(fabsim ${BASE} ${UTIL})

# c++ flags
set_target_properties(fabsim PROPERTIES
                             CXX_STANDARD 14
                             CXX_STANDARD_REQUIRED OFF
                             CXX_EXTENSIONS ON)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  set (EXTRA_LIBS ${EXTRA_LIBS}  ${OpenMP_CXX_LIBRARIES})
endif(OpenMP_CXX_FOUND)

target_link_libraries(fabsim ${EXTRA_LIBS})
target_include_directories(fabsim PUBLIC ${FABSIM_INCLUDE_DIR})
