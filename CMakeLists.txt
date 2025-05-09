cmake_minimum_required(VERSION 3.10)
project(fabsim)

# Detects whether this is a top-level project
get_directory_property(FABSIM_PARENT_DIR PARENT_DIRECTORY)
if(NOT FABSIM_PARENT_DIR)
    set(FABSIM_TOPLEVEL_PROJECT ON)
else()
    set(FABSIM_TOPLEVEL_PROJECT OFF)
endif()

# Build tests
option(FABSIM_BUILD_TESTS      "Build fabsim unit test"        ${FABSIM_TOPLEVEL_PROJECT})

### conditionally compile certain modules depending on libraries found on the system
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

### Adding fabsim: choose the path to your local copy fabsim
include(fabsim)

if(FABSIM_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()
