cmake_minimum_required(VERSION 3.10)
################################################################################

set(TEST_SOURCES
        benchmarks.cpp
        check_finite_differences.cpp
        membranes.cpp
        misc.cpp
        rods.cpp
        shells.cpp
        springs.cpp
        )

foreach(SOURCE IN ITEMS ${TEST_SOURCES})
    # Extract the filename without extension (NAME_WE) as a name for our executable
    get_filename_component(TEST_NAME ${SOURCE} NAME_WE)

    # c++ flags
    set(CMAKE_CXX_STANDARD 17)

    add_definitions (-DCATCH_CONFIG_ENABLE_BENCHMARKING)

    # Add an executable based on the source
    add_executable(${TEST_NAME} main.cpp ${SOURCE})

    # Configure (include headers and link libraries) the test
    target_link_libraries(${TEST_NAME} PRIVATE fabsim)

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})  # the working directory is the out-of-source build directory

endforeach()
