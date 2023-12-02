cmake_minimum_required(VERSION 3.8)
project(NBodyProblem VERSION 0.1.0 LANGUAGES CXX CUDA)

# Set the install prefix to the build directory
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}")

include(CTest)
enable_testing()

# Standard C++ executable
add_executable(NBodyProblem src/main.cpp)

# CUDA executable
add_executable(NBodyProblem-CUDA src/main-cuda.cu)

# OpenMP executable
add_executable(NBodyProblem-OpenMP src/main-openmp.cpp)

# Specify the CUDA standard
set_property(TARGET NBodyProblem-CUDA PROPERTY CUDA_STANDARD 14)

# Find and link OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(NBodyProblem-OpenMP PUBLIC OpenMP::OpenMP_CXX)
endif()

install(TARGETS NBodyProblem NBodyProblem-CUDA NBodyProblem-OpenMP DESTINATION bin)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
