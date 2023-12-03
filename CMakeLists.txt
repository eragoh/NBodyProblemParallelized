cmake_minimum_required(VERSION 3.8)
project(NBodyProblem VERSION 0.1.0 LANGUAGES CXX CUDA)

# Set the install prefix to the build directory
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}")

include(CTest)
enable_testing()

# Standard C++ executable
add_executable(NBodyProblem_CPP src/main.cpp)

# CUDA executable
add_executable(NBodyProblem_CUDA src/main-cuda.cu)

# Specify CUDA architecture for the CUDA executable
target_compile_options(NBodyProblem_CUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=compute_89;-code=sm_89;-allow-unsupported-compiler;-g>)

# OpenMP executable
add_executable(NBodyProblem_OpenMP src/main-openmp.cpp)

# Find and link OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(NBodyProblem_OpenMP PUBLIC OpenMP::OpenMP_CXX)
endif()

# Install targets
install(TARGETS NBodyProblem_CPP NBodyProblem_CUDA NBodyProblem_OpenMP DESTINATION bin)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
