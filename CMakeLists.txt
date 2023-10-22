cmake_minimum_required(VERSION 3.0.0)
project(NBodyProblem VERSION 0.1.0 LANGUAGES C CXX)

# Set the install prefix to the build directory
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}")

include(CTest)
enable_testing()

add_executable(NBodyProblem src/main.cpp)

install(TARGETS NBodyProblem DESTINATION bin)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
