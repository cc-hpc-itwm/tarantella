cmake_minimum_required(VERSION 3.8)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

project(tarantella VERSION 0.6.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -Werror")

option(LINK_IB "Defines whether to link against Infiniband drivers [default: disabled]" off)
option(ENABLE_TESTING "Compile tests [default: disabled]" off)
option(BUILD_DOCS "Build documentation [default: disabled]" off)

set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(CMAKE_BUILD_DIR "${CMAKE_SOURCE_DIR}/build")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(INSTALL_LIB_DIR "${CMAKE_INSTALL_PREFIX}/lib/tarantella")
set(INSTALL_BIN_DIR "${CMAKE_INSTALL_PREFIX}/bin")

find_package(GPI2 REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Tensorflow REQUIRED)

add_subdirectory(${SRC_DIR})
add_subdirectory(${SRC_DIR}/gpi_comm_lib/gpi)
add_subdirectory(${SRC_DIR}/gpi_comm_lib/collectives)
add_subdirectory(${SRC_DIR}/gpi_comm_lib)
add_subdirectory(${SRC_DIR}/gpi_comm_lib/tf_ops)

if (BUILD_DOCS)
  find_package(Sphinx)
  add_subdirectory(docs)
endif()

if (ENABLE_TESTING)
  find_package(Boost 1.61 REQUIRED COMPONENTS
                                   unit_test_framework)
  find_package(PythonModules REQUIRED COMPONENTS 
                                      numpy
                                      pytest)
  enable_testing()
  set(SLEEP_TIME_AFTER_TEST 4)
  add_subdirectory(${CMAKE_SOURCE_DIR}/test)
endif() 

