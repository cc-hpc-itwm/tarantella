cmake_minimum_required(VERSION 3.12)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include (read_version)
configure_version(VERSION_FILE "${CMAKE_SOURCE_DIR}/version.py")

project(tarantella VERSION ${TNT_VERSION})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror")

option(ENABLE_TESTING "Compile tests [default: disabled]" off)
option(BUILD_DOCS "Build documentation [default: disabled]" off)

# Set a default build type if none was specified
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting default build type to '${default_build_type}'.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Define build type." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(CMAKE_BUILD_DIR "${CMAKE_SOURCE_DIR}/build")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(INSTALL_LIB_DIR "${CMAKE_INSTALL_PREFIX}/lib/tarantella")
set(INSTALL_BIN_DIR "${CMAKE_INSTALL_PREFIX}/bin")

find_package(GPI2 REQUIRED)
find_package(GaspiCxx REQUIRED)
find_package(PythonModules REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Tensorflow REQUIRED)

add_subdirectory(${SRC_DIR})
add_subdirectory(${SRC_DIR}/gpi_comm_lib)
add_subdirectory(${SRC_DIR}/gpi_comm_lib/tf_ops)

if (BUILD_DOCS)
  find_package(Sphinx)
  add_subdirectory(docs)
endif()

if (ENABLE_TESTING)
  find_package(GPI2 REQUIRED)
  find_package(Boost 1.61 REQUIRED COMPONENTS
                                   unit_test_framework)
  find_package(PythonModules REQUIRED COMPONENTS 
                                      numpy
                                      pytest)
  enable_testing()
  add_subdirectory(${CMAKE_SOURCE_DIR}/test)
endif() 


set(CPACK_SOURCE_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${PROJECT_VERSION}-src")
set(CPACK_SOURCE_GENERATOR "TGZ")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_SOURCE_IGNORE_FILES
    /.git
    ~$
    /__pycache__
    /\\\\.DS_Store
    /\\\\.project
    /\\\\.cproject
    /\\\\.settings
    /\\\\.pytest_cache
    /\\\\.vscode
    /\\\\.pydevproject
    ${CMAKE_SOURCE_DIR}/tmp.*
    ${CMAKE_SOURCE_DIR}/log.*
    ${CMAKE_SOURCE_DIR}/build.*
    ${CMAKE_SOURCE_DIR}/Testing
    ${CMAKE_SOURCE_DIR}/meta
    )
include(CPack)
