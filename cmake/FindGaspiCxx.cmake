
#[=======================================================================[.rst:
FindGaspiCxx
-------

Finds the GaspiCxx library.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``GaspiCxx::GaspiCxx``
  The GaspiCxx library

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``GaspiCxx_FOUND``
  True if the system has the GaspiCxx library.
``GaspiCxx_INCLUDE_DIRS``
  Include directories needed to use GaspiCxx.
``GaspiCxx_LIBRARIES``
  Libraries needed to link to GaspiCxx.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``GaspiCxx_INCLUDE_DIR``
  The directory containing ``gaspi.h``.
``GaspiCxx_LIBRARY``
  The path to the GaspiCxx library.

#]=======================================================================]

set(GaspiCxx_LIBRARY_NAME "GaspiCxx")
set(GaspiCxx_INCLUDE_NAME "GaspiCxx/Runtime.hpp")

find_path (GaspiCxx_INCLUDE_DIR ${GaspiCxx_INCLUDE_NAME}
              PATHS ${GaspiCxx_ROOT} ${GaspiCxx_DIR} ${GaspiCxx_INCLUDE_PATH}
              PATHS ENV LD_LIBRARY_PATH DYLD_LIBRARY_PATH
              PATH_SUFFIXES include)

find_library (GaspiCxx_LIBRARY ${GaspiCxx_LIBRARY_NAME}
              PATHS ${GaspiCxx_ROOT} ${GaspiCxx_DIR} ${GaspiCxx_LIBRARY_PATH}
              PATHS ENV LD_LIBRARY_PATH DYLD_LIBRARY_PATH
              PATH_SUFFIXES lib lib64)

if (GaspiCxx_LIBRARY)
    message(STATUS "GaspiCxx library path: ${GaspiCxx_LIBRARY}" )
else(GaspiCxx_LIBRARY)
    message(STATUS "GaspiCxx library path: not found" )
endif()


include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set GaspiCxx_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(GaspiCxx DEFAULT_MSG
                                  GaspiCxx_INCLUDE_DIR GaspiCxx_LIBRARY)

mark_as_advanced(GaspiCxx_LIBRARY GaspiCxx_INCLUDE_DIR)
set(GaspiCxx_INCLUDE_DIRS ${GaspiCxx_INCLUDE_DIR} )
set(GaspiCxx_LIBRARIES ${GaspiCxx_LIBRARY} )

message(STATUS "Found GaspiCxx: " ${GaspiCxx_FOUND})

if(GaspiCxx_FOUND AND NOT TARGET GaspiCxx::GaspiCxx)
    find_package(GPI2 REQUIRED)

    add_library(GaspiCxx::GaspiCxx SHARED IMPORTED GLOBAL)
    target_link_libraries(GaspiCxx::GaspiCxx
                          INTERFACE optimized GPI2::GPI2
                                    debug GPI2::GPI2dbg)
    target_include_directories(GaspiCxx::GaspiCxx INTERFACE ${GaspiCxx_INCLUDE_DIRS})
    set_property(TARGET GaspiCxx::GaspiCxx PROPERTY IMPORTED_LOCATION ${GaspiCxx_LIBRARIES})
endif()
