
#[=======================================================================[.rst:
FindGPI2
-------

Finds the GPI2 library.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``GPI2::GPI2``
  The GPI2 library

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``GPI2_FOUND``
  True if the system has the GPI2 library.
``GPI2_INCLUDE_DIRS``
  Include directories needed to use GPI2.
``GPI2_LIBRARIES``
  Libraries needed to link to GPI2.
``GPI2_DBG_LIBRARIES``
  Libraries needed to link to the Debug version of GPI2.
``GPI2_GASPI_RUN``
  Path to ``gaspi_run``.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``GPI2_INCLUDE_DIR``
  The directory containing ``gaspi.h``.
``GPI2_LIBRARY``
  The path to the GPI2 library.

#]=======================================================================]

set(GPI2_LIBRARY_NAME "GPI2")
set(GPI2_DBG_LIBRARY_NAME "GPI2-dbg")

FIND_PROGRAM(GASPIRUN_PATH gaspi_run
	PATHS
      $ENV{PATH}
      $ENV{LIB_DIR}/bin
      /usr/local/bin/
      /usr/bin/
      )
          
IF (GASPIRUN_PATH) 
      get_filename_component(GASPIRUN_FOUND_HOME ${GASPIRUN_PATH} DIRECTORY)
      get_filename_component(GPI2_INSTALLED_PATH ${GASPIRUN_FOUND_HOME} DIRECTORY)
      get_filename_component(GPI2_INSTALLED_PATH ${GPI2_INSTALLED_PATH} REALPATH)
ENDIF(GASPIRUN_PATH) 

find_path (GPI2_INCLUDE_DIR GASPI.h
              PATHS ${GPI2_DEFAULT_PATH} ${GPI2_INSTALLED_PATH}
              PATHS ENV LD_LIBRARY_PATH DYLD_LIBRARY_PATH
              PATH_SUFFIXES include)

find_library (GPI2_DBG_LIBRARY ${GPI2_DBG_LIBRARY_NAME}
              PATHS ${GPI2_DEFAULT_PATH} ${GPI2_INSTALLED_PATH}
              PATHS ENV LD_LIBRARY_PATH DYLD_LIBRARY_PATH
              PATH_SUFFIXES lib lib64)

find_library (GPI2_LIBRARY ${GPI2_LIBRARY_NAME}
              PATHS ${GPI2_DEFAULT_PATH} ${GPI2_INSTALLED_PATH}
              PATHS ENV LD_LIBRARY_PATH DYLD_LIBRARY_PATH
              PATH_SUFFIXES lib lib64)

if (GPI2_DBG_LIBRARY)
    message(STATUS "GPI2-dbg library path: ${GPI2_DBG_LIBRARY}" )       
else(GPI2_DBG_LIBRARY)
    message(STATUS "GPI2-dbg  library path: not found" )
endif()


if (GPI2_LIBRARY)
    message(STATUS "GPI2 library path: ${GPI2_LIBRARY}" )    
else(GPI2_LIBRARY)
    message(STATUS "GPI2 library path: not found" )
endif()


include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set GPI2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(GPI2 DEFAULT_MSG
                                  GASPIRUN_PATH
                                  GPI2_DBG_LIBRARY GPI2_LIBRARY)

mark_as_advanced(GPI2_INCLUDE_DIR GASPIRUN_PATH
                 GPI2_DBG_LIBRARY GPI2_LIBRARY)
set(GPI2_INCLUDE_DIRS ${GPI2_INCLUDE_DIR} )
set(GPI2_DBG_LIBRARIES ${GPI2_DBG_LIBRARY} )
set(GPI2_LIBRARIES ${GPI2_LIBRARY} )
set(GPI2_GASPI_RUN ${GASPIRUN_PATH})

message(STATUS "Found GPI2: " ${GPI2_FOUND})

if(GPI2_FOUND AND NOT TARGET GPI2::GPI2)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    add_library(GPI2::GPI2 SHARED IMPORTED GLOBAL)
    target_link_libraries(GPI2::GPI2 INTERFACE Threads::Threads)
    target_include_directories(GPI2::GPI2 INTERFACE ${GPI2_INCLUDE_DIRS})
    set_property(TARGET GPI2::GPI2 PROPERTY IMPORTED_LOCATION ${GPI2_LIBRARIES})

    add_library(GPI2::GPI2dbg SHARED IMPORTED GLOBAL)
    target_link_libraries(GPI2::GPI2dbg INTERFACE Threads::Threads)
    target_include_directories(GPI2::GPI2dbg INTERFACE ${GPI2_INCLUDE_DIRS})
    set_property(TARGET GPI2::GPI2dbg  PROPERTY IMPORTED_LOCATION ${GPI2_DBG_LIBRARIES})

    if (LINK_IB)
        find_package(IBverbs)

        if (IBverbs_FOUND)
            message (STATUS "GPI2: linking against ibverbs")
            target_link_libraries(GPI2::GPI2 INTERFACE IBverbs::IBverbs)
            target_link_libraries(GPI2::GPI2dbg INTERFACE IBverbs::IBverbs)
        else()
            message (FATAL_ERROR "GPI2: could not find ibverbs, disable Infiniband \
                                  support (-DLINK_IB=OFF) to load GPI-2")
        endif()
    else()
        message (STATUS "GPI2: loading library without Infiniband support")
    endif()
endif()
