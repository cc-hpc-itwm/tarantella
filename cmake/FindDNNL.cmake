# Finds Intel DNNL library
# Martin Kuehn May 2020

find_path(DNNL_INCLUDE_DIR
          NAMES dnnl.hpp
          PATHS ${DNNL_ROOT}
                ENV DNNL_ROOT
                ${DNNL_ROOT_DIR}
                ENV DNNL_ROOT_DIR
          PATH_SUFFIXES include
          DOC "DNNL header files"
)

find_library(DNNL_LIBRARY dnnl
             PATHS ${DNNL_ROOT}
                   ENV DNNL_ROOT
                   ${DNNL_ROOT_DIR}
                   ENV DNNL_ROOT_DIR
             PATH_SUFFIXES lib lib64
             DOC "DNNL library files")

#include (FindPackageHandleStandardArgs)
find_package_handle_standard_args(DNNL
                                  DEFAULT_MSG
                                  DNNL_LIBRARY
                                  DNNL_INCLUDE_DIR)
          
mark_as_advanced(DNNL_INCLUDE_DIR DNNL_LIBRARY)

set(DNNL_INCLUDE_DIRS ${DNNL_INCLUDE_DIR})
set(DNNL_LIBRARIES ${DNNL_LIBRARY})

if(DNNL_FOUND AND NOT TARGET dnnl)
    add_library(dnnl SHARED IMPORTED GLOBAL)
    target_include_directories(dnnl INTERFACE ${DNNL_INCLUDE_DIRS})
    set_property(TARGET dnnl PROPERTY IMPORTED_LOCATION ${DNNL_LIBRARIES})
endif()
