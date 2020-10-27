
#[=======================================================================[.rst:
FindIBverbs
-------

Finds the IBverbs library.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``IBverbs::IBverbs``
  The IBverbs library

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``IBverbs_FOUND``
  True if the system has the IBverbs library.
``IBverbs_INCLUDE_DIRS``
  Include directories needed to use IBverbs.
``IBverbs_LIBRARIES``
  Libraries needed to link to IBverbs.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``IBverbs_INCLUDE_DIR``
  The directory containing the public headers.
``IBverbs_LIBRARY``
  The path to the IBverbs library.

#]=======================================================================]

find_path(IBverbs_INCLUDE_DIR
  NAMES infiniband/verbs.h
  )

find_library(IBverbs_LIBRARY
  NAMES ibverbs)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set IBverbs_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(IBverbs DEFAULT_MSG
                                  IBverbs_INCLUDE_DIR IBverbs_LIBRARY)

mark_as_advanced(IBverbs_INCLUDE_DIR IBverbs_LIBRARY)
set(IBverbs_LIBRARIES ${IBverbs_LIBRARY})
set(IBverbs_INCLUDE_DIRS ${IBverbs_INCLUDE_DIR})

if(IBverbs_FOUND AND NOT TARGET IBverbs::IBverbs)
    add_library(IBverbs::IBverbs SHARED IMPORTED GLOBAL)
    target_include_directories(IBverbs::IBverbs INTERFACE ${IBverbs_INCLUDE_DIRS})
    set_property(TARGET IBverbs::IBverbs PROPERTY IMPORTED_LOCATION ${IBverbs_LIBRARIES})
endif()
