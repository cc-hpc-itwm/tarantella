#[=======================================================================[.rst:
FindPythonModules
-------

Finds installed PythonModules

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``PythonModules_FOUND``
  True if all the required PythonModules could be loaded.
``PythonModules_modulename_FOUND``
  True if `modulename` could be loaded.
``Python_EXECUTABLE``
  Path to the Python executable.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``GPI2_INCLUDE_DIR``
  The directory containing ``gaspi.h``.
``GPI2_LIBRARY``
  The path to the GPI2 library.

#]=======================================================================]

execute_process(COMMAND sh -c "which python"
                OUTPUT_VARIABLE python_path
                RESULT_VARIABLE result
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE)
if (result EQUAL "0" AND EXISTS ${python_path})
  set(Python_EXECUTABLE "${python_path}")
endif()

set(PythonModules_FOUND TRUE)
if (Python_EXECUTABLE)
  foreach (module IN LISTS PythonModules_FIND_COMPONENTS)
    execute_process(COMMAND ${Python_EXECUTABLE} -c
      "import ${module}"
      RESULT_VARIABLE result
      ERROR_QUIET OUTPUT_QUIET)

    if(result)
      set (PythonModules_${module}_FOUND FALSE)
      set (PythonModules_FOUND FALSE)
    else()
      set (PythonModules_${module}_FOUND TRUE)
    endif()
  endforeach()
endif()

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (PythonModules
  REQUIRED_VARS Python_EXECUTABLE PythonModules_FOUND
  HANDLE_COMPONENTS)
