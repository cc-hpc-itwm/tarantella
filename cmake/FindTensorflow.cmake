
#[=======================================================================[.rst:
FindTensorflow
-------

Finds the Tensorflow package as described in:
https://www.tensorflow.org/guide/create_op#compile_the_op_using_your_system_compiler_tensorflow_binary_installation


Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``Tensorflow::Tensorflow``
  The Tensorflow library.
  The target will set the CXX11_ABI_FLAG according to the ABI used to compile the TensorFlow library.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``Tensorflow_FOUND``
  True if the system has the Tensorflow library.
``Tensorflow_INCLUDE_DIRS``
  Include directories needed to use Tensorflow.
``Tensorflow_LIBRARIES``
  Libraries needed to link to Tensorflow.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``Tensorflow_INCLUDE_DIR``
  The directory containing the Tensorflow library headers.
``Tensorflow_LIBRARY``
  The path to the Tensorflow library.

#]=======================================================================]

execute_process(COMMAND sh -c "which python"
                OUTPUT_VARIABLE python_path
                RESULT_VARIABLE result
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE)
if (result EQUAL "0" AND EXISTS ${python_path})
  set(Python_EXECUTABLE "${python_path}")
endif()

if (Python_EXECUTABLE)
  execute_process(COMMAND ${Python_EXECUTABLE} -c
    "import tensorflow as tf; print(tf.sysconfig.get_lib())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE result_tf_lib
    OUTPUT_VARIABLE Tensorflow_LIBRARY_DIR
    ERROR_QUIET)

  execute_process(COMMAND ${Python_EXECUTABLE} -c
    "import tensorflow as tf; print(tf.sysconfig.get_include())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE result_tf_incl
    OUTPUT_VARIABLE Tensorflow_INCLUDE_DIR
    ERROR_QUIET)

  execute_process(COMMAND ${Python_EXECUTABLE} -c
    "import tensorflow as tf; print(tf.sysconfig.CXX11_ABI_FLAG)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE result_tf_abi_flag
    OUTPUT_VARIABLE Tensorflow_CXX11_ABI_FLAG
    ERROR_QUIET)
endif()

set(Tensorflow_LIBRARY_NAME libtensorflow_framework.so.2)
find_library (Tensorflow_LIBRARY ${Tensorflow_LIBRARY_NAME}
              PATHS ${Tensorflow_LIBRARY_DIR}
              PATHS ENV LD_LIBRARY_PATH DYLD_LIBRARY_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Tensorflow DEFAULT_MSG 
                                  Tensorflow_LIBRARY
                                  Tensorflow_INCLUDE_DIR)

mark_as_advanced(Tensorflow_INCLUDE_DIR Tensorflow_LIBRARY)
set(Tensorflow_INCLUDE_DIRS ${Tensorflow_INCLUDE_DIR} )
set(Tensorflow_LIBRARIES ${Tensorflow_LIBRARY} )

message(STATUS "Found Tensorflow: " ${Tensorflow_FOUND})

if(Tensorflow_FOUND AND NOT TARGET tensorflow_framework)
    add_library(Tensorflow::Tensorflow SHARED IMPORTED GLOBAL)
    target_include_directories(Tensorflow::Tensorflow INTERFACE ${Tensorflow_INCLUDE_DIRS})
    set_property(TARGET Tensorflow::Tensorflow PROPERTY IMPORTED_LOCATION ${Tensorflow_LIBRARIES})

    # Enable libraries that link against the TensorFlow library to use 
    # the correct value of the CXX11_ABI_FLAG.
    # E.g., the official pip TensorFlow packages require CXX11_ABI_FLAG=0,
    # whereas the conda packages set CXX11_ABI_FLAG=1.
    if ("${result_tf_abi_flag}" EQUAL "0")
      target_compile_definitions(Tensorflow::Tensorflow INTERFACE _GLIBCXX_USE_CXX11_ABI=${Tensorflow_CXX11_ABI_FLAG})
    endif()
endif()



