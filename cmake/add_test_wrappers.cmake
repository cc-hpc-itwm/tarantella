include (add_test)

function (tarantella_compile_and_generate_gpi_test)
  set (one_value_options NAME DESCRIPTION TIMEOUT)
  set (multi_value_options LOCALRANKS_LIST SOURCES LIBRARIES INCLUDE_DIRECTORIES
                           SYSTEM_INCLUDE_DIRECTORIES ARGS COMPILE_FLAGS)
  set (required_options NAME SOURCES LOCALRANKS_LIST)
  _parse_arguments (ARG "${options}" "${one_value_options}" 
                        "${multi_value_options}" "${required_options}" ${ARGN})
  _default_if_unset (ARG_TIMEOUT 10)
  set(CLEANUP_TEST_NAME gpi_cleanup)

  set (target_name ${ARG_NAME}.test)
  compile_tarantella_test(${ARGN}
                          NAME ${target_name})

  # wrap call to the test executable in a script that exports the current environment
  # the script can then be executed within a `gaspi_run` call
  set(script_name run_${ARG_NAME}.sh)
  set(script_path ${CMAKE_CURRENT_BINARY_DIR}/${script_name})
  tarantella_gen_test_script(NAME ${script_name}
                             SCRIPT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                             TEST_FILE ${CMAKE_CURRENT_BINARY_DIR}/${target_name})

  message(STATUS "Test: Generating gaspi_run tests for ${ARG_NAME} with ${ARG_LOCALRANKS_LIST} ranks")
  foreach(nlocalranks ${ARG_LOCALRANKS_LIST})
    tarantella_add_gpi_test (NAME ${ARG_NAME}
                NRANKS ${nlocalranks}
                TARGET_FILE ${script_path}
                TEST_FILE "${CMAKE_CURRENT_BINARY_DIR}/${target_name}"
                RUNCOMMAND ${GPI2_GASPI_RUN}
                CLEANUP ${CLEANUP_TEST_NAME}
                TIMEOUT ${ARG_TIMEOUT}
                SLEEP ${SLEEP_TIME_AFTER_TEST})
  endforeach()
endfunction()

function (tarantella_compile_and_generate_test)
  set (one_value_options NAME DESCRIPTION TIMEOUT)
  set (multi_value_options SOURCES LIBRARIES INCLUDE_DIRECTORIES
                           SYSTEM_INCLUDE_DIRECTORIES ARGS COMPILE_FLAGS
                           LABELS)
  set (required_options NAME SOURCES)
  _parse_arguments (ARG "${options}" "${one_value_options}" 
                        "${multi_value_options}" "${required_options}" ${ARGN})
  _default_if_unset (ARG_TIMEOUT 10)

  set (target_name ${ARG_NAME}.test)
  compile_tarantella_test(${ARGN}
                          NAME ${target_name})
  add_test (NAME ${ARG_NAME}
            COMMAND $<TARGET_FILE:${target_name}> ${ARGS})

  # set labels if specified
  if (ARG_LABELS)
    set_property(TEST ${test_name} PROPERTY LABELS ${ARG_LABELS})
  endif()

  # set timeout if specified
  if (ARG_TIMEOUT)
    set_tests_properties(${test_name} PROPERTIES TIMEOUT ${ARG_TIMEOUT})
  endif()
endfunction()

function (tarantella_generate_python_gpi_test)
  set (one_value_options NAME TEST_FILE DESCRIPTION TIMEOUT)
  set (multi_value_options LOCALRANKS_LIST LABELS ARGS)
  set (required_options NAME TEST_FILE LOCALRANKS_LIST)
  _parse_arguments (ARG "${options}" "${one_value_options}" 
                        "${multi_value_options}" "${required_options}" ${ARGN})
  set(CLEANUP_TEST_NAME gpi_cleanup)
  _default_if_unset (ARG_TIMEOUT 600)
  _default_if_unset (ARG_LABELS "Python")

  list(APPEND ARG_LABELS "Python")
  list(REMOVE_DUPLICATES ARG_LABELS)
  
  # wrap call to the test executable in a script that exports the current environment
  # the script can then be executed within a `gaspi_run` call
  set(script_name run_${ARG_NAME}.sh)
  set(script_path ${CMAKE_CURRENT_BINARY_DIR}/${script_name})
  tarantella_gen_test_script(NAME ${script_name}
                             SCRIPT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                             TEST_FILE ${ARG_TEST_FILE}
                             IS_PYTHON_TEST)

  message(STATUS "Test: Generating gaspi_run tests for ${ARG_NAME} with ${ARG_LOCALRANKS_LIST} ranks")
  foreach(nlocalranks ${ARG_LOCALRANKS_LIST})
    tarantella_add_gpi_test (NAME ${ARG_NAME}
                NRANKS ${nlocalranks}
                TARGET_FILE ${script_path}
                TEST_FILE "${ARG_TEST_FILE}"
                RUNCOMMAND ${GPI2_GASPI_RUN}
                TIMEOUT ${ARG_TIMEOUT}
                CLEANUP ${CLEANUP_TEST_NAME}
                SLEEP ${SLEEP_TIME_AFTER_TEST}
                LABELS ${ARG_LABELS})
  endforeach()
endfunction()

function (tarantella_generate_python_test)
  set (one_value_options NAME TEST_FILE DESCRIPTION TIMEOUT)
  set (multi_value_options LABELS ARGS)
  set (required_options NAME TEST_FILE)
  _parse_arguments (ARG "${options}" "${one_value_options}"
                        "${multi_value_options}" "${required_options}" ${ARGN})
  set(CLEANUP_TEST_NAME gpi_cleanup)
  _default_if_unset (ARG_TIMEOUT 600)
  _default_if_unset (ARG_LABELS "Python")

  list(APPEND ARG_LABELS "Python")
  list(REMOVE_DUPLICATES ARG_LABELS)

  # wrap call to the test executable in a script that exports the current environment
  # the script can then be executed within a `gaspi_run` call
  set(script_name run_${ARG_NAME}.sh)
  set(script_path ${CMAKE_CURRENT_BINARY_DIR}/${script_name})
  tarantella_gen_test_script(NAME ${script_name}
                             SCRIPT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                             TEST_FILE ${ARG_TEST_FILE}
                             IS_PYTHON_TEST)

  # create gaspi_run test
  add_test(NAME ${ARG_NAME}
           WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
           COMMAND "${CMAKE_COMMAND}"
             -DRUNCOMMAND=bash
             -DRUNCOMMAND_ARGS=" "
             -DTEST_EXECUTABLE="${script_path}"
             -DTEST_DIR="${CMAKE_BINARY_DIR}"
             -DSLEEP="1"
             -P "${CMAKE_SOURCE_DIR}/cmake/run_test.cmake"
          )

  # set labels if specified
  if (ARG_LABELS)
    set_property(TEST ${ARG_NAME} PROPERTY LABELS ${ARG_LABELS})
  endif()

  # set cleanup fixture script if specified
  if (ARG_CLEANUP)
    set_tests_properties(${ARG_NAME} PROPERTIES FIXTURES_REQUIRED ${ARG_CLEANUP})
  endif()

  # set timeout if specified
  if (ARG_TIMEOUT)
    set_tests_properties(${ARG_NAME} PROPERTIES TIMEOUT ${ARG_TIMEOUT})
  endif()

  message(STATUS "Test: Generating test ${ARG_NAME}")
endfunction()
