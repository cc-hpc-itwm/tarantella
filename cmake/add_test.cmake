include (parse_arguments)

function (compile_tarantella_test)
  set(one_value_options NAME DESCRIPTION)
  set(multi_value_options SOURCES LIBRARIES INCLUDE_DIRECTORIES
                          SYSTEM_INCLUDE_DIRECTORIES ARGS COMPILE_FLAGS)
  set(required_options NAME SOURCES)

  # save each argument into a variable named "ARG_argname"
  _parse_arguments_with_unknown(ARG "${options}" "${one_value_options}" 
                                    "${multi_value_options}" "${required_options}" ${ARGN})

  _default_if_unset(ARG_DESCRIPTION "${ARG_NAME}")
  set(target_name ${ARG_NAME})

  add_executable (${target_name} ${ARG_SOURCES})
  list (APPEND ARG_LIBRARIES Boost::unit_test_framework
                             Boost::dynamic_linking)
  target_compile_definitions (${target_name} PRIVATE
    "-DBOOST_TEST_MODULE=\"${ARG_DESCRIPTION}\""
    "-DBOOST_TEST_DYN_LINK")

  #! \note Use RPATH for all tests
  set_property (TARGET ${target_name} PROPERTY BUILD_WITH_INSTALL_RPATH true)
  set_property (TARGET ${target_name} APPEND PROPERTY 
                INSTALL_RPATH 
                ${Boost_INCLUDE_DIR}/../lib:${CMAKE_BINARY_DIR})

  if (Boost_VERSION VERSION_EQUAL 1.60 OR Boost_VERSION VERSION_GREATER 1.60)
    list (INSERT ARG_ARGS 0 "--")
  endif()
 
  if (ARG_SYSTEM_INCLUDE_DIRECTORIES)
    target_include_directories (${target_name} SYSTEM
      ${ARG_SYSTEM_INCLUDE_DIRECTORIES})
  endif()
  if (ARG_INCLUDE_DIRECTORIES)
    target_include_directories (${target_name} PRIVATE ${ARG_INCLUDE_DIRECTORIES})
  endif()

  target_link_libraries (${target_name} ${ARG_LIBRARIES})
  if (ARG_COMPILE_FLAGS)
    set_property (TARGET ${target_name} PROPERTY COMPILE_FLAGS ${ARG_COMPILE_FLAGS})
  endif()
endfunction()

function (tarantella_gen_environment_paths)
  set(multi_value_options VARIABLE_LIST)
  set(required_options VARIABLE_LIST)
  _parse_arguments(ARG "${options}" "${one_value_options}" 
                       "${multi_value_options}" "${required_options}" ${ARGN})
  set(env_var_names PATH LIBRARY_PATH LD_LIBRARY_PATH DYLD_LIBRARY_PATH CPATH PYTHONPATH)
  set(env_vars )

  foreach (var_name ${env_var_names})
    if (DEFINED ENV{${var_name}})
      list(APPEND env_vars "${var_name}=$ENV{${var_name}}")
    endif()
  endforeach()
  set(${ARG_VARIABLE_LIST} ${env_vars} PARENT_SCOPE)
endfunction()

function (tarantella_gen_executable_script)
  set(one_value_options SCRIPT_DIR SCRIPT_NAME)
  set(required_options SCRIPT_DIR SCRIPT_NAME)
  _parse_arguments(ARG "${options}" "${one_value_options}" 
                       "${multi_value_options}" "${required_options}" ${ARGN})

  set(tmp_script_path ${CMAKE_CURRENT_BINARY_DIR}/tmp/${ARG_SCRIPT_NAME})
  file(REMOVE ${ARG_SCRIPT_DIR}/${ARG_SCRIPT_NAME})
  file(WRITE ${tmp_script_path} "")
  file(COPY ${tmp_script_path} 
       DESTINATION ${ARG_SCRIPT_DIR}
       FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
       )
  file(REMOVE ${tmp_script_path})
endfunction()

function (tarantella_gen_gpi_machinefile)
  set(one_value_options NRANKS FILENAME)
  set(required_options NRANKS FILENAME)
  _parse_arguments(ARG "${options}" "${one_value_options}" 
                       "${multi_value_options}" "${required_options}" ${ARGN})

  file(WRITE ${ARG_FILENAME} "")
  cmake_host_system_information(RESULT hostname QUERY HOSTNAME)
  foreach(index RANGE 1 ${ARG_NRANKS})
    file(APPEND ${ARG_FILENAME} "${hostname}\n")
  endforeach()
endfunction()

function (tarantella_gen_test_script)
  set(one_value_options NAME SCRIPT_DIR TEST_FILE)
  set(options IS_PYTHON_TEST)
  set(required_options NAME SCRIPT_DIR TEST_FILE)
  _parse_arguments_with_unknown(ARG "${options}" "${one_value_options}" 
                                    "${multi_value_options}" "${required_options}" ${ARGN})
  
  message(STATUS "Test: Generating ${ARG_NAME} script")
  tarantella_gen_executable_script(SCRIPT_NAME ${ARG_NAME}
                                   SCRIPT_DIR ${ARG_SCRIPT_DIR})

  tarantella_gen_environment_paths(VARIABLE_LIST env_paths)

  set(script_path ${ARG_SCRIPT_DIR}/${ARG_NAME})
  foreach (var ${env_paths})
    file(APPEND ${script_path} "export ${var}\n")
  endforeach()
  if (ARG_IS_PYTHON_TEST)
    # Python test
    file(APPEND ${script_path} "export PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_SOURCE_DIR}/src:\$\{PYTHONPATH\}\n")
    file(APPEND ${script_path} "\n${Python_EXECUTABLE} -m pytest ${ARG_TEST_FILE}\n")
  else()
    # regular executable test
    file(APPEND ${script_path} "\n${ARG_TEST_FILE}\n")
  endif()
endfunction()

function (tarantella_add_gpi_test)
  set(one_value_options NAME TARGET_FILE NRANKS RUNCOMMAND TEST_FILE
                        MACHINEFILE CLEANUP TIMEOUT SLEEP)
  set(multi_value_options LABELS)
  set(required_options NAME TARGET_FILE NRANKS RUNCOMMAND)
  _parse_arguments_with_unknown(ARG "${options}" "${one_value_options}"
                                    "${multi_value_options}" "${required_options}" ${ARGN})
  _default_if_unset(ARG_SLEEP 0)  
  set(test_name ${ARG_NAME}_${ARG_NRANKS}ranks)

  # increase overall timeout time to include the sleep time after the actual test
  if (ARG_TIMEOUT)
    math(EXPR ARG_TIMEOUT "${ARG_SLEEP} + ${ARG_TIMEOUT}")
  endif()

  if (ARG_MACHINEFILE)
    # use user-defined machinefile
    set(runparams "-n ${ARG_NRANKS} -m ${ARG_MACHINEFILE}")
  else()
    # generate machinefile for ARG_NRANKS running on the localhost
    set(machinefile_path ${CMAKE_CURRENT_BINARY_DIR}/machinefile_${ARG_NAME}_${ARG_NRANKS}.tmp)
    tarantella_gen_gpi_machinefile(NRANKS ${ARG_NRANKS} 
                                   FILENAME ${machinefile_path})
    set(runparams "-n ${ARG_NRANKS} -m ${machinefile_path}")
  endif()

  # create gaspi_run test
  add_test(NAME ${test_name}
          WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
          COMMAND "${CMAKE_COMMAND}"
            -DRUNCOMMAND=${ARG_RUNCOMMAND}
            -DRUNCOMMAND_ARGS="${runparams}"
            -DTEST_EXECUTABLE="${ARG_TARGET_FILE}"
            -DTEST_DIR="${CMAKE_BINARY_DIR}"
            -DSLEEP="${ARG_SLEEP}"
            -P "${CMAKE_SOURCE_DIR}/cmake/run_test.cmake"
          ) 

  # set labels if specified
  if (ARG_LABELS)
    set_property(TEST ${test_name} PROPERTY LABELS ${ARG_LABELS})
  endif()

  # set cleanup fixture script if specified
  if (ARG_CLEANUP)
    set_tests_properties(${test_name} PROPERTIES FIXTURES_REQUIRED ${ARG_CLEANUP})
  endif()
  
  # set timeout if specified
  if (ARG_TIMEOUT)
    set_tests_properties(${test_name} PROPERTIES TIMEOUT ${ARG_TIMEOUT})
  endif()

  # make sure the GPI tests are not run in parallel 
  set_tests_properties(${test_name} PROPERTIES RESOURCE_LOCK GPI_run_serial)
endfunction()
