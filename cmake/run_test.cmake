foreach(var TEST_DIR TEST_SCRIPT RUNCOMMAND RUNCOMMAND_ARGS SLEEP CLEANUP_SCRIPT)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "'${var}' must be defined on the command line")
  endif()

  separate_arguments(var_value UNIX_COMMAND "${${var}}")
  string(LENGTH "${var_value}" var_length)
  if (var_length LESS 1)
    message(FATAL_ERROR "'${var}' must be defined on the command line and not be empty")
  endif()
endforeach()

separate_arguments(runparams_list UNIX_COMMAND "${RUNCOMMAND_ARGS}")
separate_arguments(all_command_params UNIX_COMMAND 
                  "${runparams_list} ${TEST_SCRIPT} ${TEST_ARGS}")

# Execute the test-executable
execute_process(COMMAND ${RUNCOMMAND} ${all_command_params}
                COMMAND_ECHO STDOUT
                RESULT_VARIABLE result)

# Sleep to ensure all processes are done and kill the remainder
separate_arguments(sleep_time UNIX_COMMAND "${SLEEP}")
execute_process(COMMAND ${CMAKE_COMMAND} -E sleep "${sleep_time}"
                COMMAND ${CMAKE_COMMAND} -E echo "Sleep ${sleep_time}")

separate_arguments(all_command_params UNIX_COMMAND 
                "${runparams_list} /bin/bash ${CLEANUP_SCRIPT}")
execute_process(COMMAND ${RUNCOMMAND} ${all_command_params}
                COMMAND_ECHO STDOUT)

# Check return status
if(result)
  message(FATAL_ERROR "Test failed:'${result}'") 
endif()
