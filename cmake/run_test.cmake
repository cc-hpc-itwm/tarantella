# Kill old processes that may be still running
function (kill_old_processes)
  set(one_value_options TEST_DIR TEST_EXECUTABLE)
  cmake_parse_arguments(ARG "${options}" "${one_value_options}" 
                            "${multi_value_options}" ${ARGN})

  set(find_processes_command "ps -ef | grep ${ARG_TEST_DIR} | grep -v grep | grep -v ${ARG_TEST_EXECUTABLE}")
  set(kill_command "${find_processes_command} | awk '{print $2}' | xargs -r kill -9")

  execute_process(COMMAND sh -c  "echo \"Killing `${find_processes_command} | wc -l` processes\"; ${find_processes_command}")
  execute_process(COMMAND sh -c "${kill_command}"
                  COMMAND_ECHO STDOUT)
endfunction()

foreach(var TEST_DIR TEST_EXECUTABLE RUNCOMMAND RUNCOMMAND_ARGS SLEEP)
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
                  "${runparams_list} ${TEST_EXECUTABLE} ${TEST_ARGS}")
kill_old_processes(TEST_DIR ${TEST_DIR}
                   TEST_EXECUTABLE ${TEST_EXECUTABLE})

# Execute the test-executable
execute_process(COMMAND ${RUNCOMMAND} ${all_command_params}
                COMMAND_ECHO STDOUT
                RESULT_VARIABLE result)

# Sleep to ensure all processes are done and kill the remainder
separate_arguments(sleep_time UNIX_COMMAND "${SLEEP}")
execute_process(COMMAND ${CMAKE_COMMAND} -E sleep "${sleep_time}"
                COMMAND ${CMAKE_COMMAND} -E echo "Sleep ${sleep_time}")
kill_old_processes(TEST_DIR ${TEST_DIR}
                   TEST_EXECUTABLE ${TEST_EXECUTABLE})

# Check return status
if(result)
  message(FATAL_ERROR "Test failed:'${result}'") 
endif()


