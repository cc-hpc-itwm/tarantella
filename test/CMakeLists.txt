include (add_test_wrappers)

set(SLEEP_TIME_AFTER_TEST 2)

set (include_dirs ${CMAKE_SOURCE_DIR}/src/gpi_comm_lib)

set(CLEANUP_SCRIPT ${CMAKE_SOURCE_DIR}/cmake/cleanup.sh)
set(CLEANUP_TEST_NAME gpi_cleanup)
add_test (NAME ${CLEANUP_TEST_NAME} COMMAND sh ${CLEANUP_SCRIPT})
set_tests_properties(${CLEANUP_TEST_NAME} PROPERTIES FIXTURES_CLEANUP ${CLEANUP_TEST_NAME})

add_subdirectory(python)

set(localranks_list 1 2 4 5 8)
tarantella_compile_and_generate_gpi_test(NAME SynchCommunicator
                LOCALRANKS_LIST "${localranks_list}"
                TIMEOUT 200
                SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/SynchCommunicator.cpp
                LIBRARIES tnt::gpicommlib
                INCLUDE_DIRECTORIES ${include_dirs})

tarantella_compile_and_generate_test(NAME TensorFusor
                TIMEOUT 20
                SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/TensorFusor.cpp
                LIBRARIES tnt::gpicommlib
                INCLUDE_DIRECTORIES ${include_dirs})
