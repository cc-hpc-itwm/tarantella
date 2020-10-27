include (add_test_wrappers)

set (include_dirs ${CMAKE_SOURCE_DIR}/src/gpi_comm_lib
                  ${CMAKE_SOURCE_DIR}/src/gpi_comm_lib/collectives
                  ${CMAKE_SOURCE_DIR}/src/gpi_comm_lib/collectives/lib)

set(CLEANUP_SCRIPT ${CMAKE_SOURCE_DIR}/cmake/cleanup.sh)
set(CLEANUP_TEST_NAME gpi_cleanup)
add_test (NAME ${CLEANUP_TEST_NAME} COMMAND sh ${CLEANUP_SCRIPT})
set_tests_properties(${CLEANUP_TEST_NAME} PROPERTIES FIXTURES_CLEANUP ${CLEANUP_TEST_NAME})

add_subdirectory(${CMAKE_SOURCE_DIR}/test/collectives)
add_subdirectory(gpi)
add_subdirectory(python)

set(localranks_list 1 2 4 5 7)
tarantella_compile_and_generate_gpi_test(NAME SynchCommunicator
                LOCALRANKS_LIST "${localranks_list}"
                TIMEOUT 20
                SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/SynchCommunicator.cpp
                LIBRARIES tnt::gpicommlib
                INCLUDE_DIRECTORIES ${include_dirs})

tarantella_compile_and_generate_gpi_test(NAME ResourceManager
                LOCALRANKS_LIST "${localranks_list}"
                SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/ResourceManager.cpp
                LIBRARIES tnt::gpicommlib
                INCLUDE_DIRECTORIES ${include_dirs})

tarantella_compile_and_generate_test(NAME TensorFusor
                TIMEOUT 20
                SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/TensorFusor.cpp
                LIBRARIES tnt::gpicommlib
                INCLUDE_DIRECTORIES ${include_dirs})
