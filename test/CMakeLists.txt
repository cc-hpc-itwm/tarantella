include (add_test_wrappers)

set(SLEEP_TIME_AFTER_TEST 1)
set (include_dirs ${CMAKE_SOURCE_DIR}/src/gpi_comm_lib)

tarantella_generate_cleanup_test(NRANKS 4)

set(localranks_list 1 2 4 5 8)
tarantella_compile_and_generate_gpi_test(NAME DataParallel_SynchCommunicator
                LOCALRANKS_LIST "${localranks_list}"
                TIMEOUT 300
                SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/SynchCommunicator.cpp
                LIBRARIES tnt::gpicommlib
                INCLUDE_DIRECTORIES ${include_dirs})

tarantella_compile_and_generate_test(NAME DataParallel_TensorFusor
                TIMEOUT 20
                SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/TensorFusor.cpp
                LIBRARIES tnt::gpicommlib
                INCLUDE_DIRECTORIES ${include_dirs})

add_subdirectory(pipelining)
add_subdirectory(python)
