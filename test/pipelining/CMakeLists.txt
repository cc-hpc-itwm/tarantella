include (add_test_wrappers)

set (include_dirs ${CMAKE_SOURCE_DIR}/src/gpi_comm_lib
                  ${CMAKE_SOURCE_DIR}/test)

set(localranks_list 2)
tarantella_compile_and_generate_gpi_test(NAME PipelineCommunicator_SendReceive
                LOCALRANKS_LIST "${localranks_list}"
                TIMEOUT 10
                SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/PipelineCommunicator_SendRecv.cpp
                LIBRARIES tnt::gpicommlib
                INCLUDE_DIRECTORIES ${include_dirs})
