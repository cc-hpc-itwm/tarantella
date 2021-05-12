include (add_test_wrappers)

set (include_dirs ${CMAKE_SOURCE_DIR}/src/gpi_comm_lib
                  ${CMAKE_SOURCE_DIR}/test)

tarantella_compile_and_generate_gpi_test(NAME Pipeline_Communicator_SendReceive
                LOCALRANKS_LIST "2"
                TIMEOUT 20
                SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/PipelineCommunicator_SendRecv_2ranks.cpp
                LIBRARIES tnt::gpicommlib
                INCLUDE_DIRECTORIES ${include_dirs})

tarantella_compile_and_generate_gpi_test(NAME Pipeline_Communicator
                LOCALRANKS_LIST "4"
                TIMEOUT 60
                SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/PipelineCommunicator_4ranks.cpp
                LIBRARIES tnt::gpicommlib
                INCLUDE_DIRECTORIES ${include_dirs}
                LABELS cluster)
