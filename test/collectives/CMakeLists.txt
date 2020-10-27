
include (add_test_wrappers)

set (include_dirs ${CMAKE_SOURCE_DIR}/test
                  ${CMAKE_SOURCE_DIR}/src/gpi_comm_lib
                  ${CMAKE_SOURCE_DIR}/src/gpi_comm_lib/collectives/lib)

tarantella_compile_and_generate_test(NAME Allreduce_basic
                SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/allreduce_basic.cpp
                LIBRARIES tnt::gpicommlib
                INCLUDE_DIRECTORIES ${include_dirs})

set(localranks_list 8)
tarantella_compile_and_generate_gpi_test(NAME AllreduceButterfly
                LOCALRANKS_LIST "${localranks_list}"
                TIMEOUT 20
                SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/AllreduceButterfly.cpp
                LIBRARIES tnt::gpicommlib
                INCLUDE_DIRECTORIES ${include_dirs})

tarantella_compile_and_generate_gpi_test(NAME AllreduceButterflyDoubleBuffer
                LOCALRANKS_LIST "${localranks_list}"
                TIMEOUT 20
                SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/AllreduceButterflyDoubleBuffer.cpp
                LIBRARIES tnt::gpicommlib
                INCLUDE_DIRECTORIES ${include_dirs})