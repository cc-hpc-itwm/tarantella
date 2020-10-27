
include (add_test_wrappers)
include (parse_arguments)

set (include_dirs ${CMAKE_SOURCE_DIR}/src/gpi_comm_lib
                  ${CMAKE_SOURCE_DIR}/test)

set(localranks_list 1 2 4 5 7)
tarantella_compile_and_generate_gpi_test(NAME Context
                LOCALRANKS_LIST "${localranks_list}"
                SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Context.cpp                       
                LIBRARIES tnt::gpicommlib
                INCLUDE_DIRECTORIES ${include_dirs})

tarantella_compile_and_generate_gpi_test(NAME Group
                LOCALRANKS_LIST "${localranks_list}"
                SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Group.cpp                       
                LIBRARIES tnt::gpicommlib
                INCLUDE_DIRECTORIES ${include_dirs})

tarantella_compile_and_generate_gpi_test(NAME GroupManager
                LOCALRANKS_LIST "${localranks_list}"
                SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/GroupManager.cpp
                LIBRARIES tnt::gpicommlib
                INCLUDE_DIRECTORIES ${include_dirs})

tarantella_compile_and_generate_gpi_test(NAME QueueManager
                LOCALRANKS_LIST "${localranks_list}"
                SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/QueueManager.cpp
                LIBRARIES tnt::gpicommlib
                INCLUDE_DIRECTORIES ${include_dirs})

tarantella_compile_and_generate_gpi_test(NAME NotificationManager
                LOCALRANKS_LIST "${localranks_list}"
                SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/NotificationManager.cpp
                LIBRARIES tnt::gpicommlib
                INCLUDE_DIRECTORIES ${include_dirs})

tarantella_compile_and_generate_gpi_test(NAME SegmentManager
                LOCALRANKS_LIST "${localranks_list}"
                SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/SegmentManager.cpp
                LIBRARIES tnt::gpicommlib
                INCLUDE_DIRECTORIES ${include_dirs})
                
