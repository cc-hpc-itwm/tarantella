
set(TFOPS_SOURCES
    ${SRC_DIR}/gpi_comm_lib/tf_ops/AllreduceOps.cpp
    ${SRC_DIR}/gpi_comm_lib/tf_ops/SendRecvOps.cpp
)

set(TFOPS_BUILD_DIR ${CMAKE_BINARY_DIR}/tnt_tfops)
set(TFOPS_LOADER_DIR
    ${SRC_DIR}/gpi_comm_lib/tf_ops/tnt_tfops)

add_custom_target(tfops-loader ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${TFOPS_LOADER_DIR} ${TFOPS_BUILD_DIR})

extended_add_library(NAME tfops
            NAMESPACE tnt
            TYPE SHARED
            SOURCES
                ${TFOPS_SOURCES}
            LIBRARIES
                tnt::gpicommlib
                Tensorflow::Tensorflow
            INSTALL
            INSTALL_DESTINATION
                ${INSTALL_LIB_DIR}
            POSITION_INDEPENDENT)
set_property(TARGET tnt-tfops PROPERTY CXX_STANDARD 14)
