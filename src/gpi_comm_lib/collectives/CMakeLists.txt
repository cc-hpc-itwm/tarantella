
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

set(COLLECTIVES_SRC_DIR ${SRC_DIR}/gpi_comm_lib/collectives)
set(libSources
    ${COLLECTIVES_SRC_DIR}/lib/allreduceButterfly.cpp
    ${COLLECTIVES_SRC_DIR}/lib/allreduceButterflyDoubleBuffer.cpp
    ${COLLECTIVES_SRC_DIR}/lib/broadcast.cpp
    ${COLLECTIVES_SRC_DIR}/lib/counter.cpp
    ${COLLECTIVES_SRC_DIR}/lib/mailBoxGaspi.cpp
    ${COLLECTIVES_SRC_DIR}/lib/mailBoxLocal.cpp
    ${COLLECTIVES_SRC_DIR}/lib/queues.cpp
    ${COLLECTIVES_SRC_DIR}/lib/reduce.cpp
    ${COLLECTIVES_SRC_DIR}/lib/writer.cpp
    ${COLLECTIVES_SRC_DIR}/allreduce/RecursiveHalving.cpp
    ${COLLECTIVES_SRC_DIR}/allreduce/RecursiveHalvingDoubleBuffer.cpp
    ${COLLECTIVES_SRC_DIR}/allreduce/utils.cpp
    ${COLLECTIVES_SRC_DIR}/barrier/GPIBarrier.cpp
    ${COLLECTIVES_SRC_DIR}/BufferElementType.cpp
    ${COLLECTIVES_SRC_DIR}/FusedTensorInfo.cpp
    ${COLLECTIVES_SRC_DIR}/TensorInfo.cpp
)

extended_add_library(NAME collectives
            NAMESPACE tnt
            TYPE SHARED
            SOURCES
                ${libSources}
            LIBRARIES
                optimized GPI2::GPI2
                debug GPI2::GPI2dbg
                tnt::gpiresources
            INCLUDE_DIRECTORIES
                ${COLLECTIVES_SRC_DIR}/lib/
            COMPILE_OPTIONS
                -Wno-unused-private-field
            INSTALL
            INSTALL_DESTINATION
                ${INSTALL_LIB_DIR}
            POSITION_INDEPENDENT)

