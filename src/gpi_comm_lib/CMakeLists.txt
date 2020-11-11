include (add_macros)

set(GPI_LIB_MODULE "GPICommLib")

set (GPICOMMLIB_SOURCES
    ${SRC_DIR}/gpi_comm_lib/distribution/SegmentIDBuilder.cpp
    ${SRC_DIR}/gpi_comm_lib/distribution/utilities.cpp
    ${SRC_DIR}/gpi_comm_lib/PipelineCommunicator.cpp
    ${SRC_DIR}/gpi_comm_lib/SynchCommunicator.cpp
    ${SRC_DIR}/gpi_comm_lib/TensorBroadcaster.cpp
)

extended_add_library(NAME gpicommlib
            NAMESPACE tnt
            TYPE SHARED
            SOURCES
                ${GPICOMMLIB_SOURCES}
            LIBRARIES
                tnt::gpiresources
                tnt::collectives
            INCLUDE_DIRECTORIES
                ${SRC_DIR}/gpi_comm_lib/
            INSTALL
            INSTALL_DESTINATION
                ${INSTALL_LIB_DIR}
            POSITION_INDEPENDENT)

pybind11_add_module(${GPI_LIB_MODULE} MODULE
                    ${SRC_DIR}/gpi_comm_lib/pybind11_wrappers.cpp)
target_link_libraries(${GPI_LIB_MODULE} PRIVATE pybind11::module
                                                tnt::gpicommlib)
install(TARGETS ${GPI_LIB_MODULE}
        LIBRARY
        DESTINATION ${INSTALL_LIB_DIR})
