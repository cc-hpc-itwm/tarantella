include (add_macros)

set(TEMPLATE_CONFIG_FILE "${SRC_DIR}/config.hpp.in")
set(COMPILE_CONFIG_FILE "${CMAKE_BINARY_DIR}/generated/config.hpp")
configure_file(${TEMPLATE_CONFIG_FILE} ${COMPILE_CONFIG_FILE} @ONLY)

set (GPIRESOURCES_SOURCES
    ${SRC_DIR}/gpi_comm_lib/gpi/Context.cpp
    ${SRC_DIR}/gpi_comm_lib/gpi/Group.cpp
    ${SRC_DIR}/gpi_comm_lib/gpi/GroupManager.cpp
    ${SRC_DIR}/gpi_comm_lib/gpi/NotificationManager.cpp
    ${SRC_DIR}/gpi_comm_lib/gpi/QueueManager.cpp
    ${SRC_DIR}/gpi_comm_lib/gpi/ResourceManager.cpp
    ${SRC_DIR}/gpi_comm_lib/gpi/Segment.cpp
    ${SRC_DIR}/gpi_comm_lib/gpi/SegmentBuffer.cpp
    ${SRC_DIR}/gpi_comm_lib/gpi/SegmentManager.cpp
    ${SRC_DIR}/gpi_comm_lib/gpi/gaspiCheckReturn.cpp
)
extended_add_library(NAME gpiresources
            NAMESPACE tnt
            TYPE SHARED
            SOURCES
                ${GPIRESOURCES_SOURCES}
            LIBRARIES
                optimized GPI2::GPI2
                debug GPI2::GPI2dbg
            INCLUDE_DIRECTORIES
                ${SRC_DIR}/gpi_comm_lib/
                ${CMAKE_BINARY_DIR}/generated/
            INSTALL
            INSTALL_DESTINATION
                ${INSTALL_LIB_DIR}
            POSITION_INDEPENDENT)

