include (add_macros)

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
            INSTALL
            INSTALL_DESTINATION
                ${INSTALL_LIB_DIR}
            POSITION_INDEPENDENT)

