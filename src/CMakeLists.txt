set(TNT_PYTHON_DIRS
    ${SRC_DIR}/tarantella
    ${SRC_DIR}/runtime
    ${SRC_DIR}/gpi_comm_lib/tf_ops/tnt_tfops)

install(DIRECTORY ${TNT_PYTHON_DIRS}
        DESTINATION ${INSTALL_LIB_DIR}/python
        FILES_MATCHING PATTERN "*.py")

install(PROGRAMS ${SRC_DIR}/bin/tarantella
        DESTINATION ${INSTALL_BIN_DIR})

set(VERSION_FILE_TEMPLATE ${CMAKE_SOURCE_DIR}/cmake/version.py.in)
set(VERSION_FILE ${CMAKE_BUILD_DIR}/version.py)
configure_file(${VERSION_FILE_TEMPLATE} ${VERSION_FILE} @ONLY)
install(FILES ${VERSION_FILE}
        DESTINATION ${INSTALL_LIB_DIR}/python)
