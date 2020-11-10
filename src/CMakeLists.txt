set(TNT_PYTHON_DIRS
    ${SRC_DIR}/tarantella
    ${SRC_DIR}/runtime
    ${SRC_DIR}/gpi_comm_lib/tf_ops/tnt_tfops)

install(DIRECTORY ${TNT_PYTHON_DIRS}
        DESTINATION ${INSTALL_LIB_DIR}/python
        FILES_MATCHING PATTERN "*.py")

install(PROGRAMS ${SRC_DIR}/bin/tarantella
        DESTINATION ${INSTALL_BIN_DIR})
