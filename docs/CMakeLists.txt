set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/source)
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/)

if (Sphinx_FOUND)
  add_custom_target(docs ALL
                    COMMAND
                    Sphinx::Sphinx -b html
                    -Drelease=${PROJECT_VERSION}
                    ${SPHINX_SOURCE} ${SPHINX_BUILD}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    COMMENT "Generating documentation with Sphinx")

  install(DIRECTORY ${SPHINX_BUILD}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/docs)
else()
  message(STATUS "Sphinx not found. Skipping documentation build.")
endif()