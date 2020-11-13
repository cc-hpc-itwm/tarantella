include(FindPackageHandleStandardArgs)

find_program(Sphinx_EXECUTABLE
             NAMES sphinx-build sphinx-build2
             DOC "Path to sphinx-build executable")

find_package_handle_standard_args(Sphinx REQUIRED_VARS Sphinx_EXECUTABLE)

if (Sphinx_FOUND)
  mark_as_advanced(Sphinx_EXECUTABLE)
endif()

if (Sphinx_FOUND AND NOT TARGET Sphinx::Sphinx)
  add_executable(Sphinx::Sphinx IMPORTED)
  set_property(TARGET Sphinx::Sphinx PROPERTY IMPORTED_LOCATION ${Sphinx_EXECUTABLE})
endif()
