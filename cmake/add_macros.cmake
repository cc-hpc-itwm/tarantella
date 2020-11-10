
macro (_default_if_unset VAR VAL)
  if (NOT ${VAR})
    set (${VAR} ${VAL})
  endif()
endmacro()

include (parse_arguments)

function (extended_add_library)
  set (options POSITION_INDEPENDENT PRECOMPILED INSTALL)
  set (one_value_options NAME NAMESPACE TYPE INSTALL_DESTINATION)
  set (multi_value_options
    LIBRARIES SOURCES PUBLIC_HEADERS INCLUDE_DIRECTORIES RPATH
    SYSTEM_INCLUDE_DIRECTORIES COMPILE_DEFINITIONS COMPILE_OPTIONS DEPENDS
  )
  set (required_options NAME)
  _parse_arguments (ARG "${options}" "${one_value_options}" "${multi_value_options}" "${required_options}" ${ARGN})

  _default_if_unset (ARG_TYPE "STATIC")
  _default_if_unset (ARG_INSTALL_DESTINATION "lib")

  if (ARG_NAMESPACE)
    set (target_name "${ARG_NAMESPACE}-${ARG_NAME}")
  else()
    set (target_name "${ARG_NAME}")
  endif()

  if (NOT (${ARG_TYPE} STREQUAL "STATIC" OR ${ARG_TYPE} STREQUAL "SHARED" OR ${ARG_TYPE} STREQUAL "MODULE"))
    message (FATAL_ERROR "Bad library type: ${ARG_TYPE}")
  endif()

  set (_scope_specifier)
  if ((NOT ARG_SOURCES AND NOT ARG_MOC) OR ARG_PRECOMPILED)
    set (_scope_specifier INTERFACE)

    add_library (${target_name} INTERFACE)

    if (ARG_PRECOMPILED)
      if (ARG_TYPE STREQUAL "STATIC")
        list (APPEND ARG_LIBRARIES "${CMAKE_CURRENT_SOURCE_DIR}/lib${target_name}.a")
      else()
        list (APPEND ARG_LIBRARIES "${CMAKE_CURRENT_SOURCE_DIR}/lib${target_name}.so")
      endif()
    endif()

    target_link_libraries (${target_name} INTERFACE ${ARG_LIBRARIES})
  else()
    set (_scope_specifier PUBLIC)

   # _moc (${ARG_NAME}_mocced ${ARG_MOC})

    add_library (${target_name} ${ARG_TYPE} #${${ARG_NAME}_mocced}
                 ${ARG_SOURCES})

    target_link_libraries (${target_name} ${ARG_LIBRARIES})
  endif()
  if (ARG_NAMESPACE)
    add_library (${ARG_NAMESPACE}::${ARG_NAME} ALIAS ${target_name})
  endif()
  if (ARG_PUBLIC_HEADERS)
    set_property (TARGET ${target_name} APPEND
      PROPERTY PUBLIC_HEADER ${ARG_PUBLIC_HEADERS}
    )
  endif()

  if (ARG_SYSTEM_INCLUDE_DIRECTORIES)
    target_include_directories (${target_name} SYSTEM
      ${ARG_SYSTEM_INCLUDE_DIRECTORIES})
  endif()
  if (ARG_INCLUDE_DIRECTORIES)
    target_include_directories (${target_name} PUBLIC
                                $<BUILD_INTERFACE:${ARG_INCLUDE_DIRECTORIES}>)
  endif()

  if (ARG_POSITION_INDEPENDENT)
    set_property (TARGET ${target_name} APPEND
      PROPERTY COMPILE_FLAGS -fPIC
    )
  endif()

  if (ARG_DEPENDS)
    add_dependencies (${target_name} ${ARG_DEPENDS})
  endif()

  if (ARG_COMPILE_DEFINITIONS)
    target_compile_definitions (${target_name} ${_scope_specifier} ${ARG_COMPILE_DEFINITIONS})
  endif()

  if (ARG_COMPILE_OPTIONS)
    target_compile_options (${target_name} ${_scope_specifier} ${ARG_COMPILE_OPTIONS})
  endif()

  if (ARG_INSTALL)
    install (TARGETS ${target_name}
      LIBRARY DESTINATION "${ARG_INSTALL_DESTINATION}"
      ARCHIVE DESTINATION "${ARG_INSTALL_DESTINATION}"
    )
  endif()
endfunction()

