# -------------------------------------------
# Build and integrate a thirdparty dependency 
#
# Copyright Oxford university 2018
# Released under GPL v3
# --------------------------------------------

macro(add_thirdparty_subdirectory dependency_name dependency_dir)
    set(dependency_install_path "thirdparty/${dependency_name}")
    mark_as_advanced(dependency_install_path)

    # save current project state
    list(APPEND LIBRARY_INSTALL_DIR_STACK ${LIBRARY_INSTALL_DIR})
    list(APPEND INCLUDE_INSTALL_DIR_STACK ${INCLUDE_INSTALL_DIR})
    list(APPEND MODULES_INSTALL_DIR_STACK ${MODULES_INSTALL_DIR})
    list(APPEND BINARY_INSTALL_DIR_STACK ${LIBRARY_INSTALL_DIR})
    list(APPEND DOC_INSTALL_DIR_STACK ${DOC_INSTALL_DIR})

    # set dependency install targets to be relative to the parent project
    set(LIBRARY_INSTALL_DIR ${LIBRARY_INSTALL_DIR}/${dependency_install_path})
    set(INCLUDE_INSTALL_DIR ${INCLUDE_INSTALL_DIR}/${dependency_install_path})
    set(MODULES_INSTALL_DIR ${MODULES_INSTALL_DIR}/${dependency_install_path})
    set(BINARY_INSTALL_DIR ${BINARY_INSTALL_DIR}/${dependency_install_path})
    set(DOC_INSTALL_DIR ${DOC_INSTALL_DIR}/${dependency_install_path})

    add_subdirectory(${dependency_dir})
    
    # restore current project state
    list(GET LIBRARY_INSTALL_DIR_STACK -1 LIBRARY_INSTALL_DIR)
    list(REMOVE_AT LIBRARY_INSTALL_DIR_STACK -1)
    list(GET INCLUDE_INSTALL_DIR_STACK -1 INCLUDE_INSTALL_DIR)
    list(REMOVE_AT INCLUDE_INSTALL_DIR_STACK -1)
    list(GET MODULES_INSTALL_DIR_STACK -1 MODULES_INSTALL_DIR)
    list(REMOVE_AT MODULES_INSTALL_DIR_STACK -1)
    list(GET BINARY_INSTALL_DIR_STACK -1 BINARY_INSTALL_DIR)
    list(REMOVE_AT BINARY_INSTALL_DIR_STACK -1)
    list(GET DOC_INSTALL_DIR_STACK -1 DOC_INSTALL_DIR)
    list(REMOVE_AT DOC_INSTALL_DIR_STACK -1)

endmacro (add_thirdparty_directory)
