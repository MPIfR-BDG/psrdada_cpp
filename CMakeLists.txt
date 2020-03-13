# project name and specify the minumum cmake version.
project (psrdada_cpp CXX)
cmake_minimum_required (VERSION 3.0)
enable_testing()

# The version number.
set (psrdada_cpp_VERSION_MAJOR 0)
set (psrdada_cpp_VERSION_MINOR 1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# cmake setup.
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

# default installtion locations (note these will be prefixed by ${CMAKE_INSTALL_PREFIX})
if(NOT LIBRARY_INSTALL_DIR)
    set(LIBRARY_INSTALL_DIR "lib")
endif(NOT LIBRARY_INSTALL_DIR)

if(NOT INCLUDE_INSTALL_DIR)
    set(INCLUDE_INSTALL_DIR "include/${CMAKE_PROJECT_NAME}")
endif(NOT INCLUDE_INSTALL_DIR)

if(NOT MODULES_INSTALL_DIR)
    set (MODULES_INSTALL_DIR "share/${CMAKE_PROJECT_NAME}")
endif(NOT MODULES_INSTALL_DIR)

if(NOT BINARY_INSTALL_DIR)
    set(BINARY_INSTALL_DIR "bin/")
endif(NOT BINARY_INSTALL_DIR)

# get project dependencies and compiler settings.
include(dependencies)

# === process through the cheetah tree:
add_subdirectory(${CMAKE_PROJECT_NAME})

# === Print build options summary.
set(DEBUG_PRINT ON)
if (DEBUG_PRINT)
    message(STATUS "")
    message(STATUS "****************************************************************************")
    message(STATUS " name: ${CMAKE_PROJECT_NAME} version: ${PROJECT_VERSION}")
    message(STATUS "")
    message(STATUS "")
    message(STATUS " Compiler Options:")
    message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
    message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER}")
    if (${CMAKE_BUILD_TYPE} MATCHES release)
        message(STATUS "  C++ flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
    elseif (${CMAKE_BUILD_TYPE} MATCHES debug)
        message(STATUS "  C++ flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
    elseif (${CMAKE_BUILD_TYPE} MATCHES relwithdebinfo)
        message(STATUS "  C++ flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    elseif (${CMAKE_BUILD_TYPE} MATCHES minsizerel)
        message(STATUS "  C++ flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_MINSIZEREL}")
    endif ()
    message(STATUS "")
    message(STATUS "Install locations: (make install)")
    message(STATUS "  Libraries: ${CMAKE_INSTALL_PREFIX}/lib")
    message(STATUS "  Inclues:   ${CMAKE_INSTALL_PREFIX}/${INCLUDE_INSTALL_DIR}")
    message(STATUS "  Binaries:  ${CMAKE_INSTALL_PREFIX}/${BINARY_INSTALL_DIR}")
    if(ENABLE_DOC)
        message(STATUS "  Doc:       ${CMAKE_INSTALL_PREFIX}/${DOC_INSTALL_DIR}")
    else(ENABLE_DOC)
        message(STATUS "  Doc:       Disabled (-DENABLE_DOC=true to enable)")
    endif(ENABLE_DOC)
    message(STATUS "  Other:     ${CMAKE_INSTALL_PREFIX}/${MODULES_INSTALL_DIR}")
    message(STATUS "****************************************************************************")
    message(STATUS "")
endif (DEBUG_PRINT)
