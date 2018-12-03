# == googletest build - must be build with the same compiler flags
if(NOT GTEST_INCLUDE_DIR)
    if(APPLE)
    add_definitions(-DGTEST_USE_OWN_TR1_TUPLE=1)
    else(APPLE)
    add_definitions(-DGTEST_USE_OWN_TR1_TUPLE=0)
    endif(APPLE)

    add_subdirectory("thirdparty/googletest")
    set(GTEST_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/include)
    set(GTEST_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/googletest)
    message("BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}")
    set(GTEST_LIBRARIES gtest_main gtest)
endif()
