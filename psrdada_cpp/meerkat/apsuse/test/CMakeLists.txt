include_directories(${GTEST_INCLUDE_DIR})

link_directories(${GTEST_LIBRARY_DIR})

set(
    gtest_apsuse_src
    src/BeamCaptureControllerTester.cpp
    src/gtest_apsuse.cpp
)
add_executable(gtest_apsuse ${gtest_apsuse_src} )
target_link_libraries(gtest_apsuse ${PSRDADA_CPP_LIBRARIES})
add_test(gtest_apsuse gtest_apsuse --test_data "${CMAKE_CURRENT_LIST_DIR}/data")
