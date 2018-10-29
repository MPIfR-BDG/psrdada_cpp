include_directories(${GTEST_INCLUDE_DIR})

link_directories(${GTEST_LIBRARY_DIR})

set(
    gtest_fbfuse_src
    src/DelayManagerTester.cpp
    src/gtest_fbfuse.cpp
)
add_executable(gtest_fbfuse ${gtest_fbfuse_src} )
target_link_libraries(gtest_fbfuse ${PSRDADA_CPP_MEERKAT_FBFUSE_LIBRARIES} ${GTEST_LIBRARIES})
add_test(gtest_fbfuse gtest_fbfuse --test_data "${CMAKE_CURRENT_LIST_DIR}/data")
