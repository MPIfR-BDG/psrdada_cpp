include_directories(${GTEST_INCLUDE_DIR})
link_directories(${GTEST_LIBRARY_DIR})

set(
    gtest_tuse_src
    src/TestFileWriterTest.cpp
    src/TransposeTest.cpp
    src/gtest_tuse.cpp
)

add_executable(gtest_tuse ${gtest_tuse_src} )
target_link_libraries(gtest_tuse ${PSRDADA_CPP_MEERKAT_TUSE_LIBRARIES})
#add_test(gtest_tuse gtest_tuse --test_data "${CMAKE_CURRENT_LIST_DIR}/data")
