include_directories(${GTEST_INCLUDE_DIR})
link_directories(${GTEST_LIBRARY_DIR})

set(
    gtest_psrdada_cpp_src
    src/test_sigproc_file_writer.cpp
    src/gtest_psrdada_cpp.cpp
)

add_executable(gtest_psrdada_cpp ${gtest_psrdada_cpp_src} )
target_link_libraries(gtest_psrdada_cpp ${PSRDADA_CPP_LIBRARIES})
