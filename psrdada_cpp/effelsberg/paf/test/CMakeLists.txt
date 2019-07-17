include_directories(${GTEST_INCLUDE_DIR})

link_directories(${GTEST_LIBRARY_DIR})

set(
    gtest_paf_src
    src/UnpackerTester.cu
)
cuda_add_executable(gtest_paf ${gtest_paf_src} )
target_link_libraries(gtest_paf ${PSRDADA_CPP_EFFELSBERG_PAF_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
add_test(gtest_paf gtest_paf --test_data "${CMAKE_CURRENT_LIST_DIR}/data")
