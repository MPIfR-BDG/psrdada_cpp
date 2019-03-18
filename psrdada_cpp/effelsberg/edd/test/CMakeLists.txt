include_directories(${GTEST_INCLUDE_DIR})

link_directories(${GTEST_LIBRARY_DIR})

set(
    gtest_edd_src
    src/ChanneliserTester.cu
    src/DetectorAccumulatorTester.cu
    src/FftSpectrometerTester.cu
    src/UnpackerTester.cu
    src/ScaledTransposeTFtoTFTTester.cu
    src/GatedSpectrometerTest.cu
)
cuda_add_executable(gtest_edd ${gtest_edd_src} )
target_link_libraries(gtest_edd ${PSRDADA_CPP_EFFELSBERG_EDD_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} -lcublas)
add_test(gtest_edd gtest_edd --test_data "${CMAKE_CURRENT_LIST_DIR}/data")
