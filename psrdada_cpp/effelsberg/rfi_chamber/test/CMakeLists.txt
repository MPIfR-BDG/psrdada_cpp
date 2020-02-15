include_directories(${GTEST_INCLUDE_DIR})

link_directories(${GTEST_LIBRARY_DIR})

set(
    gtest_rfi_chamber_src
    src/RSSpectrometerTester.cu
    src/gtest_rfi_chamber.cu
)
cuda_add_executable(gtest_rfi_chamber ${gtest_rfi_chamber_src} )
target_link_libraries(gtest_rfi_chamber ${PSRDADA_CPP_EFFELSBERG_RFI_CHAMBER_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
add_test(gtest_rfi_chamber gtest_rfi_chamber --test_data "${CMAKE_CURRENT_LIST_DIR}/data")
