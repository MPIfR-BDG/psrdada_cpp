if(ENABLE_CUDA)

set(PSRDADA_CPP_EFFELSBERG_EDD_LIBRARIES
    ${CMAKE_PROJECT_NAME}
    ${CMAKE_PROJECT_NAME}_effelsberg_edd
    ${DEPENDENCY_LIBRARIES})

set(psrdada_cpp_effelsberg_edd_src
    src/Unpacker.cu
    src/DetectorAccumulator.cu
    )

cuda_add_library(${CMAKE_PROJECT_NAME}_effelsberg_edd ${psrdada_cpp_effelsberg_edd_src})

#simple FFT spectrometer test
#cuda_add_executable(eddfft_test src/eddfft_test.cu)
#target_link_libraries(eddfft_test ${PSRDADA_CPP_EFFELSBERG_EDD_LIBRARIES} cufft)
#install(TARGETS eddfft_test DESTINATION bin)

#simple FFT spectrometer interface
#cuda_add_executable(eddfft src/eddfft_cli.cu)
#target_link_libraries(eddfft ${PSRDADA_CPP_EFFELSBERG_EDD_LIBRARIES} cufft)
#install(TARGETS eddfft DESTINATION bin)

add_subdirectory(test)
endif(ENABLE_CUDA)
