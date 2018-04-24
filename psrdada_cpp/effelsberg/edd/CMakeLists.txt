if(ENABLE_CUDA)

set(PSRDADA_CPP_EFFELSBERG_EDD_LIBRARIES
    ${CMAKE_PROJECT_NAME}
    ${CMAKE_PROJECT_NAME}_effelsberg_edd
    ${DEPENDENCY_LIBRARIES})

set(psrdada_cpp_effelsberg_edd_src
    src/edd_simple_fft_spectrometer.cu
    )

cuda_add_library(${CMAKE_PROJECT_NAME}_effelsberg_edd ${psrdada_cpp_effelsberg_edd_src})

#simple FFT spectrometer test
cuda_add_executable(edd_specpol src/edd_simple_fft_spectrometer_test.cu)
target_link_libraries(edd_specpol ${PSRDADA_CPP_EFFELSBERG_EDD_LIBRARIES})
install(TARGETS edd_specpol DESTINATION bin)


endif(ENABLE_CUDA)
