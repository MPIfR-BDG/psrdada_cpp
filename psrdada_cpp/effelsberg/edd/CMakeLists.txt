if(ENABLE_CUDA)

set(PSRDADA_CPP_EFFELSBERG_EDD_LIBRARIES
    ${CMAKE_PROJECT_NAME}
    ${CMAKE_PROJECT_NAME}_effelsberg_edd
    ${DEPENDENCY_LIBRARIES})

set(psrdada_cpp_effelsberg_edd_src
    src/Unpacker.cu
    src/DetectorAccumulator.cu
    src/ScaledTransposeTFtoTFT.cu
    )

cuda_add_library(${CMAKE_PROJECT_NAME}_effelsberg_edd ${psrdada_cpp_effelsberg_edd_src})

#simple FFT spectrometer interface
cuda_add_executable(fft_spectrometer src/fft_spectrometer_cli.cu)
target_link_libraries(fft_spectrometer ${PSRDADA_CPP_EFFELSBERG_EDD_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
install(TARGETS fft_spectrometer DESTINATION bin)

#Gated FFT spectrometer interface
cuda_add_executable(gated_spectrometer src/GatedSpectrometer_cli.cu)
target_link_libraries(gated_spectrometer ${PSRDADA_CPP_EFFELSBERG_EDD_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} -lcublas)
install(TARGETS gated_spectrometer DESTINATION bin)

add_subdirectory(test)
endif(ENABLE_CUDA)
