if(ENABLE_CUDA)

set(PSRDADA_CPP_EFFELSBERG_PAF_LIBRARIES
    ${CMAKE_PROJECT_NAME}
    ${CMAKE_PROJECT_NAME}_effelsberg_paf
    ${DEPENDENCY_LIBRARIES})

set(psrdada_cpp_effelsberg_paf_src
    src/Unpacker.cu
    )

cuda_add_library(${CMAKE_PROJECT_NAME}_effelsberg_paf ${psrdada_cpp_effelsberg_paf_src})

#simple FFT spectrometer interface
#cuda_add_executable(fft_spectrometer src/fft_spectrometer_cli.cu)
#target_link_libraries(fft_spectrometer ${PSRDADA_CPP_EFFELSBERG_PAF_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
#install(TARGETS fft_spectrometer DESTINATION bin)

add_subdirectory(test)
endif(ENABLE_CUDA)
