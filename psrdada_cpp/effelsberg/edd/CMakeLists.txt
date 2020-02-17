# OpenMP (optional for shared memory multiprocessing)


if(ENABLE_CUDA)

set(PSRDADA_CPP_EFFELSBERG_EDD_LIBRARIES
    ${CMAKE_PROJECT_NAME}
    ${CMAKE_PROJECT_NAME}_effelsberg_edd
    ${DEPENDENCY_LIBRARIES})

set(psrdada_cpp_effelsberg_edd_src
    src/Channeliser.cu
    src/DadaBufferLayout.cpp
    src/DetectorAccumulator.cu
    src/EDDPolnMerge.cpp
    src/EDDRoach.cpp
    src/EDDRoach_merge.cpp
    src/ScaledTransposeTFtoTFT.cu
    src/Tools.cu
    src/Unpacker.cu
    src/VLBI.cu
    )

cuda_add_library(${CMAKE_PROJECT_NAME}_effelsberg_edd ${psrdada_cpp_effelsberg_edd_src})

#simple FFT spectrometer interface
cuda_add_executable(fft_spectrometer src/fft_spectrometer_cli.cu)
target_link_libraries(fft_spectrometer ${PSRDADA_CPP_EFFELSBERG_EDD_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
install(TARGETS fft_spectrometer DESTINATION bin)

#Gated FFT spectrometer interface
cuda_add_executable(gated_spectrometer src/GatedSpectrometer_cli.cu)
target_link_libraries(gated_spectrometer ${PSRDADA_CPP_EFFELSBERG_EDD_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
install(TARGETS gated_spectrometer DESTINATION bin)

cuda_add_executable(VLBI src/VLBI_cli.cu)
target_link_libraries(VLBI ${PSRDADA_CPP_EFFELSBERG_EDD_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
install(TARGETS VLBI DESTINATION bin)

cuda_add_executable(vdif_send src/vdif_send.cu)
target_link_libraries(vdif_send ${PSRDADA_CPP_EFFELSBERG_EDD_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
install(TARGETS vdif_send DESTINATION bin)

cuda_add_executable(dada_dummy_data src/dummy_data_generator.cu)
target_link_libraries(dada_dummy_data ${PSRDADA_CPP_EFFELSBERG_EDD_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
install(TARGETS VLBI DESTINATION bin)

add_executable(edd_merge src/EDDPolnMerge_cli.cpp)
target_link_libraries(edd_merge ${PSRDADA_CPP_EFFELSBERG_EDD_LIBRARIES})
install(TARGETS edd_merge DESTINATION bin)

add_executable(edd_roach src/EDDRoach_cli.cpp)
target_link_libraries(edd_roach ${PSRDADA_CPP_EFFELSBERG_EDD_LIBRARIES})
install(TARGETS edd_roach DESTINATION bin)

add_executable(edd_roach_merge src/EDDRoach_merge_cli.cpp)
target_link_libraries(edd_roach_merge ${PSRDADA_CPP_EFFELSBERG_EDD_LIBRARIES})
install(TARGETS edd_roach_merge DESTINATION bin)

add_subdirectory(test)
endif(ENABLE_CUDA)
