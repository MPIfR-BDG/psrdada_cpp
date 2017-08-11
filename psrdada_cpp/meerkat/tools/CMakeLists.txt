
if(ENABLE_CUDA)

set(PSRDADA_CPP_MEERKAT_TOOLS_LIBRARIES
    ${CMAKE_PROJECT_NAME}
    ${CMAKE_PROJECT_NAME}_meerkat_tools
    ${DEPENDENCY_LIBRARIES})

set(psrdada_cpp_meerkat_tools_src
    src/feng_to_bandpass.cu
    src/feng_to_dada.cu
    )

cuda_add_library(${CMAKE_PROJECT_NAME}_meerkat_tools ${psrdada_cpp_meerkat_tools_src})

#feng2bp
cuda_add_executable(feng2bp src/feng_to_bandpass_cli.cu)
target_link_libraries(feng2bp ${PSRDADA_CPP_MEERKAT_TOOLS_LIBRARIES})
install(TARGETS feng2bp DESTINATION bin)

#feng2bp_test
cuda_add_executable(feng2bp_test src/feng_to_bandpass_test.cu)
target_link_libraries(feng2bp_test ${PSRDADA_CPP_MEERKAT_TOOLS_LIBRARIES})

#feng2dada
cuda_add_executable(feng2dada src/feng_to_dada_cli.cu)
target_link_libraries(feng2dada ${PSRDADA_CPP_MEERKAT_TOOLS_LIBRARIES})

#feng2dada_test
cuda_add_executable(feng2dada_test src/feng_to_dada_test.cu)
target_link_libraries(feng2dada_test ${PSRDADA_CPP_MEERKAT_TOOLS_LIBRARIES})

endif(ENABLE_CUDA)