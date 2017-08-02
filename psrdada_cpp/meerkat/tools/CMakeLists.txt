
if(ENABLE_CUDA)

set(psrdada_cpp_meerkat_tools_src
    src/feng_to_bandpass.cu
    src/feng_to_bandpass_cli.cu
   )

#feng2pb
cuda_add_executable(feng2pb ${psrdada_cpp_meerkat_tools_src})
target_link_libraries(feng2pb ${PSRDADA_CPP_LIBRARIES})
install(TARGETS feng2pb DESTINATION bin)

set(psrdada_cpp_meerkat_tools_test_src
    src/feng_to_bandpass.cu
    src/feng_to_bandpass_test.cu
   )

#feng2pb_test
cuda_add_executable(feng2pb_test ${psrdada_cpp_meerkat_tools_test_src})
target_link_libraries(feng2pb_test ${PSRDADA_CPP_LIBRARIES})

endif(ENABLE_CUDA)