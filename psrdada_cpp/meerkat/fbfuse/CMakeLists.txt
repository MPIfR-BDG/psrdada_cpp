if(ENABLE_CUDA)

set(PSRDADA_CPP_MEERKAT_FBFUSE_LIBRARIES
    ${CMAKE_PROJECT_NAME}
    ${CMAKE_PROJECT_NAME}_meerkat_fbfuse
    ${DEPENDENCY_LIBRARIES})

set(psrdada_cpp_meerkat_fbfuse_src
    src/PipelineConfig.cpp
    src/Header.cpp
    src/DelayManager.cu
    src/CoherentBeamformer.cu
    src/WeightsManager.cu
    src/SplitTranspose.cu
    src/Pipeline.cu
    )

cuda_add_library(${CMAKE_PROJECT_NAME}_meerkat_fbfuse ${psrdada_cpp_meerkat_fbfuse_src})

cuda_add_executable(fbfuse src/fbfuse_cli.cu)
target_link_libraries(fbfuse ${PSRDADA_CPP_MEERKAT_FBFUSE_LIBRARIES})
install(TARGETS fbfuse DESTINATION bin)

add_subdirectory(test)

endif(ENABLE_CUDA)
