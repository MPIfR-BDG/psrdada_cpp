if(ENABLE_CUDA)

set(PSRDADA_CPP_MEERKAT_FBFUSE_LIBRARIES
    ${CMAKE_PROJECT_NAME}
    ${CMAKE_PROJECT_NAME}_meerkat_fbfuse
    ${DEPENDENCY_LIBRARIES})

set(psrdada_cpp_meerkat_fbfuse_src
    src/PipelineConfig.cpp
    src/Header.cpp
    src/DelayManager.cu	
    src/WeightsManager.cu	
    )

cuda_add_library(${CMAKE_PROJECT_NAME}_meerkat_fbfuse ${psrdada_cpp_meerkat_fbfuse_src})

endif(ENABLE_CUDA)
