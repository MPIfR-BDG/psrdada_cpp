set(PSRDADA_CPP_MEERKAT_FBFUSE_LIBRARIES
    ${CMAKE_PROJECT_NAME}_meerkat_fbfuse
    ${CMAKE_PROJECT_NAME}
    ${DEPENDENCY_LIBRARIES})

if(ENABLE_CUDA)

   set(psrdada_cpp_meerkat_fbfuse_src
      src/PipelineConfig.cpp
      src/DelayEngineSimulator.cu
      src/DelayManager.cu
      src/IncoherentBeamformer.cu
      src/CoherentBeamformer.cu
      src/WeightsManager.cu
      src/SplitTranspose.cu
      src/Pipeline.cu
      )

   cuda_add_library(${CMAKE_PROJECT_NAME}_meerkat_fbfuse ${psrdada_cpp_meerkat_fbfuse_src})

   cuda_add_executable(fbfuse src/fbfuse_cli.cu)
   target_link_libraries(fbfuse ${PSRDADA_CPP_MEERKAT_FBFUSE_LIBRARIES})
   install(TARGETS fbfuse DESTINATION bin)

else()
   set(psrdada_cpp_meerkat_fbfuse_src
       src/PipelineConfig.cpp
      )

   add_library(${CMAKE_PROJECT_NAME}_meerkat_fbfuse ${psrdada_cpp_meerkat_fbfuse_src})
endif(ENABLE_CUDA)

add_executable(beam_bandpass src/beam_bandpass_cli.cpp)
target_link_libraries(beam_bandpass ${PSRDADA_CPP_MEERKAT_FBFUSE_LIBRARIES})
install(TARGETS beam_bandpass DESTINATION bin)

add_executable(beam_bandpass_shm src/beam_bandpass_shm_cli.cpp)
target_link_libraries(beam_bandpass_shm ${PSRDADA_CPP_MEERKAT_FBFUSE_LIBRARIES} rt)
install(TARGETS beam_bandpass_shm DESTINATION bin)

add_executable(bufferdump_to_null src/bufferdump_to_null_cli.cpp)
target_link_libraries(bufferdump_to_null ${PSRDADA_CPP_MEERKAT_FBFUSE_LIBRARIES})

add_subdirectory(test)


