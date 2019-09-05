if(ENABLE_CUDA)
   set(PSRDADA_CPP_MEERKAT_FBFUSE_LIBRARIES
    ${CMAKE_PROJECT_NAME}
    ${CMAKE_PROJECT_NAME}_meerkat_fbfuse
    ${DEPENDENCY_LIBRARIES})


   set(psrdada_cpp_meerkat_fbfuse_src
      src/PipelineConfig.cpp
      src/Header.cpp
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

   cuda_add_executable(beam_bandpass src/beam_bandpass_cli.cpp)
   target_link_libraries(beam_bandpass ${PSRDADA_CPP_MEERKAT_FBFUSE_LIBRARIES})
   install(TARGETS beam_bandpass DESTINATION bin)

   add_subdirectory(test)

   add_executable(beam_bandpass_shm src/beam_bandpass_shm_cli.cpp)
   target_link_libraries(beam_bandpass_shm ${PSRDADA_CPP_MEERKAT_FBFUSE_LIBRARIES})
   install(TARGETS beam_bandpass_shm DESTINATION bin)


else()
   set(PSRDADA_CPP_MEERKAT_FBFUSE_LIBRARIES
    ${CMAKE_PROJECT_NAME}
    ${CMAKE_PROJECT_NAME}_meerkat_fbfuse
    ${DEPENDENCY_LIBRARIES})


   set(psrdada_cpp_meerkat_fbfuse_src
       src/PipelineConfig.cpp
       src/Header.cpp
      )

   add_library(${CMAKE_PROJECT_NAME}_meerkat_fbfuse ${psrdada_cpp_meerkat_fbfuse_src})

   add_executable(beam_bandpass src/beam_bandpass_cli.cpp)
   target_link_libraries(beam_bandpass ${PSRDADA_CPP_MEERKAT_FBFUSE_LIBRARIES})
   install(TARGETS beam_bandpass DESTINATION bin)


   add_executable(beam_bandpass_shm src/beam_bandpass_shm_cli.cpp)
   target_link_libraries(beam_bandpass_shm ${PSRDADA_CPP_MEERKAT_FBFUSE_LIBRARIES} rt)
   install(TARGETS beam_bandpass_shm DESTINATION bin)

endif(ENABLE_CUDA)



