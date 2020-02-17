set(PSRDADA_CPP_EFFELSBERG_RFI_CHAMBER_LIBRARIES
    ${CMAKE_PROJECT_NAME}_effelsberg_rfi_chamber
    ${CMAKE_PROJECT_NAME}
    ${DEPENDENCY_LIBRARIES})

if(ENABLE_CUDA)

   set(psrdada_cpp_effelsberg_rfi_chamber_src
      src/RSSpectrometer.cu
      )

   cuda_add_library(${CMAKE_PROJECT_NAME}_effelsberg_rfi_chamber ${psrdada_cpp_effelsberg_rfi_chamber_src})
   cuda_add_executable(rsspectrometer src/rfi_chamber_cli.cu)
   target_link_libraries(rsspectrometer ${PSRDADA_CPP_EFFELSBERG_RFI_CHAMBER_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
   install(TARGETS rsspectrometer DESTINATION bin)

else()
   set(psrdada_cpp_effelsberg_rfi_chamber_src
      )

   add_library(${CMAKE_PROJECT_NAME}_effelsberg_rfi_chamber ${psrdada_cpp_effelsberg_rfi_chamber_src})
endif(ENABLE_CUDA)

add_subdirectory(test)


