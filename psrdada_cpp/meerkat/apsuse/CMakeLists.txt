
   set(PSRDADA_CPP_MEERKAT_APSUSE_LIBRARIES
    ${CMAKE_PROJECT_NAME}
    ${CMAKE_PROJECT_NAME}_meerkat_apsuse
    ${DEPENDENCY_LIBRARIES})


   set(psrdada_cpp_meerkat_apsuse_src
      )

  # add_library(${CMAKE_PROJECT_NAME}_meerkat_apsuse ${psrdada_cpp_meerkat_apsuse_src})

   #add_executable(apsuse src/apsuse_cli.cu)
   #target_link_libraries(apsuse ${PSRDADA_CPP_MEERKAT_APSUSE_LIBRARIES})
   #install(TARGETS apsuse DESTINATION bin)

   add_subdirectory(test)






