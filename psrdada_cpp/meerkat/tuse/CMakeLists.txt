set(PSRDADA_CPP_MEERKAT_TUSE_LIBRARIES
    ${CMAKE_PROJECT_NAME}
    ${CMAKE_PROJECT_NAME}_meerkat_tuse
    ${DEPENDENCY_LIBRARIES})

set(psrdada_cpp_meerkat_tuse_src
    src/transpose_to_dada.cpp
    )

add_library(${CMAKE_PROJECT_NAME}_meerkat_tuse ${psrdada_cpp_meerkat_tuse_src})

#transpose_to_dada_cli
add_executable(transpose_to_dada_cli src/transpose_to_dada_cli.cpp)
target_link_libraries (transpose_to_dada_cli ${PSRDADA_CPP_MEERKAT_TUSE_LIBRARIES})

#transpose_to_files_cli
add_executable(transpose_to_file_cli src/transpose_to_file_cli.cpp)
target_link_libraries (transpose_to_file_cli ${PSRDADA_CPP_MEERKAT_TUSE_LIBRARIES})

#transpose_to_null_cli
add_executable(transpose_to_null_cli src/transpose_to_null_cli.cpp)
target_link_libraries (transpose_to_null_cli ${PSRDADA_CPP_MEERKAT_TUSE_LIBRARIES})

#dada_dbevent
add_executable(dada_dbevent src/dada_dbevent.cpp)
target_link_libraries (dada_dbevent ${PSRDADA_CPP_MEERKAT_TUSE_LIBRARIES})

install (TARGETS transpose_to_null_cli transpose_to_file_cli transpose_to_dada_cli DESTINATION bin)
install(FILES transpose_to_dada.hpp DESTINATION include/psrdada_cpp/meerkat/tuse)

add_subdirectory(test)
