# define the cheetah libraries
set(PSRDADA_CPP_LIBRARIES ${CMAKE_PROJECT_NAME} ${DEPENDENCY_LIBRARIES})

include_directories(..)
include_directories(${PROJECT_BINARY_DIR})

set(psrdada_cpp_src
    #src/multilog.cpp
    #src/raw_bytes.cpp
    #src/dada_client_base.cpp
    #src/dada_write_client.cpp
    #src/dada_read_client.cpp
   )

# -- the main library target
#add_library(${CMAKE_PROJECT_NAME} ${psrdada_cpp_src})
add_executable(junkdb examples/junkdb.cpp)
target_link_libraries (junkdb ${PSRDADA_CPP_LIBRARIES})
install (TARGETS junkdb DESTINATION bin)
#install (FILES MathFunctions.h DESTINATION include)