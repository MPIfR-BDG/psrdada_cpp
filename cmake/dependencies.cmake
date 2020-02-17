include(cuda)
include(compiler_settings)
include(cmake/googletest.cmake)
include(cmake/boost.cmake)
include(cmake/psrdada.cmake)
include_directories(SYSTEM ${Boost_INCLUDE_DIR} ${PSRDADA_INCLUDE_DIR})
include_directories(BEFORE ${GTEST_INCLUDE_DIR})
set(DEPENDENCY_LIBRARIES
    ${GTEST_LIBRARIES}
    ${Boost_LIBRARIES}
    ${PSRDADA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message(STATUS "Found OpenMP" )
    set(DEPENDENCY_LIBRARIES ${DEPENDENCY_LIBRARIES} ${OpenMP_EXE_LINKER_FLAGS})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
