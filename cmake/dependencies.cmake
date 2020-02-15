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

if (ENABLE_OPENMP)
  include(FindOpenMP)
  if(OPENMP_FOUND)
     set(DEPENDENCY_LIBRARIES ${DEPENDENCY_LIBRARIES} -fopenmp)
  endif(OPENMP_FOUND)
endif(ENABLE_OPENMP)

