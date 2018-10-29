include(cuda)
include(compiler_settings)
include(cmake/googletest.cmake)
include(cmake/boost.cmake)
include(cmake/psrdada.cmake)
include_directories(SYSTEM ${Boost_INCLUDE_DIR} ${PSRDADA_INCLUDE_DIR})
set(DEPENDENCY_LIBRARIES
    ${GTEST_LIBRARIES}
    ${Boost_LIBRARIES}
    ${PSRDADA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)
