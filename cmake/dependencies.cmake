include(cuda)
include(compiler_settings)
include(cmake/boost.cmake)
include(cmake/psrdada.cmake)
include_directories(SYSTEM ${Boost_INCLUDE_DIR} ${PSRDADA_INCLUDE_DIR})
set(DEPENDENCY_LIBRARIES
    ${Boost_LIBRARIES}
    ${PSRDADA_LIBRARIES}
)