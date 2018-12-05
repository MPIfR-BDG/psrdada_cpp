#ifndef PSRDADA_CPP_DOUBLE_DEVICE_BUFFER_HPP
#define PSRDADA_CPP_DOUBLE_DEVICE_BUFFER_HPP

#include "psrdada_cpp/double_buffer.cuh"
#include "thrust/device_vector.h"

namespace psrdada_cpp {

template <typename T>
using DoubleDeviceBuffer = DoubleBuffer<thrust::device_vector<T>>;

} //namespace psrdada_cpp

#endif //PSRDADA_CPP_DOUBLE_DEVICE_BUFFER_HPP
