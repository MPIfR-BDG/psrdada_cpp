#ifndef PSRDADA_CPP_DOUBLE_HOST_BUFFER_HPP
#define PSRDADA_CPP_DOUBLE_HOST_BUFFER_HPP

#include "psrdada_cpp/double_buffer.cuh"
#include "thrust/host_vector.h"
#include "thrust/system/cuda/experimental/pinned_allocator.h"

namespace psrdada_cpp {

template <typename T>
using DoubleHostBuffer = DoubleBuffer<thrust::host_vector<T>>;

template <typename T>
using DoublePinnedHostBuffer = DoubleBuffer<thrust::host_vector<T,
    thrust::system::cuda::experimental::pinned_allocator<char>>>;

} //namespace psrdada_cpp

#endif //PSRDADA_CPP_DOUBLE_HOST_BUFFER_HPP
