#ifndef PSRDADA_CPP_EFFELSBERG_EDD_VLBI_CUH
#define PSRDADA_CPP_EFFELSBERG_EDD_VLBI_CUH

#include "psrdada_cpp/common.hpp"
#include <thrust/device_vector.h>


namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace kernels {



__global__
void pack_edd_float32_to_2bit(float* __restrict__ in, uint32_t * __restrict__ out,  size_t n);

} //namespace kernels

void pack_2bit(thrust::device_vector<float> const& input, thrust::device_vector<uint32_t>& output, float minV, float maxV, cudaStream_t _stream = 0);

} //namespace edd
} //namespace effelsberg
} //namespace psrdada_cpp

#endif // PSRDADA_CPP_EFFELSBERG_EDD_VLBI_CUH



