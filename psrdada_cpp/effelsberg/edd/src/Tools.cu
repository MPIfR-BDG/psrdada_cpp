#include "psrdada_cpp/effelsberg/edd/Tools.cuh"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

__global__ void array_sum(float *in, size_t N, float *out) {
  __shared__ float data[array_sum_Nthreads];

  size_t tid = threadIdx.x;

  float ls = 0;

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; (i < N);
       i += blockDim.x * gridDim.x) {
    ls += in[i];
  }

  data[tid] = ls;
  __syncthreads();

  for (size_t i = blockDim.x / 2; i > 0; i /= 2) {
    if (tid < i) {
      data[tid] += data[tid + i];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[blockIdx.x] = data[0];
  }
}


__global__ void scaled_square_residual_sum(float *in, size_t N, float *offset,
                                         float *out) {
  __shared__ float data[array_sum_Nthreads];

  size_t tid = threadIdx.x;

  double ls = 0;

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; (i < N);
       i += blockDim.x * gridDim.x) {
    const double t = (in[i] - (*offset) / N);
    ls += t * t;
  }

  data[tid] = ls / N;
  __syncthreads();

  for (size_t i = blockDim.x / 2; i > 0; i /= 2) {
    if (tid < i) {
      data[tid] += data[tid + i];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[blockIdx.x] = data[0];
  }
}


uint32_t bitMask(uint32_t firstBit, uint32_t lastBit) {
  uint32_t mask = 0U;
  for (uint32_t i = firstBit; i <= lastBit; i++)
    mask |= 1 << i;
  return mask;
}


void setBitsWithValue(uint32_t &target, uint32_t firstBit, uint32_t lastBit,
                      uint32_t value) {
  // check if value is larger than bit range
  if (value > (1u << (lastBit + firstBit))) {
    BOOST_LOG_TRIVIAL(error)
        << "value: " << value << ", 1 << (last-bit - firstbit) "
        << (1 << (lastBit - firstBit)) << ", bitrange: " << lastBit - firstBit
        << std::endl;
    throw std::runtime_error("Value does not fit into bitrange");
  }

  uint32_t mask = bitMask(firstBit, lastBit);

  // zero out relevant bits in data
  target &= ~mask;

  // shift value to corerct position
  value = value << firstBit;

  // update target with value
  target |= value;
}


uint32_t getBitsValue(const uint32_t &target, uint32_t firstBit,
                      uint32_t lastBit) {
  uint32_t mask = bitMask(firstBit, lastBit);
  uint32_t res = target & mask;
  return res >> firstBit;
}


} // edd
} // effelsberg
} // psrdada_cpp
