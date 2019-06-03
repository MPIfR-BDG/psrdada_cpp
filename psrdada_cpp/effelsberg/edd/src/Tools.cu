#include "psrdada_cpp/effelsberg/edd/Tools.cuh"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

  __global__ void array_sum(float *in, size_t N, float *out)
{
  __shared__ float data[array_sum_Nthreads];

  size_t tid = threadIdx.x;

  float ls = 0;

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; (i < N);
       i += blockDim.x * gridDim.x) {
    ls += in[i]; // + in[i + blockDim.x];   // loading two elements increase the used bandwidth by ~10% but requires matching blocksize and size of input array
  }

  data[tid] = ls;
  __syncthreads();

  for (size_t i = blockDim.x / 2; i > 0; i /= 2) {
    if (tid < i) {
      data[tid] += data[tid + i];
    }
    __syncthreads();
  }

  // unroll last warp
  // if (tid < 32)
  //{
  //  warpReduce(data, tid);
  //}

  if (tid == 0) {
    out[blockIdx.x] = data[0];
  }
}



__global__ void scaled_square_offset_sum(float *in, size_t N, float* offset, float *out) {
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


} // edd
} // effelsberg
} // psrdada_cpp
