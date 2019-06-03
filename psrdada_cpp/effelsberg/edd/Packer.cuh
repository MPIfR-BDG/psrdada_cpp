#ifndef PSRDADA_CPP_EFFELSBERG_EDD_PACKER_CUH
#define PSRDADA_CPP_EFFELSBERG_EDD_PACKER_CUH

#include "psrdada_cpp/common.hpp"
#include <thrust/device_vector.h>
#include "psrdada_cpp/cuda_utils.hpp"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace kernels {


// pack float to 2,4,8,16 bit integers with linear scaling
template <unsigned int input_bit_depth>
__global__ void packNbit(const float *__restrict__ input,
                         uint32_t *__restrict__ output, size_t inputSize,
                         float minV, float maxV) {
  // number of values to pack into one output element, use 32 bit here to
  // maximize number of threads
  const uint8_t NPACK = 32 / input_bit_depth;

  const float l = (maxV - minV) / ((1 << input_bit_depth) - 1);
  __shared__ uint32_t tmp[1024];

  for (uint32_t i = NPACK * blockIdx.x * blockDim.x + threadIdx.x;
       (i < inputSize); i += blockDim.x * gridDim.x * NPACK) {
    tmp[threadIdx.x] = 0;

    #pragma unroll
    for (uint8_t j = 0; j < NPACK; j++) {
      // Load new input value, clip and convert to Nbit integer
      const float inp = input[i + j * blockDim.x];

      uint32_t p = 0;
      #pragma unroll
      for (int k = 1; k < (1 << input_bit_depth); k++) {
        p += (inp > ((k * l) + minV));
      } // this is more efficient than fmin, fmax for clamp and cast.

      // store in shared memory with linear access
      tmp[threadIdx.x] += p << (input_bit_depth * j);
    }
    __syncthreads();

    // load value from shared memory and rearange to output - the read value is
    // reused per warp
    uint32_t out = 0;

    // bit mask: Thread 0 always first input_bit_depth bits, thread 1 always
    // second input_bit_depth bits, ...
    const uint32_t mask = ((1 << input_bit_depth) - 1) << (input_bit_depth * (threadIdx.x % NPACK));
    #pragma unroll
    for (uint32_t j = 0; j < NPACK; j++) {
      uint32_t v = tmp[(threadIdx.x / NPACK) * NPACK + j] & mask;
      // retrieve correct bits
      v = v >> (input_bit_depth * (threadIdx.x % NPACK));
      v = v << (input_bit_depth * j);
      out += v;
    }

    size_t oidx = threadIdx.x / NPACK + (threadIdx.x % NPACK) * (blockDim.x / NPACK) + (i - threadIdx.x) / NPACK;
    output[oidx] = out;
    __syncthreads();
  }
}
} // namespace kernels


template <unsigned int input_bit_depth>
void pack(const thrust::device_vector<float> &input, thrust::device_vector<uint32_t> &output, float minV, float maxV, cudaStream_t &stream)
{
  BOOST_LOG_TRIVIAL(debug) << "Packing data with bitdepth " << input_bit_depth << " in range " << minV << " - " << maxV;

  const uint32_t NPACK = 32 / input_bit_depth;
  assert(input.size() % NPACK == 0);
  output.resize(input.size() / NPACK);
  BOOST_LOG_TRIVIAL(debug) << "Input size: " << input.size() << " elements";
  BOOST_LOG_TRIVIAL(debug) << "Resizing output buffer to " << output.size() << " elements";

  kernels::packNbit<input_bit_depth><<<128, 1024, 0,stream>>>(thrust::raw_pointer_cast(input.data()),
    thrust::raw_pointer_cast(output.data()),
    input.size(), minV, maxV);

  CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
};


} //namespace edd
} //namespace effelsberg
} //namespace psrdada_cpp

#endif // PSRDADA_CPP_EFFELSBERG_EDD_UNPACKER_CUH



