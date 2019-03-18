#include "psrdada_cpp/effelsberg/edd/VLBI.cuh"
#include "psrdada_cpp/cuda_utils.hpp"

#define EDD_NTHREADS_PACK 1024 
#define NPACK 16

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace kernels {



__global__
void pack_edd_float32_to_2bit(float const* __restrict__ in, uint32_t* __restrict__ out, size_t n, float minV, float maxV)
{

    __shared__ uint32_t tmp_in[EDD_NTHREADS_PACK];
    //__shared__ uint32_t tmp_in[EDD_NTHREADS_PACK];
    //__shared__ volatile uint8_t tmp_out[EDD_NTHREADS_PACK / 4];

    const float s = (maxV - minV) / 3.;
    int odx = blockIdx.x * blockDim.x / NPACK + threadIdx.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n ; idx += gridDim.x * blockDim.x)
    {
        const float delta = (in[idx] - minV);
        tmp_in[threadIdx.x] = 0;
        tmp_in[threadIdx.x] += (delta >= 1 * s);
        tmp_in[threadIdx.x] += (delta >= 2 * s);
        tmp_in[threadIdx.x] += (delta >= 3 * s);
        __syncthreads();

        // can be improved by distributing work on more threads in tree
        // structure, but already at 60-70% memory utilization  
        if (threadIdx.x < EDD_NTHREADS_PACK / NPACK)
        {
          for (size_t i = 1; i < NPACK; i++)
          {
            tmp_in[threadIdx.x] |= tmp_in[threadIdx.x * NPACK + i] << i*2;
          }
          //out[odx] = tmp_out;
          out[odx] = tmp_in[threadIdx.x];
        }
        odx += gridDim.x * blockDim.x / NPACK;

        __syncthreads();
    }
}

} //namespace kernels


void pack_2bit(thrust::device_vector<float> const& input, thrust::device_vector<uint32_t>& output, float minV, float maxV, cudaStream_t _stream)
{
    BOOST_LOG_TRIVIAL(debug) << "Packing 2-bit data";
    assert(input.size() % NPACK == 0);
    output.resize(input.size() / NPACK);
    BOOST_LOG_TRIVIAL(debug) << "INput size: " << input.size() << " elements";
    BOOST_LOG_TRIVIAL(debug) << "Resizing output buffer to " << output.size() << " elements";

    size_t nblocks = std::min(input.size() / EDD_NTHREADS_PACK, 4096uL);
    BOOST_LOG_TRIVIAL(debug) << "  using " << nblocks << " blocks of " << EDD_NTHREADS_PACK << " threads";

    float const* input_ptr = thrust::raw_pointer_cast(input.data());

    uint32_t* output_ptr = thrust::raw_pointer_cast(output.data());

    kernels::pack_edd_float32_to_2bit<<< nblocks, EDD_NTHREADS_PACK, 0, _stream>>>(
            input_ptr, output_ptr, input.size(), minV, maxV);
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_stream));
}

} //namespace edd
} //namespace effelsberg
} //namespace psrdada_cpp
