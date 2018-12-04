#include "psrdada_cpp/effelsberg/edd/Unpacker.cuh"
#include "psrdada_cpp/cuda_utils.hpp"

#define EDD_NTHREADS_UNPACK 512

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace kernels {

__device__ __forceinline__ uint64_t swap64(uint64_t x)
{
    uint64_t result;
    uint2 t;
    asm("mov.b64 {%0,%1},%2; \n\t"
        : "=r"(t.x), "=r"(t.y) : "l"(x));
    t.x = __byte_perm(t.x, 0, 0x0123);
    t.y = __byte_perm(t.y, 0, 0x0123);
    asm("mov.b64 %0,{%1,%2}; \n\t"
        : "=l"(result) : "r"(t.y), "r"(t.x));
    return result;
}

__global__
void unpack_edd_12bit_to_float32(uint64_t* __restrict__ in, float* __restrict__ out, int n)
{
    /**
     * Note: This kernels will not work with more than 512 threads.
     */
    __shared__ volatile float tmp_out[EDD_NTHREADS_UNPACK * 16];
    __shared__ volatile uint64_t tmp_in[EDD_NTHREADS_UNPACK * 3];
    int block_idx = blockIdx.x;
    uint64_t val;
    uint64_t rest;
    volatile float* sout = tmp_out + (16 * threadIdx.x);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
        (3 * idx + 2) < n;
        idx+=gridDim.x*blockDim.x)
    {
        //Read to shared memeory
        int block_read_start = block_idx * EDD_NTHREADS_UNPACK * 3;
        tmp_in[threadIdx.x]                = in[block_read_start + threadIdx.x];
        tmp_in[EDD_NTHREADS_UNPACK + threadIdx.x]     = in[block_read_start + EDD_NTHREADS_UNPACK + threadIdx.x];
        tmp_in[EDD_NTHREADS_UNPACK * 2 + threadIdx.x] = in[block_read_start + EDD_NTHREADS_UNPACK * 2 + threadIdx.x];
        __syncthreads();
        val  = swap64(tmp_in[3*threadIdx.x]);
        sout[0] = (float)((int64_t)(( 0xFFF0000000000000 & val) <<  0) >> 52);
        sout[1] = (float)((int64_t)(( 0x000FFF0000000000 & val) << 12) >> 52);
        sout[2] = (float)((int64_t)(( 0x000000FFF0000000 & val) << 24) >> 52);
        sout[3] = (float)((int64_t)(( 0x000000000FFF0000 & val) << 36) >> 52);
        sout[4] = (float)((int64_t)(( 0x000000000000FFF0 & val) << 48) >> 52);
        rest    =                   ( 0x000000000000000F & val) << 60;
        val  = swap64(tmp_in[3*threadIdx.x + 1]);
        sout[5] = (float)((int64_t)((( 0xFF00000000000000 & val) >> 4) | rest) >> 52);
        sout[6] = (float)((int64_t)((  0x00FFF00000000000 & val) << 8)  >> 52);
        sout[7] = (float)((int64_t)((  0x00000FFF00000000 & val) << 20) >> 52);
        sout[8] = (float)((int64_t)((  0x00000000FFF00000 & val) << 32) >> 52);
        sout[9] = (float)((int64_t)((  0x00000000000FFF00 & val) << 44) >> 52);
        rest    =                   (  0x00000000000000FF & val) << 56;
        val  = swap64(tmp_in[3*threadIdx.x + 2]);
        sout[10] = (float)((int64_t)((( 0xF000000000000000 & val) >>  8) | rest) >> 52);
        sout[11] = (float)((int64_t)((  0x0FFF000000000000 & val) <<  4) >> 52);
        sout[12] = (float)((int64_t)((  0x0000FFF000000000 & val) << 16) >> 52);
        sout[13] = (float)((int64_t)((  0x0000000FFF000000 & val) << 28) >> 52);
        sout[14] = (float)((int64_t)((  0x0000000000FFF000 & val) << 40) >> 52);
        __syncthreads();
        int block_write_start = block_idx * EDD_NTHREADS_UNPACK * 16;
        for (int ii = threadIdx.x; ii < 16 * EDD_NTHREADS_UNPACK; ii += blockDim.x)
        {
            out[block_write_start + ii] = tmp_out[ii];
        }
        block_idx += gridDim.x;
    }
}

__global__
void unpack_edd_8bit_to_float32(uint64_t* __restrict__ in, float* __restrict__ out, int n)
{
    /**
     * Note: This kernels will not work with more than 512 threads.
     */
    __shared__ volatile float tmp_out[EDD_NTHREADS_UNPACK * 8];
    int block_idx = blockIdx.x;
    uint64_t val;
    volatile float* sout = tmp_out + (8 * threadIdx.x);

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x ; idx < n ; idx+=gridDim.x*blockDim.x)
    {
        int block_read_start = block_idx * EDD_NTHREADS_UNPACK;
        val = swap64(in[block_read_start + threadIdx.x]);
        sout[0] = (float)((int64_t)(( 0xFF00000000000000 & val) <<  0) >> 56);
        sout[1] = (float)((int64_t)(( 0x00FF000000000000 & val) <<  8) >> 56);
        sout[2] = (float)((int64_t)(( 0x0000FF0000000000 & val) << 16) >> 56);
        sout[3] = (float)((int64_t)(( 0x000000FF00000000 & val) << 24) >> 56);
        sout[4] = (float)((int64_t)(( 0x00000000FF000000 & val) << 32) >> 56);
        sout[5] = (float)((int64_t)(( 0x0000000000FF0000 & val) << 40) >> 56);
        sout[6] = (float)((int64_t)(( 0x000000000000FF00 & val) << 48) >> 56);
        sout[7] = (float)((int64_t)(( 0x00000000000000FF & val) << 56) >> 56);
        __syncthreads();
        int block_write_start = block_idx * EDD_NTHREADS_UNPACK * 8;
        for (int ii = threadIdx.x; ii < 8 * EDD_NTHREADS_UNPACK; ii+=blockDim.x)
        {
            out[block_write_start+ii] = tmp_out[ii];
        }
        block_idx += gridDim.x;
    }
}

} //namespace kernels


Unpacker::Unpacker(cudaStream_t stream)
    : _stream(stream)
{

}

Unpacker::~Unpacker()
{

}

template <>
void Unpacker::unpack<12>(InputType const& input, OutputType& output)
{
    BOOST_LOG_TRIVIAL(debug) << "Unpacking 12-bit data";
    std::size_t output_size = input.size() * 16 / 3;
    BOOST_LOG_TRIVIAL(debug) << "Resizing output buffer to " << output_size << " elements";
    output.resize(output_size);
    int nblocks = input.size() / EDD_NTHREADS_UNPACK;
    InputType::value_type const* input_ptr = thrust::raw_pointer_cast(input.data())
    OutputType::value_type* output_ptr = thrust::raw_pointer_cast(output.data())
    kernels::unpack_edd_12bit_to_float32<<< nblocks, EDD_NTHREADS_UNPACK, 0, _stream>>>(
            input_ptr, output_ptr, input.size());
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_stream));
}

template <>
void Unpacker::unpack<8>(InputType const& input, OutputType& output)
{
    BOOST_LOG_TRIVIAL(debug) << "Unpacking 12-bit data";
    std::size_t output_size = input.size() * 8;
    BOOST_LOG_TRIVIAL(debug) << "Resizing output buffer to " << output_size << " elements";
    output.resize(output_size);
    int nblocks = input.size() / EDD_NTHREADS_UNPACK;
    InputType::value_type const* input_ptr = thrust::raw_pointer_cast(input.data())
    OutputType::value_type* output_ptr = thrust::raw_pointer_cast(output.data())
    kernels::unpack_edd_8bit_to_float32<<< nblocks, EDD_NTHREADS_UNPACK, 0, _stream>>>(
            input_ptr, output_ptr, input.size());
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_stream));
}

} //namespace edd
} //namespace effelsberg
} //namespace psrdada_cpp