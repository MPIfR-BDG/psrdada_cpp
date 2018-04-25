#include "psrdada_cpp/effelsberg/edd/eddfft.cuh"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/cuda_utils.hpp"

#define BSWAP64(x) ((0xFF00000000000000 & x) >> 56) | \
                   ((0x00FF000000000000 & x) >> 40) | \
                   ((0x0000FF0000000000 & x) >> 24) | \
                   ((0x000000FF00000000 & x) >>  8) | \
                   ((0x00000000FF000000 & x) <<  8) | \
                   ((0x0000000000FF0000 & x) << 24) | \
                   ((0x000000000000FF00 & x) << 40) | \
                   ((0x00000000000000FF & x) << 56)


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

    __shared__ volatile float tmp_out[NTHREADS_UNPACK * 16];
    __shared__ volatile uint64_t tmp_in[NTHREADS_UNPACK * 3];
    int block_idx = blockIdx.x;

    uint64_t val;
    uint64_t rest;
    volatile float* sout = tmp_out + (16 * threadIdx.x);

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x ; (3*idx+2) < n ; idx+=gridDim.x*blockDim.x)
    {

        //Read to shared memeory
        int block_read_start = block_idx * NTHREADS_UNPACK * 3;
        tmp_in[threadIdx.x]                = in[block_read_start + threadIdx.x];
        tmp_in[NTHREADS_UNPACK + threadIdx.x]     = in[block_read_start + NTHREADS_UNPACK + threadIdx.x];
        tmp_in[NTHREADS_UNPACK * 2 + threadIdx.x] = in[block_read_start + NTHREADS_UNPACK * 2 + threadIdx.x];

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

        int block_write_start = block_idx * NTHREADS_UNPACK * 16;

        for (int ii = threadIdx.x; ii < 16 * NTHREADS_UNPACK; ii+=blockDim.x)
        {
            out[block_write_start+ii] = tmp_out[ii];
        }
        block_idx += gridDim.x;
    }
}

__global__
void detect_and_accumulate(cufftComplex* __restrict__ in, float* __restrict__ out, int nchans, int nsamps, int naccumulate)
{
    for (int block_idx = blockIdx.x; block_idx < nsamps/naccumulate; ++block_idx)
    {
        int read_offset = block_idx * naccumulate * nchans;
        int write_offset = block_idx * nchans;
        for (int chan_idx = threadIdx.x; threadIdx.x < nchans; chan_idx += blockDim.x)
        {
            float sum = 0.0f;
            for (int ii=0; ii < naccumulate; ++ii)
            {
                cufftComplex tmp = in[read_offset + chan_idx];
                float x = tmp.x * tmp.x;
                float y = tmp.y * tmp.y;
                sum += x + y;
            }
            out[write_offset + chan_idx] = sum;
        }
    }
}



} //kernels
} //edd
} //effelsberg
} //psrdada_cpp