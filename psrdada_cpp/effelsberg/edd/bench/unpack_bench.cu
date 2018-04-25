#include "thrust/device_vector.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>

#define NTHREADS 512

#define CUDA_ERROR_CHECK(ans) { cuda_assert_success((ans), __FILE__, __LINE__); }

/**
 * @brief Function that raises an error on receipt of any cudaError_t
 *  value that is not cudaSuccess
 */
//inline void cuda_assert_success(cudaError_t code, const char *file, int line)
inline void cuda_assert_success(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        /* Ewan note 28/07/2015:
         * This stringstream needs to be made safe.
         * Error message formatting needs to be defined.
         */
        std::stringstream error_msg;
        error_msg << "CUDA failed with error: "
              << cudaGetErrorString(code) << std::endl
              << "File: " << file << std::endl
              << "Line: " << line << std::endl;
        throw std::runtime_error(error_msg.str());
    }
}

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
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x ; (3*idx+2) < n ; idx+=gridDim.x*blockDim.x)
    {
        uint64_t val;
        uint64_t rest;
        int read_idx = 3*idx;
        int write_idx = 16*idx;
        float* sout = out+write_idx;
        val  = swap64(in[read_idx]);
        sout[0] = (float)((int64_t)(( 0xFFF0000000000000 & val) <<  0) >> 52);
        sout[1] = (float)((int64_t)(( 0x000FFF0000000000 & val) << 12) >> 52);
        sout[2] = (float)((int64_t)(( 0x000000FFF0000000 & val) << 24) >> 52);
        sout[3] = (float)((int64_t)(( 0x000000000FFF0000 & val) << 36) >> 52);
        sout[4] = (float)((int64_t)(( 0x000000000000FFF0 & val) << 48) >> 52);
        rest    =                   ( 0x000000000000000F & val) << 60;
        val  = swap64(in[read_idx+1]);
        sout[5] = (float)((int64_t)((( 0xFF00000000000000 & val) >> 4) | rest) >> 52);
        sout[6] = (float)((int64_t)((  0x00FFF00000000000 & val) << 8)  >> 52);
        sout[7] = (float)((int64_t)((  0x00000FFF00000000 & val) << 20) >> 52);
        sout[8] = (float)((int64_t)((  0x00000000FFF00000 & val) << 32) >> 52);
        sout[9] = (float)((int64_t)((  0x00000000000FFF00 & val) << 44) >> 52);
        rest    =                   (  0x00000000000000FF & val) << 56;
        val  = swap64(in[read_idx+2]);
        sout[10] = (float)((int64_t)((( 0xF000000000000000 & val) >>  8) | rest) >> 52);
        sout[11] = (float)((int64_t)((  0x0FFF000000000000 & val) <<  4) >> 52);
        sout[12] = (float)((int64_t)((  0x0000FFF000000000 & val) << 16) >> 52);
        sout[13] = (float)((int64_t)((  0x0000000FFF000000 & val) << 28) >> 52);
        sout[14] = (float)((int64_t)((  0x0000000000FFF000 & val) << 40) >> 52);
    }
}

__global__
void unpack_edd_12bit_to_float32_shared(uint64_t* __restrict__ in, float* __restrict__ out, int n)
{

    __shared__ volatile float tmp_out[NTHREADS * 16];
    __shared__ volatile uint64_t tmp_in[NTHREADS * 3];
    int block_idx = blockIdx.x;

    uint64_t val;
    uint64_t rest;
    volatile float* sout = tmp_out + (16 * threadIdx.x);

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x ; (3*idx+2) < n ; idx+=gridDim.x*blockDim.x)
    {

        //Read to shared memeory
        int block_read_start = block_idx * NTHREADS * 3;
        tmp_in[threadIdx.x]                = in[block_read_start + threadIdx.x];
        tmp_in[NTHREADS + threadIdx.x]     = in[block_read_start + NTHREADS + threadIdx.x];
        tmp_in[NTHREADS * 2 + threadIdx.x] = in[block_read_start + NTHREADS * 2 + threadIdx.x];

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

        int block_write_start = block_idx * NTHREADS * 16;

        for (int ii = threadIdx.x; ii < 16 * NTHREADS; ii+=blockDim.x)
        {
            out[block_write_start+ii] = tmp_out[ii];
        }
        block_idx += gridDim.x;
    }
}


int main()
{
    // equivalent of 1 second of data at 4 4GHz
    int size = 1<<26;
    int n64bit_words = 3*size/16;
    thrust::host_vector<uint64_t> input_host(n64bit_words,1);
    thrust::device_vector<uint64_t> input = input_host;
    thrust::device_vector<float> output(size);
    thrust::device_vector<float> output_shared(size);

    // 64 block
    for (int ii=0; ii<1; ++ii)
    {
        unpack_edd_12bit_to_float32_shared<<<n64bit_words/NTHREADS, NTHREADS>>>(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output_shared.data()), n64bit_words);
    }
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // 1024 block
    for (int ii=0; ii<1; ++ii)
    {
        unpack_edd_12bit_to_float32<<<1024, 1024>>>(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()), n64bit_words);
    }
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());


    thrust::host_vector<float> host_output_shared = output_shared;
    thrust::host_vector<float> host_output = output;

    for (int ii=0; ii<size; ++ii)
    {
        if (host_output_shared[ii] != host_output[ii])
        {
            printf("Fail at index %d with %.4f != %.4f\n", ii, host_output_shared[ii], host_output[ii]);
        }
    }

}