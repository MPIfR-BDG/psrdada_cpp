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

} //kernels

namespace test
{
    void unpack_edd_12bit_to_float32_cpu(thrust::host_vector<uint64_t>const& input, thrust::host_vector<float>& output)
    {
        uint64_t val;
        uint64_t rest;
        output.reserve(16 * input.size() / 3);
        for (int ii=0; ii<input.size(); ii+=3)
        {
            int idx = ii;
            val  = be64toh(input[idx]);
            output.push_back( (float)((int64_t)(( 0xFFF0000000000000 & val) <<  0) >> 52));
            output.push_back( (float)((int64_t)(( 0x000FFF0000000000 & val) << 12) >> 52));
            output.push_back( (float)((int64_t)(( 0x000000FFF0000000 & val) << 24) >> 52));
            output.push_back( (float)((int64_t)(( 0x000000000FFF0000 & val) << 36) >> 52));
            output.push_back( (float)((int64_t)(( 0x000000000000FFF0 & val) << 48) >> 52));
            rest    =                   ( 0x000000000000000F & val) << 60;

            val  = be64toh(input[++idx]);
            output.push_back( (float)((int64_t)((( 0xFF00000000000000 & val) >> 4) | rest) >> 52));
            output.push_back( (float)((int64_t)((  0x00FFF00000000000 & val) << 8)  >> 52));
            output.push_back( (float)((int64_t)((  0x00000FFF00000000 & val) << 20) >> 52));
            output.push_back( (float)((int64_t)((  0x00000000FFF00000 & val) << 32) >> 52));
            output.push_back( (float)((int64_t)((  0x00000000000FFF00 & val) << 44) >> 52));
            rest    =                   (  0x00000000000000FF & val) << 56;

            val  = be64toh(input[++idx]);
            output.push_back( (float)((int64_t)((( 0xF000000000000000 & val) >>  8) | rest) >> 52));
            output.push_back( (float)((int64_t)((  0x0FFF000000000000 & val) <<  4) >> 52));
            output.push_back( (float)((int64_t)((  0x0000FFF000000000 & val) << 16) >> 52));
            output.push_back( (float)((int64_t)((  0x0000000FFF000000 & val) << 28) >> 52));
            output.push_back( (float)((int64_t)((  0x0000000000FFF000 & val) << 40) >> 52));
            output.push_back( (float)((int64_t)((  0x0000000000000FFF & val) << 52) >> 52));
        }
    }

    /*
    void unpack_edd_12bit_to_float32_test()
    {

        int nvalues = 1600;
        int nlongs = 3 * nvalues / 16;
        int nthreads = nlongs/3;
        int nblocks = 1;
        int smem_size = nthreads * 16 * sizeof(float);

        thrust::host_vector<uint64_t> input;

        for (int val=0; val<nlongs; ++val)
        {
            input.push_back(val);
        }

        // Run CUDA version
        thrust::device_vector<uint64_t> input_d = input;
        thrust::device_vector<float> output_d(nvalues);
        uint64_t* input_ptr = thrust::raw_pointer_cast(input_d.data());
        float* output_ptr = thrust::raw_pointer_cast(output_d.data())
        unpack_edd_12bit_to_float32<<<nblocks, nthreads, smem_size, 0>>>(input_ptr, output_ptr, nlongs);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        thrust::host_vector<float> output = output_d;

        // Run host version
        thrust::host_vector<float> test_output;
        unpack_edd_12bit_to_float32_cpu(input, test_output);

        for (int ii=0; ii<nvalues; ++ii)
        {
            if (output[ii] != test_output[ii])
            {
                printf("Error at index %d -> %f != %f \n", ii, output[ii], test_output[ii]);
            }
        }
    }
    */
} //test

} //edd
} //effelsberg
} //psrdada_cpp