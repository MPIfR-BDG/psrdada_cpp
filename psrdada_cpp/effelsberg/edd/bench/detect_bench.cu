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

__global__
void detect_and_accumulate(float2* __restrict__ in, float* __restrict__ out, int nchans, int nsamps, int naccumulate)
{
    for (int block_idx = blockIdx.x; block_idx < nsamps/naccumulate; block_idx+=gridDim.x)
    {
        int read_offset = block_idx * naccumulate * nchans;
        int write_offset = block_idx * nchans;
        for (int chan_idx = threadIdx.x; chan_idx < nchans; chan_idx += blockDim.x)
        {
            float sum = 0.0f;
            for (int ii=0; ii < naccumulate; ++ii)
            {
                float2 tmp = in[read_offset + chan_idx + ii*nchans];
                float x = tmp.x * tmp.x;
                float y = tmp.y * tmp.y;
                sum += x + y;
            }
            out[write_offset + chan_idx] = sum;
        }
    }
}

int main()
{
    int nchan = 8192/2 + 1;
    int nsamp = (1<<26) / 8192;
    int naccumulate = 4;

    thrust::host_vector<float2> input_host(nchan * nsamp);

    for (int ii=0 ; ii<input_host.size(); ++ii)
    {
        input_host[ii].x = (float) ii;
        input_host[ii].y = (float) ii;
    }

    thrust::device_vector<float2> input = input_host;
    thrust::device_vector<float> output(nchan * nsamp / naccumulate);

    for (int ii=0; ii< 100; ++ii)
    {
        detect_and_accumulate<<<1024, 1024>>>(
            thrust::raw_pointer_cast(input.data()),
            thrust::raw_pointer_cast(output.data()),
            nchan,
            nsamp,
            naccumulate);
    }

    thrust::host_vector<float> out_host = output;

    for (int samp=0; samp<nsamp/naccumulate; ++samp)
    {
        for (int chan=0; chan<nchan; ++chan)
        {
            float val = 0.0f;
            for (int jj=0; jj<naccumulate; ++jj)
            {
                val += (jj * jj) + (jj * jj);
            }

            if (out_host[samp*nchan + chan] != val)
            {
                printf("%f\n",out_host[samp*nchan + chan]);
            }
        }
    }

    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

}