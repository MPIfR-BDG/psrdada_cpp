#include "psrdada_cpp/meerkat/tools/feng_to_dada.cuh"
#include "psrdada_cpp/meerkat/constants.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/common.hpp"
#include "cuComplex.h"
#include <fstream>
#include <iomanip>

namespace psrdada_cpp {
namespace meerkat {
namespace tools {
namespace kernels {

    /*
    __global__ void feng_heaps_to_dada(
        int* __restrict__ in, int* __restrict__ out,
        int nchans)
    {
        //This kernel only works with 256 threads
        //Nchans must be a multiple of 32

        __shared__ int transpose_buffer[8][32][32+1];
        int const warp_idx = threadIdx.x >> 0x5;
        int const lane_idx = threadIdx.x & 0x1f;

        //blockIdx.x == timestamp

        int offset = blockIdx.x * nchans * MEERKAT_FENG_NSAMPS_PER_HEAP;

        //Treat the real and imaginary for both polarisations as
        //a single integer. This means we are dealing with TFT data.

        //Each warp does a 32 x 32 element transform

        int chan_idx;
        for (int channel_offset=0; channel_offset<nchans; channel_offset+=32)
        {
            int chan_offset = offset + channel_offset * MEERKAT_FENG_NSAMPS_PER_HEAP;
            for (chan_idx=0; chan_idx<32; ++chan_idx)
            {
                transpose_buffer[warp_idx][chan_idx][lane_idx] = in[chan_offset + MEERKAT_FENG_NSAMPS_PER_HEAP * chan_idx + threadIdx.x];
            }

            int samp_offset = offset + nchans * warp_idx * 32 + channel_offset;
            for (chan_idx=0; chan_idx<32; ++chan_idx)
            {
                out[samp_offset + lane_idx] = transpose_buffer[warp_idx][lane_idx][chan_idx];
            }
        }
    }
    */


    __global__ void feng_heaps_to_dada(
        int* __restrict__ in,
        int* __restrict__ out,
        int nchans)
    {
        __shared__ int transpose_buffer[32][32];
        int const warp_idx = threadIdx.x >> 0x5;
        int const lane_idx = threadIdx.x & 0x1f;
        int offset = blockIdx.x * nchans * MEERKAT_FENG_NSAMPS_PER_HEAP;
        for (int time_idx=0; time_idx < MEERKAT_FENG_NSAMPS_PER_HEAP; time_idx+=warpSize)
        {
            int toffset = offset + (time_idx + lane_idx);
            int coffset = offset + (time_idx + warp_idx) * nchans;
            for (int chan_idx = 0; chan_idx < nchans; chan_idx += warpSize)
            {
                int input_idx = (chan_idx + warp_idx) * MEERKAT_FENG_NSAMPS_PER_HEAP + toffset;
                int output_idx = coffset + (chan_idx + lane_idx);
                transpose_buffer[warp_idx][lane_idx] = in[input_idx];
                __syncthreads();
                out[output_idx] = transpose_buffer[lane_idx][warp_idx];
            }
        }
    }






} //kernels
} //tools
} //meerkat
} //psrdada_cpp