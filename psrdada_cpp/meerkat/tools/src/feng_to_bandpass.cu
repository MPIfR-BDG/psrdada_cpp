#include "psrdada_cpp/meerkat/tools/feng_to_bandpass.cuh"
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

    __global__ void feng_heaps_to_bandpass(
        char2* __restrict__ in, float* __restrict__ out,
        int nchans, int nants,
        int ntimestamps)
    {
        __shared__ float time_pol_ar[MEERKAT_FENG_NPOL_PER_HEAP*MEERKAT_FENG_NSAMPS_PER_HEAP];

        float total_sum = 0.0f;
        int antenna_idx = blockIdx.x;
        int channel_idx = blockIdx.y;
        for (int heap_idx=0; heap_idx<ntimestamps; ++heap_idx)
        {
            int offset = MEERKAT_FENG_NSAMPS_PER_HEAP * MEERKAT_FENG_NPOL_PER_HEAP * (
                nchans * (heap_idx * nants + antenna_idx)
                + channel_idx);

            char2 tmp = in[offset + threadIdx.x];
            cuComplex voltage = make_cuComplex(tmp.x,tmp.y);
            float val = voltage.x * voltage.x + voltage.y * voltage.y;
            time_pol_ar[threadIdx.x] = val;
            __syncthreads();

            for (int ii=1; ii<9; ++ii)
            {
                if ((threadIdx.x + (1<<ii)) < (MEERKAT_FENG_NSAMPS_PER_HEAP*MEERKAT_FENG_NPOL_PER_HEAP))
                {
                    val += time_pol_ar[threadIdx.x + (1<<ii)];
                }
                __syncthreads();
                time_pol_ar[threadIdx.x] = val;
                __syncthreads();
            }
            total_sum += val;
        }
        if (threadIdx.x == 0 || threadIdx.x == 1)
        {
            out[antenna_idx * nchans * MEERKAT_FENG_NPOL_PER_HEAP
                + antenna_idx * nchans * threadIdx.x
                + channel_idx] = total_sum;
        }
    }
}

}
}
}