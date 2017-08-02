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
        int antenna_idx = gridIdx.x;
        int channel_idx = gridIdx.y;
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

            for (int ii=0; ii<9; ++ii)
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
        if (threadIdx.x == 0)
        {
            out[antenna_idx * nchans + channel_idx] = total_sum;
        }
    }
}


    FengToBandpass::FengToBandpass(key_t key, MultiLog& log,
        std::size_t nchans, std::size_t nants)
    : DadaIoLoopReader<FengToBandpass>(key,log)
    , _nchans(nchans)
    , _natnennas(nants)
    , _dump_counter(0)
    {
        //Will output data as an array of bandpasses for each
        //polarisation and antenna
        _output.resize(_nchans * _natnennas);
        _h_output.resize(_output.size());
    }

    FengToBandpass::~FengToBandpass()
    {
    }

    void FengToBandpass::on_connect(RawBytes& /*block*/)
    {
        //null, we currently do not read the headers
    }

    void FengToBandpass::on_next(RawBytes& block)
    {
        std::size_t used = block.used_bytes();
        std::size_t nbytes_per_timestamp =
            _natnennas * _nchans * MEERKAT_FENG_NSAMPS_PER_HEAP
            * MEERKAT_FENG_NPOL_PER_HEAP * sizeof(char2);
        if (used%nbytes_per_timestamp != 0)
        {
            throw std::runtime_error("Number of bytes in buffer is not an integer "
                "muliple of the number of bytes per timestamp");
        }
        std::size_t size = used/sizeof(char2);
        int ntimestamps = used/nbytes_per_timestamp;
        _input.resize(size);
        char2* d_input_ptr = thrust::raw_pointer_cast(_input.data());
        float* d_output_ptr = thrust::raw_pointer_cast(_output.data());
        CUDA_ERROR_CHECK(cudaMemcpy(d_input_ptr, block.ptr(), used, cudaMemcpyHostToDevice));
        dim3 grid(_natnennas,_nchans,1);
        kernels::feng_heaps_to_bandpass<<<grid,MEERKAT_FENG_NPOL_PER_HEAP*MEERKAT_FENG_NSAMPS_PER_HEAP>>>
            (d_input_ptr, d_output_ptr, _nchans, _natnennas, ntimestamps);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        thrust::copy(_output.begin(),_output.end(),_h_output.begin());
        write_output_file();
        ++_dump_counter;
    }

    void FengToBandpass::write_output_file()
    {
        std::size_t out_bytes = _h_output.size() * sizeof(decltype(_h_output)::value_type);
        std::stringstream filename;
        filename << "bp_"
        << std::setfill('0')
        << std::setw(9)
        << _dump_counter
        << ".dat";
        std::ofstream outfile;
        outfile.open(filename.str().c_str(),std::ifstream::out | std::ifstream::binary);
        outfile.write((char*) thrust::raw_pointer_cast(_h_output.data()) ,out_bytes);
        outfile.close();
    }
}
}
}