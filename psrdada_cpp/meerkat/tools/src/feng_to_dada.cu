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

    __global__ void feng_heaps_to_dada(
        int* __restrict__ in, int* __restrict__ out,
        int nchans)
    {
        //This kernel only works with 256 threads
        //Nchans must be a multiple of 32

        __shared__ char2 transpose_buffer[8][32][32+1];
        int const warp_idx = threadIdx.x >> 0x5;
        int const lane_idx = threadIdx.x & 0x1f;
        int const nelems_per_timestamp = nchans * MEERKAT_FENG_NSAMPS_PER_HEAP;

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
}


    FengToDada::FengToDada(key_t key, MultiLog& log,
        std::size_t nchans)
    : DadaIoLoopReader<FengToDada>(key,log)
    , _nchans(nchans)
    , _dump_counter(0)
    {
    }

    FengToDada::~FengToDada()
    {
    }

    void FengToDada::on_connect(RawBytes& /*block*/)
    {
        //open new file
        //copy header
        //seek to 4096th byte
    }

    void FengToDada::on_next(RawBytes& block)
    {
        //resize output buffers

        std::size_t used = block.used_bytes();
        std::size_t nbytes_per_timestamp =
             * _nchans * MEERKAT_FENG_NSAMPS_PER_HEAP
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
        kernels::feng_heaps_to_bandpass<<<ntimestamps,MEERKAT_FENG_NSAMPS_PER_HEAP>>>
            (d_input_ptr, d_output_ptr, _nchans);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        thrust::copy(_output.begin(),_output.end(),_h_output.begin());
        write_output_file();
        ++_dump_counter;
    }

    void FengToDada::write_output_file()
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