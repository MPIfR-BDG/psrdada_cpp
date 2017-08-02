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

    /**
     * @brief      Convert data that is in MeerKAT F-engine order into
     *             a spectrum for each antenna
     *
     * @detail     The heaps from the the MeerKAT F-engine are in in FTP
     *             order with the T = 256 and P = 2. The number of frequency
     *             channels in a heap is variable but is always a power of two.
     *             The data itself is 8-bit complex (8-bit real, 8-bit imaginary).
     *             In this kernel we perform a vectorised char2 load such that
     *             each thread gets a complete complex voltage.
     *
     *             As each block with process all heaps from a given antenna
     *             for a given frequency channel we use 512 threads per block
     *             which matches nicely with the inner TP dimensions of the heaps.
     *
     *             The heaps themselves are ordered in TAF (A=antenna) order. As
     *             such the full order of the input can be considered to be
     *             tAFFTP which simplifies to tAFTP (using small t and big T
     *             to disambiguate between the two time axes). Each block of
     *             threads will process TP for all t (for one A and one F).
     *
     * @param      in               Input buffer
     * @param      out              Output buffer
     * @param[in]  nchans           The number of frequency chans
     * @param[in]  nants            The number of antennas
     * @param[in]  ntimestamps      The number of timestamps (this corresponds to the
     *                              number of heaps in the time axis)
     */
    __global__ void feng_heaps_to_bandpass(
        char2* __restrict__ in, float* __restrict__ out,
        int nchans, int nants,
        int ntimestamps)
    {
        __shared__ float time_pol_ar[MEERKAT_FENG_NPOL_PER_HEAP*MEERKAT_FENG_NSAMPS_PER_HEAP];

        float total_sum = 0.0f;
        int antenna_idx = gridDim.x;
        int channel_idx = gridDim.y;
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