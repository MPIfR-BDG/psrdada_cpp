#ifndef PSRDADA_CPP_EFFELSBERG_EDD_EDDFFT_HPP
#define PSRDADA_CPP_EFFELSBERG_EDD_EDDFFT_HPP

#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/double_buffer.hpp"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/system/cuda/experimental/pinned_allocator.h"
#include "cufft.h"

#define NTHREADS_UNPACK 512

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace kernels {

__global__
void unpack_edd_12bit_to_float32(uint64_t* __restrict__ in, float* __restrict__ out, int n);

__global__
void detect_and_accumulate(cufftComplex* __restrict__ in, char* __restrict__ out,
    int nchans, int nsamps, int naccumulate, float scale, float offset);


} //kernels

template <class HandlerType>
class SimpleFFTSpectrometer
{
public:
    SimpleFFTSpectrometer(
        int nsamps_per_block,
        int fft_length,
        int naccumulate,
        int nbits,
        HandlerType& handler);
    ~SimpleFFTSpectrometer();

    /**
     * @brief      A callback to be called on connection
     *             to a ring buffer.
     *
     * @detail     The first available header block in the
     *             in the ring buffer is provided as an argument.
     *             It is here that header parameters could be read
     *             if desired.
     *
     * @param      block  A RawBytes object wrapping a DADA header buffer
     */
    void init(RawBytes& block);

    /**
     * @brief      A callback to be called on acqusition of a new
     *             data block.
     *
     * @param      block  A RawBytes object wrapping a DADA data buffer
     */
    bool operator()(RawBytes& block);

private:
    void process(thrust::device_vector<uint64_t>* digitiser_raw,
        thrust::device_vector<char>* detected);

private:
    int _nsamps;
    int _fft_length;
    int _naccumulate;
    int _nbits;
    HandlerType& _handler;
    cufftHandle _fft_plan;
    int _nchans;
    int _pass;

    thrust::device_vector<float> _edd_unpacked;
    thrust::device_vector<cufftComplex> _channelised;
    DoubleBuffer<thrust::device_vector<uint64_t>> _edd_raw;
    DoubleBuffer<thrust::device_vector<char>> _detected;
    DoubleBuffer<thrust::host_vector<char, thrust::system::cuda::experimental::pinned_allocator<char>>> _detected_host;
    cudaStream_t _h2d_stream;
    cudaStream_t _proc_stream;
    cudaStream_t _d2h_stream;
};


} //edd
} //effelsberg
} //psrdada_cpp

#include "psrdada_cpp/effelsberg/edd/detail/eddfft.cu"
#endif //PSRDADA_CPP_EFFELSBERG_EDD_EDDFFT_HPP