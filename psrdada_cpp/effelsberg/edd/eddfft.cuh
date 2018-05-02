#ifndef PSRDADA_CPP_EFFELSBERG_EDD_EDDFFT_HPP
#define PSRDADA_CPP_EFFELSBERG_EDD_EDDFFT_HPP

#include "psrdada_cpp/raw_bytes.hpp"
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
    void detect_and_accumulate(cufftComplex* __restrict__ in, float* __restrict__ out, int nchans, int nsamps, int naccumulate);


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
        thrust::device_vector<float>* detected);

private:
    int _nsamps;
    int _fft_length;
    int _naccumulate;
    int _nbits;
    HandlerType& _handler;
    cufftHandle _fft_plan;
    int _nchans;
    bool _first;
    bool _second;
    bool _third;

    thrust::device_vector<float> _edd_unpacked;
    thrust::device_vector<cufftComplex> _channelised;

    thrust::device_vector<uint64_t> _edd_raw_a;
    thrust::device_vector<uint64_t> _edd_raw_b;
    thrust::device_vector<uint64_t>* _edd_raw_current;
    thrust::device_vector<uint64_t>* _edd_raw_previous;

    thrust::device_vector<float> _detected_a;
    thrust::device_vector<float> _detected_b;
    thrust::device_vector<float>* _detected_current;
    thrust::device_vector<float>* _detected_previous;

    thrust::host_vector<float, thrust::system::cuda::experimental::pinned_allocator<float> > _detected_host_a;
    thrust::host_vector<float, thrust::system::cuda::experimental::pinned_allocator<float> > _detected_host_b;
    thrust::host_vector<float, thrust::system::cuda::experimental::pinned_allocator<float> >* _detected_host_current;
    thrust::host_vector<float, thrust::system::cuda::experimental::pinned_allocator<float> >* _detected_host_previous;


    cudaStream_t _h2d_stream;
    cudaStream_t _proc_stream;
    cudaStream_t _d2h_stream;
};


} //edd
} //effelsberg
} //psrdada_cpp

#include "psrdada_cpp/effelsberg/edd/detail/eddfft.cu"
#endif //PSRDADA_CPP_EFFELSBERG_EDD_EDDFFT_HPP