#ifndef PSRDADA_CPP_EFFELSBERG_EDD_SIMPLE_FFT_SPECTROMETER_HPP
#define PSRDADA_CPP_EFFELSBERG_EDD_SIMPLE_FFT_SPECTROMETER_HPP

#include "psrdada_cpp/effelsberg/edd/"
#include "psrdada_cpp/raw_bytes.hpp"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "cufft.h"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace kernels {

    __global__
    void unpack_edd_12bit_to_float32(char* __restrict__ in, float* __restrict__ out, int n);

} //kernels

template <class HandlerType>
class SimpleFFTSpectrometer
{
public:
    SimpleFFTSpectrometer(
        std::size_t fft_length,
        std::size_t naccumulate,
        std::size_t nbits,
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
    std::size_t _fft_length;
    std::size_t _naccumulate;
    std::size_t _nbits;
    HandlerType& _handler;
    bool _first_block;
    std::size_t _nsamps;
    cufftHandle _fft_plan;
    thrust::device_vector<uint64_t> _edd_raw;
    thrust::device_vector<float> _edd_unpacked;
    thrust::device_vector<cufftComplex> _channelised;
    thrust::device_vector<float> _detected;
};


} //edd
} //effelsberg
} //psrdada_cpp

#include "psrdada_cpp/effelsberg/edd/detail/edd_simple_fft_spectrometer.cu"
#endif //PSRDADA_CPP_EFFELSBERG_EDD_SIMPLE_FFT_SPECTROMETER_HPP