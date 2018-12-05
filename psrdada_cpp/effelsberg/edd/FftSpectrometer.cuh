#ifndef PSRDADA_CPP_EFFELSBERG_EDD_FftSpectrometer_HPP
#define PSRDADA_CPP_EFFELSBERG_EDD_FftSpectrometer_HPP

#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/double_buffer.hpp"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/system/cuda/experimental/pinned_allocator.h"
#include "cufft.h"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

template <class HandlerType>
class FftSpectrometer
{
public:
    typedef uint64_t RawVoltageType;
    typedef float UnpackedVoltageType;
    typedef float2 ChannelisedVoltageType;
    typedef int8_t IntegratedPowerType;

public:
    FftSpectrometer(
        int fft_length,
        int naccumulate,
        int nbits,
        HandlerType& handler);
    ~FftSpectrometer();

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
    void process(thrust::device_vector<uint64_t> const& digitiser_raw,
        thrust::device_vector<int8_t>* detected);

private:
    int _fft_length;
    int _naccumulate;
    int _nbits;
    HandlerType& _handler;
    cufftHandle _fft_plan;
    int _nchans;
    int _pass;

    std::unique_ptr<Unpacker> _unpacker;
    std::unique_ptr<DetectorAccumulator> _detector;
    DoubleDeviceBuffer<RawVoltageType> _raw_voltage_db;
    DoubleDeviceBuffer<IntegratedPowerType> _power_db;
    thrust::device_vector<UnpackedVoltageType> _unpacked_voltage;
    thrust::device_vector<ChannelisedVoltageType> _channelised_voltage;
    DoubleBuffer<thrust::host_vector<char, thrust::system::cuda::experimental::pinned_allocator<char>>> _detected_host;
    cudaStream_t _h2d_stream;
    cudaStream_t _proc_stream;
    cudaStream_t _d2h_stream;
};


} //edd
} //effelsberg
} //psrdada_cpp

#include "psrdada_cpp/effelsberg/edd/detail/eddfft.cu"
#endif //PSRDADA_CPP_EFFELSBERG_EDD_FftSpectrometer_HPP