#ifndef PSRDADA_CPP_EFFELSBERG_EDD_CHANNELISER_HPP
#define PSRDADA_CPP_EFFELSBERG_EDD_CHANNELISER_HPP

#include "psrdada_cpp/effelsberg/edd/Unpacker.cuh"
#include "psrdada_cpp/effelsberg/edd/ScaledTransposeTFtoTFT.cuh"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/double_device_buffer.cuh"
#include "psrdada_cpp/double_host_buffer.cuh"
#include "psrdada_cpp/dada_write_client.hpp"
#include "thrust/device_vector.h"
#include "cufft.h"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

class Channeliser
{
public:
    typedef uint64_t RawVoltageType;
    typedef float UnpackedVoltageType;
    typedef float2 ChannelisedVoltageType;
    typedef char2 PackedChannelisedVoltageType;

public:
    Channeliser(
        std::size_t buffer_bytes,
        std::size_t fft_length,
        std::size_t nbits,
        float input_level,
        float output_level,
        DadaWriteClient& client);
    ~Channeliser();

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
    void process(thrust::device_vector<RawVoltageType> const& digitiser_raw,
        thrust::device_vector<PackedChannelisedVoltageType>& packed_channelised);

private:
    std::size_t _buffer_bytes;
    std::size_t _fft_length;
    std::size_t _nbits;
    DadaWriteClient& _client;
    cufftHandle _fft_plan;
    int _nchans;
    int _call_count;
    std::unique_ptr<Unpacker> _unpacker;
    std::unique_ptr<ScaledTransposeTFtoTFT> _transposer;
    DoubleDeviceBuffer<RawVoltageType> _raw_voltage_db;
    DoubleDeviceBuffer<PackedChannelisedVoltageType> _packed_channelised_voltage;
    thrust::device_vector<UnpackedVoltageType> _unpacked_voltage;
    thrust::device_vector<ChannelisedVoltageType> _channelised_voltage;
    cudaStream_t _h2d_stream;
    cudaStream_t _proc_stream;
    cudaStream_t _d2h_stream;
};


} //edd
} //effelsberg
} //psrdada_cpp

#endif //PSRDADA_CPP_EFFELSBERG_EDD_CHANNELISER_HPP
