#ifndef PSRDADA_CPP_EFFELSBERG_RFI_CHAMBER_RSSPECTROMETER_CUH
#define PSRDADA_CPP_EFFELSBERG_RFI_CHAMBER_RSSPECTROMETER_CUH

#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/double_device_buffer.cuh"
#include "psrdada_cpp/common.hpp"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "cufft.h"

namespace psrdada_cpp {
namespace effelsberg {
namespace rfi_chamber {

class RSSpectrometer
{
public:
    typedef short2 InputType;
    typedef float2 FftType;
    typedef float OutputType;

public:
    RSSpectrometer(std::size_t input_nchans, std::size_t fft_length,
        std::size_t naccumulate, std::size_t nskip);
    RSSpectrometer(RSSpectrometer const&) = delete;
    ~RSSpectrometer();
    void init(RawBytes &header);
    bool operator()(RawBytes &block);

private:
    void process(std::size_t chan_block_idx);
    void copy(RawBytes& block, std::size_t spec_idx, std::size_t chan_block_idx, std::size_t nspectra_in);

private:
    DoubleDeviceBuffer<InputType> _copy_buffer;
    thrust::device_vector<FftType> _fft_buffer;
    thrust::device_vector<OutputType> _accumulation_buffer;
    thrust::host_vector<OutputType> _h_accumulation_buffer;
    std::size_t _input_nchans;
    std::size_t _fft_length;
    std::size_t _naccumulate;
    std::size_t _nskip;
    std::size_t _output_nchans;
    std::size_t _bytes_per_input_spectrum;
    std::size_t _chans_per_copy;
    cufftHandle _fft_plan;
    cudaStream_t _copy_stream;
    cudaStream_t _proc_stream;
};


} //namespace rfi_chamber
} //namespace effelsberg
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_EFFELSBERG_RFI_CHAMBER_RSSPECTROMETER_CUH
