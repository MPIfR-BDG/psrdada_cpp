#ifndef PSRDADA_CPP_EFFELSBERG_RFI_CHAMBER_RSSPECTROMETER_CUH
#define PSRDADA_CPP_EFFELSBERG_RFI_CHAMBER_RSSPECTROMETER_CUH

#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/double_device_buffer.cuh"
#include "psrdada_cpp/common.hpp"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "cufft.h"
#include <string>

namespace psrdada_cpp {
namespace effelsberg {
namespace rfi_chamber {

/**
 * @brief      Pipeline for processing single polarisation channelised
 *             data in TF order.
 *
 * @detail     Pipeline has been developed to handle the output of an FPGA
 *             attached to a Rohde & Schwarz spectrum analyser running in
 *             IQ sampling mode.
 *
 *             Data passed to the operator() method of the class is first converted
 *             from network-order shorts to host-order single precision floats. The
 *             floating point data is then FFT'd along the T axis of the data before
 *             the power is detected and the data is integrated into an accumulation
 *             buffer. After a certain number of spectra have been accumulated the
 *             integrated spectrum will be written to disk and the object will be
 *             destroyed.
 */
class RSSpectrometer
{
public:
    typedef short2 InputType;
    typedef float2 FftType;
    typedef float OutputType;

public:
    /**
     * @brief      Constructs a new instance.
     *
     * @param[in]  input_nchans  The number of input nchans
     * @param[in]  fft_length    The length of the FFT to apply to each channel
     * @param[in]  naccumulate   The number of detected spectra to accumulate
     * @param[in]  nskip         The number of DADA blocks to skip before accumulating
     *                           (to allow network settle time)
     * @param[in]  filename      The name of the output file to write to
     */
    RSSpectrometer(
        std::size_t input_nchans, std::size_t fft_length,
        std::size_t naccumulate, std::size_t nskip,
        std::string filename, float reference_dbm);
    RSSpectrometer(RSSpectrometer const&) = delete;
    ~RSSpectrometer();

    /**
     * @brief      Handle the DADA header.
     *
     * @param      header  The header in DADA format
     *
     * @detail     Currently a NO-OP, as no information is required from the header.
     */
    void init(RawBytes &header);

    /**
     * @brief      Invoke the pipeline for a block of valid TF data
     *
     * @param      block  A RawBytes block containing network order shorts in TF[IQ] order
     *
     * @return     Flag indicating if the complete number of spectra has been accumulated already
     */
    bool operator()(RawBytes &block);

private:
    void process(std::size_t chan_block_idx);
    void copy(RawBytes& block, std::size_t spec_idx, std::size_t chan_block_idx, std::size_t nspectra_in);
    void write_spectrum();
    void write_histogram(thrust::device_vector<int> const& histogram);

private:
    DoubleDeviceBuffer<InputType> _copy_buffer;
    thrust::device_vector<FftType> _fft_input_buffer;
    thrust::device_vector<FftType> _fft_output_buffer;
    thrust::device_vector<OutputType> _accumulation_buffer;
    thrust::host_vector<OutputType> _h_accumulation_buffer;
    std::size_t _input_nchans;
    std::size_t _fft_length;
    std::size_t _naccumulate;
    std::size_t _nskip;
    std::string _filename;
    float _reference_dbm;
    std::size_t _output_nchans;
    std::size_t _bytes_per_input_spectrum;
    std::size_t _naccumulated;
    std::size_t _chans_per_copy;
    cufftHandle _fft_plan;
    cudaStream_t _copy_stream;
    cudaStream_t _proc_stream;
};


} //namespace rfi_chamber
} //namespace effelsberg
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_EFFELSBERG_RFI_CHAMBER_RSSPECTROMETER_CUH
