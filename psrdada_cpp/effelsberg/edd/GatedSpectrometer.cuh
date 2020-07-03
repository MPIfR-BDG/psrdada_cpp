#ifndef PSRDADA_CPP_EFFELSBERG_EDD_GATEDSPECTROMETER_HPP
#define PSRDADA_CPP_EFFELSBERG_EDD_GATEDSPECTROMETER_HPP

#include "psrdada_cpp/effelsberg/edd/Unpacker.cuh"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/double_device_buffer.cuh"
#include "psrdada_cpp/double_host_buffer.cuh"
#include "psrdada_cpp/effelsberg/edd/DetectorAccumulator.cuh"
#include "psrdada_cpp/effelsberg/edd/DadaBufferLayout.hpp"

#include "thrust/device_vector.h"
#include "cufft.h"

#include "cublas_v2.h"


#include <iostream>
#include <iomanip>
#include <cstring>
#include <sstream>


namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

/// Macro to manipulate single bits of an 64-bit type.
#define BIT_MASK(bit) (1uL << (bit))
#define SET_BIT(value, bit) ((value) |= BIT_MASK(bit))
#define CLEAR_BIT(value, bit) ((value) &= ~BIT_MASK(bit))
#define TEST_BIT(value, bit) (((value)&BIT_MASK(bit)) ? 1 : 0)

typedef unsigned long long int uint64_cu;
static_assert(sizeof(uint64_cu) == sizeof(uint64_t), "Long long int not of 64 bit! This is problematic for CUDA!");

typedef uint64_t RawVoltageType;
typedef float UnpackedVoltageType;
typedef float2 ChannelisedVoltageType;
typedef float IntegratedPowerType;

/**
 @class PolarizationData
 @brief Device data arrays for raw voltage input and intermediate processing data for one polarization
 */
struct PolarizationData
{
    size_t _nbits;
    /**
    * @brief      Constructor
    *
    * @param      nbits Bit-depth of the input data.
    */
    PolarizationData(size_t nbits): _nbits(nbits) {};
    /// Raw ADC Voltage
    DoubleDeviceBuffer<RawVoltageType> _raw_voltage;
    /// Side channel data
    DoubleDeviceBuffer<uint64_t> _sideChannelData;

    /// Baseline in gate 0 state
    thrust::device_vector<UnpackedVoltageType> _baseLineG0;
    /// Baseline in gate 1 state
    thrust::device_vector<UnpackedVoltageType> _baseLineG1;

    /// Baseline in gate 0 state after update
    thrust::device_vector<UnpackedVoltageType> _baseLineG0_update;
    /// Baseline in gate 1 state after update
    thrust::device_vector<UnpackedVoltageType> _baseLineG1_update;

    /// Channelized voltage in gate 0 state
    thrust::device_vector<ChannelisedVoltageType> _channelised_voltage_G0;
    /// Channelized voltage in gate 1 state
    thrust::device_vector<ChannelisedVoltageType> _channelised_voltage_G1;

    /// Swaps input buffers
    void swap();

    ///resize the internal buffers
    void resize(size_t rawVolttageBufferBytes, size_t nsidechannelitems, size_t channelized_samples);
};


/**
 @class SinglePolarizationInput
 @brief Input data for a buffer containing one polarization
 */
class SinglePolarizationInput : public PolarizationData
{
    DadaBufferLayout _dadaBufferLayout;
    size_t _fft_length;

public:

    /**
    * @brief      Constructor
    *
    * @param      fft_length length of the fft.
    * @param      nbits bit-depth of the input data.
    * @param      dadaBufferLayout layout of the input dada buffer
    */
    SinglePolarizationInput(size_t fft_length, size_t nbits,
            const DadaBufferLayout &dadaBufferLayout);

    /// Copy data from input block to input dubble buffer
    void getFromBlock(RawBytes &block, cudaStream_t &_h2d_stream);

    /// Number of samples per input polarization
    size_t getSamplesPerInputPolarization();
};


/**
 @class SinglePolarizationInput
 @brief Input data for a buffer containing two polarizations
 */
class DualPolarizationInput
{
    DadaBufferLayout _dadaBufferLayout;
    size_t _fft_length;

    public:
    PolarizationData polarization0, polarization1;

    /**
    * @brief      Constructor
    *
    * @param      fft_length length of the fft.
    * @param      nbits bit-depth of the input data.
    * @param      dadaBufferLayout layout of the input dada buffer
    */
    DualPolarizationInput(size_t fft_length, size_t nbits, const DadaBufferLayout
            &dadaBufferLayout);

    /// Swaps input buffers for both polarizations
    void swap();

    /// Copy data from input block to input dubble buffer
    void getFromBlock(RawBytes &block, cudaStream_t &_h2d_stream);

    /// Number of samples per input polarization
    size_t getSamplesPerInputPolarization();
};




/**
 @class PowerSpectrumOutput
 @brief Output data for one gate, single power spectrum
 */
struct PowerSpectrumOutput
{
    /**
    * @brief      Constructor
    *
    * @param      size size of the output, i.e. number of channels.
    * @param      blocks number of blocks in the output.
    */
    PowerSpectrumOutput(size_t size, size_t blocks);

    /// spectrum data
    DoubleDeviceBuffer<IntegratedPowerType> data;

    /// Number of samples integrated in this output block
    DoubleDeviceBuffer<uint64_cu> _noOfBitSets;

    /// Swap output buffers and reset the buffer in given stream for new integration
    void swap(cudaStream_t &_proc_stream);
};


/**
 @class OutputDataStream
 @brief Interface for the processed output data stream
 */
struct OutputDataStream
{
    size_t _nchans;
    size_t _blocks;

    /**
    * @brief      Constructor
    *
    * @param      nchans number of channels.
    * @param      blocks number of blocks in the output.
    */
    OutputDataStream(size_t nchans, size_t blocks) : _nchans(nchans), _blocks(blocks)
    {
    }

    /// Swap output buffers
    virtual void swap(cudaStream_t &_proc_stream) = 0;

    // output buffer on the host
    DoublePinnedHostBuffer<char> _host_power;

    // copy data from internal buffers of the implementation to the host output
    // buffer
    virtual void data2Host(cudaStream_t &_d2h_stream) = 0;
};


/**
 @class GatedPowerSpectrumOutput
 @brief Output Stream for power spectrum output
 */
struct GatedPowerSpectrumOutput : public OutputDataStream
{
    GatedPowerSpectrumOutput(size_t nchans, size_t blocks);

    /// Power spectrum for off and on gate
    PowerSpectrumOutput G0, G1;

    void swap(cudaStream_t &_proc_stream);

    void data2Host(cudaStream_t &_d2h_stream);
};


/**
 @class FullStokesOutput
 @brief Output data for one gate full stokes
 */
struct FullStokesOutput
{
    /**
    * @brief      Constructor
    *
    * @param      size size of the output, i.e. number of channels.
    * @param      blocks number of blocks in the output.
    */
    FullStokesOutput(size_t size, size_t blocks);

    /// Buffer for Stokes Parameters
    DoubleDeviceBuffer<IntegratedPowerType> I, Q, U, V;

    /// Number of samples integrated in this output block
    DoubleDeviceBuffer<uint64_cu> _noOfBitSets;

    /// Swap output buffers
    void swap(cudaStream_t &_proc_stream);
};


/**
 @class GatedPowerSpectrumOutput
 @brief Output Stream for power spectrum output
 */
struct GatedFullStokesOutput: public OutputDataStream
{
    /// stokes output for on/off gate
    FullStokesOutput G0, G1;

    GatedFullStokesOutput(size_t nchans, size_t blocks);

    /// Swap output buffers
    void swap(cudaStream_t &_proc_stream);

    void data2Host(cudaStream_t &_d2h_stream);
};




/**
 @class GatedSpectrometer
 @brief Split data into two streams and create integrated spectra depending on
 bit set in side channel data.
 */
template <class HandlerType,
         class InputType,
         class OutputType
         > class GatedSpectrometer {
public:
  /**
   * @brief      Constructor
   *
   * @param      buffer_bytes A RawBytes object wrapping a DADA header buffer
   * @param      nSideChannels Number of side channel items in the data stream,
   * @param      selectedSideChannel Side channel item used for gating
   * @param      selectedBit bit of side channel item used for gating
   * @param      speadHeapSize Size of the spead heap block.
   * @param      fftLength Size of the FFT
   * @param      naccumulate Number of samples to integrate in the individual
   *             FFT bins
   * @param      nbits Bit depth of the sampled signal
   * @param      input_level Normalization level of the input signal
   * @param      output_level Normalization level of the output signal
   * @param      handler Output handler
   *
   */
  GatedSpectrometer(const DadaBufferLayout &bufferLayout,
                    std::size_t selectedSideChannel, std::size_t selectedBit,
                     std::size_t fft_length,
                    std::size_t naccumulate, std::size_t nbits,
                    float input_level, float output_level,
                    HandlerType &handler);
  ~GatedSpectrometer();

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
  void init(RawBytes &block);

  /**
   * @brief      A callback to be called on acqusition of a new
   *             data block.
   *
   * @param      block  A RawBytes object wrapping a DADA data buffer output
   *             are the integrated specttra with/without bit set.
   */
  bool operator()(RawBytes &block);

private:
  // Processing strategy for single pol mode
  void process(SinglePolarizationInput *inputDataStream, GatedPowerSpectrumOutput *outputDataStream);

  // Processing strategy for dual pol  power mode
  //void process(DualPolarizationInput*inputDataStream, GatedPowerSpectrumOutput *outputDataStream);

  // Processing strategy for full stokes mode
  void process(DualPolarizationInput *inputDataStream, GatedFullStokesOutput *outputDataStream);

  // gate the data from the input stream and fft data per gate. Number of bit
  // sets in every gate is stored in the output datasets
  void gated_fft(PolarizationData &data,
    thrust::device_vector<uint64_cu> &_noOfBitSetsIn_G0,
    thrust::device_vector<uint64_cu> &_noOfBitSetsIn_G1);

private:
  DadaBufferLayout _dadaBufferLayout;
  std::size_t _fft_length;
  std::size_t _naccumulate;
  std::size_t _selectedSideChannel;
  std::size_t _selectedBit;
  std::size_t _batch;
  std::size_t _nsamps_per_heap;

  HandlerType &_handler;
  cufftHandle _fft_plan;
  uint64_t _nchans;
  uint64_t _nBlocks;
  uint64_t _call_count;

  std::unique_ptr<Unpacker> _unpacker;

  OutputType* outputDataStream;
  InputType* inputDataStream;

  // Temporary processing block
  // ToDo: Use inplace FFT to avoid temporary voltage array
  thrust::device_vector<UnpackedVoltageType> _unpacked_voltage_G0;
  thrust::device_vector<UnpackedVoltageType> _unpacked_voltage_G1;

  cudaStream_t _h2d_stream;
  cudaStream_t _proc_stream;
  cudaStream_t _d2h_stream;
};



/**
   * @brief      Splits the input data depending on a bit set into two arrays.
   *
   * @detail     The resulting gaps are filled with zeros in the other stream.
   *
   * @param      GO Input data. Data is set to the baseline value if corresponding
   *             sideChannelData bit at bitpos os set.
   * @param      G1 Data in this array is set to the baseline value if corresponding
   *             sideChannelData bit at bitpos is not set.
   * @param      sideChannelData noOfSideChannels items per block of heapSize
   *             bytes in the input data.
   * @param      N lebgth of the input/output arrays G0.
   * @param      heapsize Size of the blocks for which there is an entry in the
                 sideChannelData.
   * @param      bitpos Position of the bit to evaluate for processing.
   * @param      noOfSideChannels Number of side channels items per block of
   *             data.
   * @param      selectedSideChannel No. of side channel item to be eveluated.
                 0 <= selectedSideChannel < noOfSideChannels.
   * @param      stats_G0 No. of sampels contributing to G0, accounting also
   *             for loat heaps
   * @param      stats_G1 No. of sampels contributing to G1, accounting also
   *             for loat heaps
   */
__global__ void gating(float *G0, float *G1, const int64_t *sideChannelData,
                       size_t N, size_t heapSize, size_t bitpos,
                       size_t noOfSideChannels, size_t selectedSideChannel,
                       const float*  __restrict__ _baseLineG0,
                       const float*  __restrict__ _baseLineG1,
                       float* __restrict__ baseLineNG0,
                       float* __restrict__ baseLineNG1,
                       uint64_cu* stats_G0,
                       uint64_cu* stats_G1);


/**
 * @brief calculate stokes IQUV from two complex valuies for each polarization
 */
__host__ __device__ void stokes_IQUV(const float2 &p1, const float2 &p2, float &I, float &Q, float &U, float &V);


/**
 * @brief calculate stokes IQUV spectra pol1, pol2 are arrays of naccumulate
 * complex spectra for individual polarizations
 */
__global__ void stokes_accumulate(float2 const __restrict__ *pol1,
        float2 const __restrict__ *pol2, float *I, float* Q, float *U, float*V,
        int nchans, int naccumulate);



} // edd
} // effelsberg
} // psrdada_cpp

#include "psrdada_cpp/effelsberg/edd/detail/GatedSpectrometer.cu"
#endif //PSRDADA_CPP_EFFELSBERG_EDD_GATEDSPECTROMETER_HPP
