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

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {


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
//typedef int8_t IntegratedPowerType;

/// Input data and intermediate processing data for one polarization
struct PolarizationData
{
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
    void swap()
    {
        _raw_voltage.swap();
        _sideChannelData.swap();
    }
};


// Output data for one gate
struct StokesOutput
{
    /// Stokes parameters
    DoubleDeviceBuffer<IntegratedPowerType> I;
    DoubleDeviceBuffer<IntegratedPowerType> Q;
    DoubleDeviceBuffer<IntegratedPowerType> U;
    DoubleDeviceBuffer<IntegratedPowerType> V;

    /// Number of samples integrated in this output block
    DoubleDeviceBuffer<uint64_cu> _noOfBitSets;

    /// Reset outptu for new integration
    void reset(cudaStream_t &_proc_stream)
    {
      thrust::fill(thrust::cuda::par.on(_proc_stream),I.a().begin(), I.a().end(), 0.);
      thrust::fill(thrust::cuda::par.on(_proc_stream),Q.a().begin(), Q.a().end(), 0.);
      thrust::fill(thrust::cuda::par.on(_proc_stream),U.a().begin(), U.a().end(), 0.);
      thrust::fill(thrust::cuda::par.on(_proc_stream),V.a().begin(), V.a().end(), 0.);
      thrust::fill(thrust::cuda::par.on(_proc_stream), _noOfBitSets.a().begin(), _noOfBitSets.a().end(), 0L);
    }

    /// Swap output buffers
    void swap()
    {
      I.swap();
      Q.swap();
      U.swap();
      V.swap();
      _noOfBitSets.swap();
    }

    /// Resize all buffers
    void resize(size_t size, size_t blocks)
    {
      I.resize(size * blocks);
      Q.resize(size * blocks);
      U.resize(size * blocks);
      V.resize(size * blocks);
      _noOfBitSets.resize(blocks);
    }
};







/**
 @class GatedSpectrometer
 @brief Split data into two streams and create integrated spectra depending on
 bit set in side channel data.

 */
template <class HandlerType> class GatedSpectrometer {
public:



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
  // gate the data and fft data per gate
  void gated_fft(PolarizationData &data,
  thrust::device_vector<uint64_cu> &_noOfBitSetsIn_G0,
  thrust::device_vector<uint64_cu> &_noOfBitSetsIn_G1);

private:
  DadaBufferLayout _dadaBufferLayout;
  std::size_t _fft_length;
  std::size_t _naccumulate;
  std::size_t _nbits;
  std::size_t _selectedSideChannel;
  std::size_t _selectedBit;
  std::size_t _batch;
  std::size_t _nsamps_per_output_spectra;
  std::size_t _nsamps_per_buffer;
  std::size_t _nsamps_per_heap;

  HandlerType &_handler;
  cufftHandle _fft_plan;
  uint64_t _nchans;
  uint64_t _call_count;
  double _processing_efficiency;

  std::unique_ptr<Unpacker> _unpacker;

  // Input data and per pol intermediate data
  PolarizationData polarization0, polarization1;

  // Output data
  StokesOutput stokes_G0, stokes_G1;

  DoublePinnedHostBuffer<char> _host_power_db;

  // Temporary processing block
  // ToDo: Use inplace FFT to avoid temporary coltage array
  thrust::device_vector<UnpackedVoltageType> _unpacked_voltage_G0;
  thrust::device_vector<UnpackedVoltageType> _unpacked_voltage_G1;


  cudaStream_t _h2d_stream;
  cudaStream_t _proc_stream;
  cudaStream_t _d2h_stream;
};


/**
   * @brief      Splits the input data depending on a bit set into two arrays.
   *
   * @detail     The resulting gaps are filled with a given baseline value in the other stream.
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
//__host__ __device__ void stokes_IQUV(const float2 &p1, const float2 &p2, float &I, float &Q, float &U, float &V);
__host__ __device__ void stokes_IQUV(const float2 &p1, const float2 &p2, float &I, float &Q, float &U, float &V)
{
    I = fabs(p1.x*p1.x + p1.y * p1.y) + fabs(p2.x*p2.x + p2.y * p2.y);
    Q = fabs(p1.x*p1.x + p1.y * p1.y) - fabs(p2.x*p2.x + p2.y * p2.y);
    U = 2 * (p1.x*p2.x + p1.y * p2.y);
    V = -2 * (p1.y*p2.x - p1.x * p2.y);
}




/**
 * @brief calculate stokes IQUV spectra pol1, pol2 are arrays of naccumulate
 * complex spectra for individual polarizations
 */
__global__ void stokes_accumulate(float2 const __restrict__ *pol1,
        float2 const __restrict__ *pol2, float *I, float* Q, float *U, float*V,
        int nchans, int naccumulate)
{

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; (i < nchans);
       i += blockDim.x * gridDim.x)
  {
      float rI = 0;
      float rQ = 0;
      float rU = 0;
      float rV = 0;

      for (int k=0; k < naccumulate; k++)
      {
        const float2 p1 = pol1[i + k * nchans];
        const float2 p2 = pol2[i + k * nchans];

        rI += fabs(p1.x * p1.x + p1.y * p1.y) + fabs(p2.x * p2.x + p2.y * p2.y);
        rQ += fabs(p1.x * p1.x + p1.y * p1.y) - fabs(p2.x * p2.x + p2.y * p2.y);
        rU += 2.f * (p1.x * p2.x + p1.y * p2.y);
        rV += -2.f * (p1.y * p2.x - p1.x * p2.y);
      }
      I[i] += rI;
      Q[i] += rQ;
      U[i] += rU;
      V[i] += rV;
  }

}




} // edd
} // effelsberg
} // psrdada_cpp

#include "psrdada_cpp/effelsberg/edd/detail/GatedSpectrometer.cu"
#endif //PSRDADA_CPP_EFFELSBERG_EDD_GATEDSPECTROMETER_HPP
