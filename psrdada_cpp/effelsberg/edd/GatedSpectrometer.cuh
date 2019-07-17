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
/**
 @class GatedSpectrometer
 @brief Split data into two streams and create integrated spectra depending on
 bit set in side channel data.

 */
template <class HandlerType, typename IntegratedPowerType> class GatedSpectrometer {
public:
  typedef uint64_t RawVoltageType;
  typedef float UnpackedVoltageType;
  typedef float2 ChannelisedVoltageType;

//  typedef float IntegratedPowerType;
  //typedef int8_t IntegratedPowerType;

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
  void process(thrust::device_vector<RawVoltageType> const &digitiser_raw,
               thrust::device_vector<uint64_t> const &sideChannelData,
               thrust::device_vector<IntegratedPowerType> &detected,
               thrust::device_vector<uint64_cu> &noOfBitSetsIn_G0,
               thrust::device_vector<uint64_cu> &noOfBitSetsIn_G1);

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
  std::unique_ptr<DetectorAccumulator<IntegratedPowerType> > _detector;

  // Input data
  DoubleDeviceBuffer<RawVoltageType> _raw_voltage_db;
  DoubleDeviceBuffer<uint64_t> _sideChannelData_db;

  // Output data
  DoubleDeviceBuffer<IntegratedPowerType> _power_db;


  DoubleDeviceBuffer<uint64_cu> _noOfBitSetsIn_G0;
  DoubleDeviceBuffer<uint64_cu> _noOfBitSetsIn_G1;
  DoublePinnedHostBuffer<char> _host_power_db;

  // Intermediate process steps
  thrust::device_vector<UnpackedVoltageType> _unpacked_voltage_G0;
  thrust::device_vector<UnpackedVoltageType> _unpacked_voltage_G1;
  thrust::device_vector<ChannelisedVoltageType> _channelised_voltage;
  thrust::device_vector<UnpackedVoltageType> _baseLineN;

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
                       const float *_baseLine, uint64_cu* stats_G0, uint64_cu*
                       stats_G1);




} // edd
} // effelsberg
} // psrdada_cpp

#include "psrdada_cpp/effelsberg/edd/detail/GatedSpectrometer.cu"
#endif //PSRDADA_CPP_EFFELSBERG_EDD_GATEDSPECTROMETER_HPP
