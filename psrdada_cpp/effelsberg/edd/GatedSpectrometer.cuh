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



struct InputDataStream
{
    virtual void swap() = 0;
    virtual void resize(size_t _nchans, size_t _batch, const DadaBufferLayout &dadaBufferLayout) = 0;
    virtual void getFromBlock(RawBytes &block, DadaBufferLayout &dadaBufferLayout, cudaStream_t &_h2d_stream) = 0;
};


/// Input data and intermediate processing data for one polarization
struct PolarizationData : public InputDataStream
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

    void resize(size_t _nchans, size_t _batch, const DadaBufferLayout &dadaBufferLayout)
    {
        _raw_voltage.resize(dadaBufferLayout.sizeOfData() / sizeof(uint64_t));
        BOOST_LOG_TRIVIAL(debug) << "  Input voltages size (in 64-bit words): " << _raw_voltage.size();

        _baseLineG0.resize(1);
        _baseLineG0_update.resize(1);
        _baseLineG1.resize(1);
        _baseLineG1_update.resize(1);
        _channelised_voltage_G0.resize(_nchans * _batch);
        _channelised_voltage_G1.resize(_nchans * _batch);
        BOOST_LOG_TRIVIAL(debug) << "  Channelised voltages size: " << _channelised_voltage_G0.size();
    }

    void getFromBlock(RawBytes &block, DadaBufferLayout &dadaBufferLayout, cudaStream_t &_h2d_stream)
    {
      BOOST_LOG_TRIVIAL(debug) << "   block.used_bytes() = " << block.used_bytes()
                               << ", dataBlockBytes = " << dadaBufferLayout.sizeOfData() << "\n";

      CUDA_ERROR_CHECK(cudaMemcpyAsync(static_cast<void *>(_raw_voltage.a_ptr()),
                                       static_cast<void *>(block.ptr()),
                                       dadaBufferLayout.sizeOfData() , cudaMemcpyHostToDevice,
                                       _h2d_stream));
      CUDA_ERROR_CHECK(cudaMemcpyAsync(
          static_cast<void *>(_sideChannelData.a_ptr()),
          static_cast<void *>(block.ptr() + dadaBufferLayout.sizeOfData() + dadaBufferLayout.sizeOfGap()),
          dadaBufferLayout.sizeOfSideChannelData(), cudaMemcpyHostToDevice, _h2d_stream));
      BOOST_LOG_TRIVIAL(debug) << "First side channel item: 0x" <<   std::setw(16)
          << std::setfill('0') << std::hex <<
          (reinterpret_cast<uint64_t*>(block.ptr() + dadaBufferLayout.sizeOfData()
                                       + dadaBufferLayout.sizeOfGap()))[0] <<
          std::dec;
    }


};


//struct DualPolarizationData : public InputDataStream
//{
//    PolarizationData pol0, pol1;
//    void swap()
//    {
//        pol0.swap(); pol1.swap();
//    }
//
//    void resize(size_t _nchans, size_t _batch, const DadaBufferLayout &dadaBufferLayout)
//    {
//        BOOST_LOG_TRIVIAL(debug) << "  Pol0";
//        pol0.resize(_nchans, _batch, dadaBufferLayout);
//        BOOST_LOG_TRIVIAL(debug) << "  Pol1";
//        pol1.resize(_nchans, _batch, dadaBufferLayout);
//    }
//};





// Output data for one gate N = 1 for one pol, or 4 for full stokes
struct OutputDataStream
{
    /// Reset outptu for new integration
    virtual void reset(cudaStream_t &_proc_stream) = 0;
    virtual void swap() = 0;
    virtual void resize(size_t size, size_t blocks) = 0;

    DoublePinnedHostBuffer<char> _host_power;
};


// Output data for one gate, single power spectrum
struct PowerSpectrumOutput
{
    /// spectrum data
    DoubleDeviceBuffer<IntegratedPowerType> data;

    /// Number of samples integrated in this output block
    DoubleDeviceBuffer<uint64_cu> _noOfBitSets;

    /// Reset outptu for new integration
    void reset(cudaStream_t &_proc_stream)
    {
        thrust::fill(thrust::cuda::par.on(_proc_stream), data.a().begin(),data.a().end(), 0.);
        thrust::fill(thrust::cuda::par.on(_proc_stream), _noOfBitSets.a().begin(), _noOfBitSets.a().end(), 0L);
    }

    /// Swap output buffers
    void swap()
    {
        data.swap();
        _noOfBitSets.swap();
    }

    /// Resize all buffers
    void resize(size_t size, size_t blocks)
    {
        data.resize(size * blocks);
        _noOfBitSets.resize(blocks);
    }
};


struct GatedPowerSpectrumOutput : public OutputDataStream
{
    PowerSpectrumOutput G0, G1;
    size_t _nchans;

    void reset(cudaStream_t &_proc_stream)
    {
        G0.reset(_proc_stream);
        G1.reset(_proc_stream);
    }

    /// Swap output buffers
    void swap()
    {
        G0.swap();
        G1.swap();
        _host_power.swap();
    }

    /// Resize all buffers
    void resize(size_t size, size_t blocks)
    {
        // ToDo:  size passed in constructor, also number of blocks.
        G0.resize(size, blocks);
        G1.resize(size, blocks);
        _nchans = size;

      // on the host both power are stored in the same data buffer together with
      // the number of bit sets
      _host_power.resize( 2 * ( size * sizeof(IntegratedPowerType) + sizeof(size_t) ) * G0._noOfBitSets.size());
    }

    void data2Host(cudaStream_t &_d2h_stream)
    {
        // copy data to host if block is finished
      CUDA_ERROR_CHECK(cudaStreamSynchronize(_d2h_stream));

      for (size_t i = 0; i < G0._noOfBitSets.size(); i++)
      {
        // size of individual spectrum + meta
        size_t memslicesize = (_nchans * sizeof(IntegratedPowerType));
        // number of spectra per output
        size_t memOffset = 2 * i * (memslicesize +  + sizeof(size_t));

        // copy 2x channel data
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(static_cast<void *>(_host_power.a_ptr() + memOffset) ,
                            static_cast<void *>(G0.data.b_ptr() + i * memslicesize),
                            _nchans * sizeof(IntegratedPowerType),
                            cudaMemcpyDeviceToHost, _d2h_stream));

        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(static_cast<void *>(_host_power.a_ptr() + memOffset + 1 * memslicesize) ,
                            static_cast<void *>(G1.data.b_ptr() + i * memslicesize),
                            _nchans * sizeof(IntegratedPowerType),
                            cudaMemcpyDeviceToHost, _d2h_stream));

        // copy noOf bit set data
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync( static_cast<void *>(_host_power.a_ptr() + memOffset + 2 * _nchans * sizeof(IntegratedPowerType)),
              static_cast<void *>(G0._noOfBitSets.b_ptr() + i ),
                1 * sizeof(size_t),
                cudaMemcpyDeviceToHost, _d2h_stream));

        CUDA_ERROR_CHECK(
            cudaMemcpyAsync( static_cast<void *>(_host_power.a_ptr() + memOffset + 2 * _nchans * sizeof(IntegratedPowerType) + sizeof(size_t)),
              static_cast<void *>(G1._noOfBitSets.b_ptr() + i ),
                1 * sizeof(size_t),
                cudaMemcpyDeviceToHost, _d2h_stream));
      }
    }


};


// Output data for one gate full stokes
struct FullStokesOutput : public OutputDataStream
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
        thrust::fill(thrust::cuda::par.on(_proc_stream), I.a().begin(), I.a().end(), 0.);
        thrust::fill(thrust::cuda::par.on(_proc_stream), Q.a().begin(), Q.a().end(), 0.);
        thrust::fill(thrust::cuda::par.on(_proc_stream), U.a().begin(), U.a().end(), 0.);
        thrust::fill(thrust::cuda::par.on(_proc_stream), V.a().begin(), V.a().end(), 0.);
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


struct GatedFullStokesOutput: public OutputDataStream
{
    FullStokesOutput G0, G1;
     void reset(cudaStream_t &_proc_stream)
    {
        G0.reset(_proc_stream);
        G1.reset(_proc_stream);
    }

    /// Swap output buffers
    void swap()
    {
        G0.swap();
        G1.swap();
    }

    /// Resize all buffers
    void resize(size_t size, size_t blocks)
    {
        G0.resize(size, blocks);
        G1.resize(size, blocks);
    }
};




/**
 @class GatedSpectrometer
 @brief Split data into two streams and create integrated spectra depending on
 bit set in side channel data.

 */
template <class HandlerType > class GatedSpectrometer {
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
  // gate the data from the input stream and fft data per gate. Number of bit
  // sets in every gate is stored in the output datasets
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
  uint64_t _nBlocks;
  uint64_t _call_count;
  double _processing_efficiency;

  std::unique_ptr<Unpacker> _unpacker;

  OutputDataStream* outputDataStream;
  InputDataStream* inputDataStream;

  std::unique_ptr<DetectorAccumulator<IntegratedPowerType> > _detector;

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




} // edd
} // effelsberg
} // psrdada_cpp

#include "psrdada_cpp/effelsberg/edd/detail/GatedSpectrometer.cu"
#endif //PSRDADA_CPP_EFFELSBERG_EDD_GATEDSPECTROMETER_HPP
