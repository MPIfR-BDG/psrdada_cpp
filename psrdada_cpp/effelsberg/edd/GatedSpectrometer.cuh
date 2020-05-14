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



/// Input data and intermediate processing data for one polarization
struct PolarizationData
{
    DadaBufferLayout _dadaBufferLayout;
    size_t _fft_length;
    size_t _batch;

    // a buffer contains batch * fft_length samples
    PolarizationData(size_t fft_length, size_t batch, const DadaBufferLayout
            &dadaBufferLayout) : _fft_length(fft_length), _batch(batch), _dadaBufferLayout(dadaBufferLayout)
    {
        _raw_voltage.resize(_dadaBufferLayout.sizeOfData() / sizeof(uint64_t));
        BOOST_LOG_TRIVIAL(debug) << "  Input voltages size (in 64-bit words): " << _raw_voltage.size();

        _baseLineG0.resize(1);
        _baseLineG0_update.resize(1);
        _baseLineG1.resize(1);
        _baseLineG1_update.resize(1);
        _channelised_voltage_G0.resize((_fft_length / 2 + 1) * _batch);
        _channelised_voltage_G1.resize((_fft_length / 2 + 1) * _batch);
        _sideChannelData.resize(_dadaBufferLayout.getNSideChannels() * _dadaBufferLayout.getNHeaps());
        BOOST_LOG_TRIVIAL(debug) << "  Channelised voltages size: " << _channelised_voltage_G0.size();
    };

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

    void getFromBlock(RawBytes &block, cudaStream_t &_h2d_stream)
    {
      BOOST_LOG_TRIVIAL(debug) << "   block.used_bytes() = " << block.used_bytes()
                               << ", dataBlockBytes = " << _dadaBufferLayout.sizeOfData() << "\n";

      CUDA_ERROR_CHECK(cudaMemcpyAsync(static_cast<void *>(_raw_voltage.a_ptr()),
                                       static_cast<void *>(block.ptr()),
                                       _dadaBufferLayout.sizeOfData() , cudaMemcpyHostToDevice,
                                       _h2d_stream));
      CUDA_ERROR_CHECK(cudaMemcpyAsync(
          static_cast<void *>(_sideChannelData.a_ptr()),
          static_cast<void *>(block.ptr() + _dadaBufferLayout.sizeOfData() + _dadaBufferLayout.sizeOfGap()),
          _dadaBufferLayout.sizeOfSideChannelData(), cudaMemcpyHostToDevice, _h2d_stream));
      BOOST_LOG_TRIVIAL(debug) << "First side channel item: 0x" <<   std::setw(16)
          << std::setfill('0') << std::hex <<
          (reinterpret_cast<uint64_t*>(block.ptr() + _dadaBufferLayout.sizeOfData()
                                       + _dadaBufferLayout.sizeOfGap()))[0] <<
          std::dec;
    }
};


/// Input data and intermediate processing data for two polarizations
struct DualPolarizationData
{
    DadaBufferLayout _dadaBufferLayout;

    DualPolarizationData(size_t fft_length, size_t batch, const DadaBufferLayout
            &dadaBufferLayout) : polarization0(fft_length, batch, dadaBufferLayout),
                                polarization1(fft_length, batch, dadaBufferLayout),
                                _dadaBufferLayout(dadaBufferLayout)
    {
    };

    PolarizationData polarization0, polarization1;
    void swap()
    {
        polarization0.swap(); polarization1.swap();
    }

    void getFromBlock(RawBytes &block, cudaStream_t &_h2d_stream)
    {
    // Copy the data with stride to the GPU:
    // CPU: P1P2P1P2P1P2 ...
    // GPU: P1P1P1 ... P2P2P2 ...
    // If this is a bottleneck the gating kernel could sort the layout out
    // during copy
    int heapsize_bytes =  _dadaBufferLayout.getHeapSize();
    CUDA_ERROR_CHECK(cudaMemcpy2DAsync(
      static_cast<void *>(polarization0._raw_voltage.a_ptr()),
        heapsize_bytes,
        static_cast<void *>(block.ptr()),
        2 * heapsize_bytes,
        heapsize_bytes, _dadaBufferLayout.sizeOfData() / heapsize_bytes/ 2,
        cudaMemcpyHostToDevice, _h2d_stream));

    CUDA_ERROR_CHECK(cudaMemcpy2DAsync(
      static_cast<void *>(polarization1._raw_voltage.a_ptr()),
        heapsize_bytes,
        static_cast<void *>(block.ptr()) + heapsize_bytes,
        2 * heapsize_bytes,
        heapsize_bytes, _dadaBufferLayout.sizeOfData() / heapsize_bytes/ 2,
        cudaMemcpyHostToDevice, _h2d_stream));

    CUDA_ERROR_CHECK(cudaMemcpy2DAsync(
        static_cast<void *>(polarization0._sideChannelData.a_ptr()),
        sizeof(uint64_t),
        static_cast<void *>(block.ptr() + _dadaBufferLayout.sizeOfData() + _dadaBufferLayout.sizeOfGap()),
        2 * sizeof(uint64_t),
        sizeof(uint64_t),
        _dadaBufferLayout.sizeOfSideChannelData() / 2 / sizeof(uint64_t),
        cudaMemcpyHostToDevice, _h2d_stream));

    CUDA_ERROR_CHECK(cudaMemcpy2DAsync(
        static_cast<void *>(polarization1._sideChannelData.a_ptr()),
        sizeof(uint64_t),
        static_cast<void *>(block.ptr() + _dadaBufferLayout.sizeOfData() + _dadaBufferLayout.sizeOfGap() + sizeof(uint64_t)),
        2 * sizeof(uint64_t),
        sizeof(uint64_t),
        _dadaBufferLayout.sizeOfSideChannelData() / 2 / sizeof(uint64_t), cudaMemcpyHostToDevice, _h2d_stream));

    BOOST_LOG_TRIVIAL(debug) << "First side channel item: 0x" <<   std::setw(16)
        << std::setfill('0') << std::hex <<
        (reinterpret_cast<uint64_t*>(block.ptr() + _dadaBufferLayout.sizeOfData()
                                     + _dadaBufferLayout.sizeOfGap()))[0] << std::dec;
    }

};





/// INterface for the processed output data stream
struct OutputDataStream
{
    size_t _nchans;
    size_t _blocks;

    OutputDataStream(size_t nchans, size_t blocks) : _nchans(nchans), _blocks(blocks)
    {
    }

    /// Reset output to for new integration
    virtual void reset(cudaStream_t &_proc_stream) = 0;
    /// Swap output buffers
    virtual void swap() = 0;

    // output buffer on the host
    DoublePinnedHostBuffer<char> _host_power;

    // copy data from internal buffers of the implementation to the host output
    // buffer
    virtual void data2Host(cudaStream_t &_d2h_stream) = 0;
};


// Output data for one gate, single power spectrum
struct PowerSpectrumOutput
{
    PowerSpectrumOutput(size_t size, size_t blocks)
    {
        BOOST_LOG_TRIVIAL(debug) << "Setting size of power spectrum output size = " << size << ", blocks =  " << blocks;
       data.resize(size * blocks);
       _noOfBitSets.resize(blocks);
    }

    /// spectrum data
    DoubleDeviceBuffer<IntegratedPowerType> data;

    /// Number of samples integrated in this output block
    DoubleDeviceBuffer<uint64_cu> _noOfBitSets;

    /// Reset outptu for new integration
    void reset(cudaStream_t &_proc_stream)
    {
        thrust::fill(thrust::cuda::par.on(_proc_stream), data.a().begin(), data.a().end(), 0.);
        thrust::fill(thrust::cuda::par.on(_proc_stream), _noOfBitSets.a().begin(), _noOfBitSets.a().end(), 0L);
    }

    /// Swap output buffers
    void swap()
    {
        data.swap();
        _noOfBitSets.swap();
    }
};


struct GatedPowerSpectrumOutput : public OutputDataStream
{

    GatedPowerSpectrumOutput(size_t nchans, size_t blocks) : OutputDataStream(nchans, blocks),
        G0(nchans, blocks), G1(nchans, blocks)
    {
      // on the host both power are stored in the same data buffer together with
      // the number of bit sets
      _host_power.resize( 2 * ( _nchans * sizeof(IntegratedPowerType) + sizeof(size_t) ) * G0._noOfBitSets.size());
    }

    PowerSpectrumOutput G0, G1;

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

/*
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

    void data2Host(cudaStream_t &_d2h_stream)
    {
    for (size_t i = 0; i < G0._noOfBitSets.size(); i++)
    {
        size_t memslicesize = (_nchans * sizeof(IntegratedPowerType));
        size_t memOffset = 8 * i * (memslicesize +  + sizeof(size_t));
        // Copy  II QQ UU VV
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(static_cast<void *>(_host_power.a_ptr() + memOffset) ,
                            static_cast<void *>(G0.I.b_ptr() + i * memslicesize),
                            _nchans * sizeof(IntegratedPowerType),
                            cudaMemcpyDeviceToHost, _d2h_stream));

        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(static_cast<void *>(_host_power_db.a_ptr() + memOffset + 1 * memslicesize) ,
                            static_cast<void *>(G1.I.b_ptr() + i * memslicesize),
                            _nchans * sizeof(IntegratedPowerType),
                            cudaMemcpyDeviceToHost, _d2h_stream));

        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(static_cast<void *>(_host_power_db.a_ptr() + memOffset + 2 * memslicesize) ,
                            static_cast<void *>(G0.Q.b_ptr() + i * memslicesize),
                            _nchans * sizeof(IntegratedPowerType),
                            cudaMemcpyDeviceToHost, _d2h_stream));

        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(static_cast<void *>(_host_power_db.a_ptr() + memOffset + 3 * memslicesize) ,
                            static_cast<void *>(G1.Q.b_ptr() + i * memslicesize),
                            _nchans * sizeof(IntegratedPowerType),
                            cudaMemcpyDeviceToHost, _d2h_stream));

        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(static_cast<void *>(_host_power_db.a_ptr() + memOffset + 4 * memslicesize) ,
                            static_cast<void *>(G0.U.b_ptr() + i * memslicesize),
                            _nchans * sizeof(IntegratedPowerType),
                            cudaMemcpyDeviceToHost, _d2h_stream));

        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(static_cast<void *>(_host_power_db.a_ptr() + memOffset + 5 * memslicesize) ,
                            static_cast<void *>(G1.U.b_ptr() + i * memslicesize),
                            _nchans * sizeof(IntegratedPowerType),
                            cudaMemcpyDeviceToHost, _d2h_stream));

        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(static_cast<void *>(_host_power_db.a_ptr() + memOffset + 6 * memslicesize) ,
                            static_cast<void *>(G0.V.b_ptr() + i * memslicesize),
                            _nchans * sizeof(IntegratedPowerType),
                            cudaMemcpyDeviceToHost, _d2h_stream));

        CUDA_ERROR_CHECK(
            cudaMemcpyAsync(static_cast<void *>(_host_power_db.a_ptr() + memOffset + 7 * memslicesize) ,
                            static_cast<void *>(G1.V.b_ptr() + i * memslicesize),
                            _nchans * sizeof(IntegratedPowerType),
                            cudaMemcpyDeviceToHost, _d2h_stream));

        // Copy SCI
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync( static_cast<void *>(_host_power_db.a_ptr() + memOffset + 8 * memslicesize),
              static_cast<void *>(G0._noOfBitSets.b_ptr() + i ),
                1 * sizeof(size_t),
                cudaMemcpyDeviceToHost, _d2h_stream));
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync( static_cast<void *>(_host_power_db.a_ptr() + memOffset + 8 * memslicesize + 1 * sizeof(size_t)),
              static_cast<void *>(G1._noOfBitSets.b_ptr() + i ),
                1 * sizeof(size_t),
                cudaMemcpyDeviceToHost, _d2h_stream));
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync( static_cast<void *>(_host_power_db.a_ptr() + memOffset + 8 * memslicesize + 2 * sizeof(size_t)),
              static_cast<void *>(G0._noOfBitSets.b_ptr() + i ),
                1 * sizeof(size_t),
                cudaMemcpyDeviceToHost, _d2h_stream));
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync( static_cast<void *>(_host_power_db.a_ptr() + memOffset + 8 * memslicesize + 3 * sizeof(size_t)),
              static_cast<void *>(G1._noOfBitSets.b_ptr() + i ),
                1 * sizeof(size_t),
                cudaMemcpyDeviceToHost, _d2h_stream));
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync( static_cast<void *>(_host_power_db.a_ptr() + memOffset + 8 * memslicesize + 4 * sizeof(size_t)),
              static_cast<void *>(G0._noOfBitSets.b_ptr() + i ),
                1 * sizeof(size_t),
                cudaMemcpyDeviceToHost, _d2h_stream));
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync( static_cast<void *>(_host_power_db.a_ptr() + memOffset + 8 * memslicesize + 5 * sizeof(size_t)),
              static_cast<void *>(G1._noOfBitSets.b_ptr() + i ),
                1 * sizeof(size_t),
                cudaMemcpyDeviceToHost, _d2h_stream));
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync( static_cast<void *>(_host_power_db.a_ptr() + memOffset + 8 * memslicesize + 6 * sizeof(size_t)),
              static_cast<void *>(G0._noOfBitSets.b_ptr() + i ),
                1 * sizeof(size_t),
                cudaMemcpyDeviceToHost, _d2h_stream));
        CUDA_ERROR_CHECK(
            cudaMemcpyAsync( static_cast<void *>(_host_power_db.a_ptr() + memOffset + 8 * memslicesize + 7 * sizeof(size_t)),
              static_cast<void *>(G1._noOfBitSets.b_ptr() + i ),
                1 * sizeof(size_t),
                cudaMemcpyDeviceToHost, _d2h_stream));

      }



    }

};
*/



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

  // Processing strategy for single pol mode
  void process(PolarizationData *inputDataStream, GatedPowerSpectrumOutput *outputDataStream);

  // Processing strategy for dual pol  power mode
  //void process(DualPolarizationData*inputDataStream, GatedPowerSpectrumOutput *outputDataStream);

  // Processing strategy for full stokes mode
  //void process(DualPolarizationData*inputDataStream, FullStokesOutput *outputDataStream);

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




} // edd
} // effelsberg
} // psrdada_cpp

#include "psrdada_cpp/effelsberg/edd/detail/GatedSpectrometer.cu"
#endif //PSRDADA_CPP_EFFELSBERG_EDD_GATEDSPECTROMETER_HPP
