#include "psrdada_cpp/effelsberg/edd/GatedSpectrometer.cuh"
#include "psrdada_cpp/effelsberg/edd/Tools.cuh"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/raw_bytes.hpp"

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <thrust/system/cuda/execution_policy.h>

#include <iostream>
#include <iomanip>
#include <cstring>
#include <sstream>
#include <typeinfo>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

// Reduce thread local vatiable v in shared array x, so that x[0] contains sum
template<typename T>
__device__ void sum_reduce(T *x, const T &v)
{
  x[threadIdx.x] = v;
  __syncthreads();
  for(int s = blockDim.x / 2; s > 0; s = s / 2)
  {
    if (threadIdx.x < s)
      x[threadIdx.x] += x[threadIdx.x + s];
    __syncthreads();
  }
}


// If one of the side channel items is lost, then both are considered as lost
// here
__global__ void mergeSideChannels(uint64_t* __restrict__ A, uint64_t*
        __restrict__ B, size_t N);


__global__ void gating(float* __restrict__ G0,
        float* __restrict__ G1,
        const uint64_t* __restrict__ sideChannelData,
        size_t N, size_t heapSize, size_t bitpos,
        size_t noOfSideChannels, size_t selectedSideChannel,
        const float*  __restrict__ _baseLineG0,
        const float*  __restrict__ _baseLineG1,
        float* __restrict__ baseLineNG0,
        float* __restrict__ baseLineNG1,
        uint64_cu* stats_G0, uint64_cu* stats_G1);


// Updates the baselines of the gates for the polarization set for the next
// block
// only few output blocks per input block thus execution on only one thread.
// Important is that the execution is async on the GPU.
__global__ void update_baselines(float*  __restrict__ baseLineG0,
        float*  __restrict__ baseLineG1,
        float* __restrict__ baseLineNG0,
        float* __restrict__ baseLineNG1,
        uint64_cu* stats_G0, uint64_cu* stats_G1,
        size_t N);


template <class HandlerType, class InputType, class OutputType>
GatedSpectrometer<HandlerType, InputType, OutputType>::GatedSpectrometer(
    const DadaBufferLayout &dadaBufferLayout, std::size_t selectedSideChannel,
    std::size_t selectedBit, std::size_t fft_length, std::size_t naccumulate,
    std::size_t nbits, float input_level, float output_level, HandlerType
    &handler) : _dadaBufferLayout(dadaBufferLayout),
    _selectedSideChannel(selectedSideChannel), _selectedBit(selectedBit),
    _fft_length(fft_length), _naccumulate(naccumulate),
    _handler(handler), _fft_plan(0), _call_count(0), _nsamps_per_heap(4096)
{

  // Sanity checks
  assert(((nbits == 12) || (nbits == 8)));
  assert(_naccumulate > 0);

  // check for any device errors
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());

  BOOST_LOG_TRIVIAL(info)
      << "Creating new GatedSpectrometer instance with parameters: \n"
      << "  fft_length           " << _fft_length << "\n"
      << "  naccumulate          " << _naccumulate << "\n"
      << "  nSideChannels        " << _dadaBufferLayout.getNSideChannels() << "\n"
      << "  speadHeapSize        " << _dadaBufferLayout.getHeapSize() << " byte\n"
      << "  selectedSideChannel  " << _selectedSideChannel << "\n"
      << "  selectedBit          " << _selectedBit << "\n"
      << "  output bit depth     " << sizeof(IntegratedPowerType) * 8;

  assert((_dadaBufferLayout.getNSideChannels() == 0) ||
         (selectedSideChannel < _dadaBufferLayout.getNSideChannels()));  // Sanity check of side channel value
  assert(selectedBit < 64); // Sanity check of selected bit


  _nchans = _fft_length / 2 + 1;

  // Calculate the scaling parameters for 8 bit output
  float dof = 2 * _naccumulate;
  float scale =
      std::pow(input_level * std::sqrt(static_cast<float>(_nchans)), 2);
  float offset = scale * dof;
  float scaling = scale * std::sqrt(2 * dof) / output_level;
  BOOST_LOG_TRIVIAL(debug)
      << "Correction factors for 8-bit conversion: offset = " << offset
      << ", scaling = " << scaling;

  inputDataStream = new InputType(fft_length, nbits, _dadaBufferLayout);

  //How many output spectra per input block?
  size_t nsamps_per_output_spectra = fft_length * naccumulate;

  size_t nsamps_per_pol = inputDataStream->getSamplesPerInputPolarization();
  if (nsamps_per_output_spectra <= nsamps_per_pol)
  { // one buffer block is used for one or multiple output spectra
    size_t N = nsamps_per_pol / nsamps_per_output_spectra;
    // All data in one block has to be used
    assert(N * nsamps_per_output_spectra == nsamps_per_pol);
    _nBlocks = 1;
  }
  else
  { // multiple blocks are integrated intoone output
    size_t N =  nsamps_per_output_spectra /  nsamps_per_pol;
    // All data in multiple blocks has to be used
    assert(N * nsamps_per_pol == nsamps_per_output_spectra);
    _nBlocks = N;
  }
  BOOST_LOG_TRIVIAL(debug) << "Integrating  " << nsamps_per_output_spectra <<
      " samples from " << _nBlocks << "blocks into one output spectrum.";


  // plan the FFT
  size_t nsamps_per_buffer = _dadaBufferLayout.sizeOfData() * 8 / nbits;
  int batch = nsamps_per_pol / _fft_length;
  int n[] = {static_cast<int>(_fft_length)};
  BOOST_LOG_TRIVIAL(debug) << "Generating FFT plan: \n"
      << "   fft_length = " << _fft_length << "\n"
      << "   n[0] = " << n[0] << "\n"
      << "   _nchans = " << _nchans << "\n"
      << "   batch = " << batch << "\n";
  CUFFT_ERROR_CHECK(cufftPlanMany(&_fft_plan, 1, n, NULL, 1, _fft_length, NULL,
                                  1, _nchans, CUFFT_R2C, batch));


  // We unpack and fft one pol at a time
  _unpacked_voltage_G0.resize(nsamps_per_pol);
  _unpacked_voltage_G1.resize(nsamps_per_pol);
  BOOST_LOG_TRIVIAL(debug) << "  Unpacked voltages size (in samples): " << _unpacked_voltage_G0.size();

  outputDataStream = new OutputType(_nchans, batch / (_naccumulate / _nBlocks));

  CUDA_ERROR_CHECK(cudaStreamCreate(&_h2d_stream));
  CUDA_ERROR_CHECK(cudaStreamCreate(&_proc_stream));
  CUDA_ERROR_CHECK(cudaStreamCreate(&_d2h_stream));
  CUFFT_ERROR_CHECK(cufftSetStream(_fft_plan, _proc_stream));

  _unpacker.reset(new Unpacker(_proc_stream));
} // constructor


template <class HandlerType, class InputType, class OutputType>
GatedSpectrometer<HandlerType, InputType, OutputType>::~GatedSpectrometer() {
  BOOST_LOG_TRIVIAL(debug) << "Destroying GatedSpectrometer";
  if (_fft_plan)
    cufftDestroy(_fft_plan);

  delete inputDataStream;
  delete outputDataStream;

  cudaStreamDestroy(_h2d_stream);
  cudaStreamDestroy(_proc_stream);
  cudaStreamDestroy(_d2h_stream);
}


template <class HandlerType, class InputType, class OutputType>
void GatedSpectrometer<HandlerType, InputType, OutputType>::init(RawBytes &block) {
  BOOST_LOG_TRIVIAL(debug) << "GatedSpectrometer init called";
  std::stringstream headerInfo;
  headerInfo << "\n"
      << "# Gated spectrometer parameters: \n"
      << "fft_length               " << _fft_length << "\n"
      << "nchannels                " << _nchans << "\n"
      << "naccumulate              " << _naccumulate << "\n"
      << "selected_side_channel    " << _selectedSideChannel << "\n"
      << "selected_bit             " << _selectedBit << "\n"
      << "output_bit_depth         " << sizeof(IntegratedPowerType) * 8 << "\n"
      << "full_stokes_output       ";
  if (typeid(OutputType) == typeid(GatedFullStokesOutput))
  {
          headerInfo << "yes\n";
  }
  else
  {
          headerInfo << "no\n";
  }

  size_t bEnd = std::strlen(block.ptr());
  if (bEnd + headerInfo.str().size() < block.total_bytes())
  {
    std::strcpy(block.ptr() + bEnd, headerInfo.str().c_str());
  }
  else
  {
    BOOST_LOG_TRIVIAL(warning) << "Header of size " << block.total_bytes()
      << " bytes already contains " << bEnd
      << "bytes. Cannot add gated spectrometer info of size "
      << headerInfo.str().size() << " bytes.";
  }

  _handler.init(block);
}



template <class HandlerType, class InputType, class OutputType>
void GatedSpectrometer<HandlerType, InputType, OutputType>::gated_fft(
  PolarizationData &data,
  thrust::device_vector<uint64_cu> &_noOfBitSetsIn_G0,
  thrust::device_vector<uint64_cu> &_noOfBitSetsIn_G1
        )
{
  BOOST_LOG_TRIVIAL(debug) << "Unpacking raw voltages";
  switch (data._nbits) {
  case 8:
    _unpacker->unpack<8>(data._raw_voltage.b(), _unpacked_voltage_G0);
    break;
  case 12:
    _unpacker->unpack<12>(data._raw_voltage.b(), _unpacked_voltage_G0);
    break;
  default:
    throw std::runtime_error("Unsupported number of bits");
  }

  // Loop over outputblocks, for case of multiple output blocks per input block
  int step = data._sideChannelData.b().size() / _noOfBitSetsIn_G0.size();

  for (size_t i = 0; i < _noOfBitSetsIn_G0.size(); i++)
  { // ToDo: Should be in one kernel call
  gating<<<1024, 1024, 0, _proc_stream>>>(
      thrust::raw_pointer_cast(_unpacked_voltage_G0.data() + i * step * _nsamps_per_heap),
      thrust::raw_pointer_cast(_unpacked_voltage_G1.data() + i * step * _nsamps_per_heap),
      thrust::raw_pointer_cast(data._sideChannelData.b().data() + i * step),
      _unpacked_voltage_G0.size() / _noOfBitSetsIn_G0.size(),
      _dadaBufferLayout.getHeapSize(),
      _selectedBit,
      _dadaBufferLayout.getNSideChannels(),
      _selectedSideChannel,
      thrust::raw_pointer_cast(data._baseLineG0.data()),
      thrust::raw_pointer_cast(data._baseLineG1.data()),
      thrust::raw_pointer_cast(data._baseLineG0_update.data()),
      thrust::raw_pointer_cast(data._baseLineG1_update.data()),
      thrust::raw_pointer_cast(_noOfBitSetsIn_G0.data() + i),
      thrust::raw_pointer_cast(_noOfBitSetsIn_G1.data() + i)
      );
  }

    // only few output blocks per input block thus execution on only one thread.
    // Important is that the execution is async on the GPU.
    update_baselines<<<1,1,0, _proc_stream>>>(
        thrust::raw_pointer_cast(data._baseLineG0.data()),
        thrust::raw_pointer_cast(data._baseLineG1.data()),
        thrust::raw_pointer_cast(data._baseLineG0_update.data()),
        thrust::raw_pointer_cast(data._baseLineG1_update.data()),
        thrust::raw_pointer_cast(_noOfBitSetsIn_G0.data()),
        thrust::raw_pointer_cast(_noOfBitSetsIn_G1.data()),
        _noOfBitSetsIn_G0.size()
            );

  BOOST_LOG_TRIVIAL(debug) << "Performing FFT 1";
  BOOST_LOG_TRIVIAL(debug) << "Accessing unpacked voltage";
  UnpackedVoltageType *_unpacked_voltage_ptr =
      thrust::raw_pointer_cast(_unpacked_voltage_G0.data());
  BOOST_LOG_TRIVIAL(debug) << "Accessing channelized voltage";
  ChannelisedVoltageType *_channelised_voltage_ptr =
      thrust::raw_pointer_cast(data._channelised_voltage_G0.data());

  CUFFT_ERROR_CHECK(cufftExecR2C(_fft_plan, (cufftReal *)_unpacked_voltage_ptr,
                                 (cufftComplex *)_channelised_voltage_ptr));

  BOOST_LOG_TRIVIAL(debug) << "Performing FFT 2";
  _unpacked_voltage_ptr = thrust::raw_pointer_cast(_unpacked_voltage_G1.data());
  _channelised_voltage_ptr = thrust::raw_pointer_cast(data._channelised_voltage_G1.data());
  CUFFT_ERROR_CHECK(cufftExecR2C(_fft_plan, (cufftReal *)_unpacked_voltage_ptr,
                                 (cufftComplex *)_channelised_voltage_ptr));

//  CUDA_ERROR_CHECK(cudaStreamSynchronize(_proc_stream));
//  BOOST_LOG_TRIVIAL(debug) << "Exit processing";
} // process






template <class HandlerType, class InputType, class OutputType>
bool GatedSpectrometer<HandlerType, InputType, OutputType>::operator()(RawBytes &block) {
  ++_call_count;
  BOOST_LOG_TRIVIAL(debug) << "GatedSpectrometer operator() called (count = "
                           << _call_count << ")";
  if (block.used_bytes() != _dadaBufferLayout.getBufferSize()) { /* Unexpected buffer size */
    BOOST_LOG_TRIVIAL(error) << "Unexpected Buffer Size - Got "
                             << block.used_bytes() << " byte, expected "
                             << _dadaBufferLayout.getBufferSize() << " byte)";
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    cudaProfilerStop();
    return true;
  }

  // Copy data to device
  CUDA_ERROR_CHECK(cudaStreamSynchronize(_h2d_stream));
  inputDataStream->swap();
  inputDataStream->getFromBlock(block, _h2d_stream);


  if (_call_count == 1) {
    return false;
  }
  // process data

  // check if new outblock is started:  _call_count -1 because this is the block number on the device
  bool newBlock = (((_call_count-1)  % (_nBlocks)) == 0);

  // only if  a newblock is started the output buffer is swapped. Otherwise the
  // new data is added to it
  if (newBlock)
  {
    BOOST_LOG_TRIVIAL(debug) << "Starting new output block.";
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_d2h_stream));
    outputDataStream->swap(_proc_stream);
  }

  BOOST_LOG_TRIVIAL(debug) << "Processing block.";
  process(inputDataStream, outputDataStream);
  CUDA_ERROR_CHECK(cudaStreamSynchronize(_proc_stream));
  BOOST_LOG_TRIVIAL(debug) << "Processing block finished.";
  /// For one pol input and power out
  /// ToDo: For two pol input and power out
  /// ToDo: For two pol input and stokes out


  if ((_call_count == 2) || (!newBlock)) {
    return false;
  }

  outputDataStream->data2Host(_d2h_stream);
  if (_call_count == 3) {
    return false;
  }

  // Wrap in a RawBytes object here;
  RawBytes bytes(reinterpret_cast<char *>(outputDataStream->_host_power.b_ptr()),
                 outputDataStream->_host_power.size(),
                 outputDataStream->_host_power.size());
  BOOST_LOG_TRIVIAL(debug) << "Calling handler";
  // The handler can't do anything asynchronously without a copy here
  // as it would be unsafe (given that it does not own the memory it
  // is being passed).

  _handler(bytes);
  return false; //
} // operator ()



template <class HandlerType, class InputType, class OutputType>
void GatedSpectrometer<HandlerType, InputType, OutputType>::process(SinglePolarizationInput *inputDataStream, GatedPowerSpectrumOutput *outputDataStream)
{
  gated_fft(*inputDataStream, outputDataStream->G0._noOfBitSets.a(), outputDataStream->G1._noOfBitSets.a());



  kernels::detect_and_accumulate<IntegratedPowerType> <<<1024, 1024, 0, _proc_stream>>>(
            thrust::raw_pointer_cast(inputDataStream->_channelised_voltage_G0.data()),
            thrust::raw_pointer_cast(outputDataStream->G0.data.a().data()),
            _nchans,
            inputDataStream->_channelised_voltage_G0.size() / _nchans,
            _naccumulate / _nBlocks,
            1, 0., 1, 0);

  kernels::detect_and_accumulate<IntegratedPowerType> <<<1024, 1024, 0, _proc_stream>>>(
            thrust::raw_pointer_cast(inputDataStream->_channelised_voltage_G1.data()),
            thrust::raw_pointer_cast(outputDataStream->G1.data.a().data()),
            _nchans,
            inputDataStream->_channelised_voltage_G1.size() / _nchans,
            _naccumulate / _nBlocks,
            1, 0., 1, 0);

}


template <class HandlerType, class InputType, class OutputType>
void GatedSpectrometer<HandlerType, InputType, OutputType>::process(DualPolarizationInput *inputDataStream, GatedFullStokesOutput *outputDataStream)
{
  mergeSideChannels<<<1024, 1024, 0, _proc_stream>>>(thrust::raw_pointer_cast(inputDataStream->polarization0._sideChannelData.a().data()),
          thrust::raw_pointer_cast(inputDataStream->polarization1._sideChannelData.a().data()), inputDataStream->polarization1._sideChannelData.a().size());

  gated_fft(inputDataStream->polarization0, outputDataStream->G0._noOfBitSets.a(), outputDataStream->G1._noOfBitSets.a());
  gated_fft(inputDataStream->polarization1, outputDataStream->G0._noOfBitSets.a(), outputDataStream->G1._noOfBitSets.a());

  for(int output_block_number = 0; output_block_number < outputDataStream->G0._noOfBitSets.size(); output_block_number++)
  {
      size_t input_offset = output_block_number * inputDataStream->polarization0._channelised_voltage_G0.size() / outputDataStream->G0._noOfBitSets.size();
      size_t output_offset = output_block_number * outputDataStream->G0.I.a().size() / outputDataStream->G0._noOfBitSets.size();
      BOOST_LOG_TRIVIAL(debug) << "Accumulating data for output block " << output_block_number << " with input offset " << input_offset << " and output_offset " << output_offset;
      stokes_accumulate<<<1024, 1024, 0, _proc_stream>>>(
              thrust::raw_pointer_cast(inputDataStream->polarization0._channelised_voltage_G0.data() + input_offset),
              thrust::raw_pointer_cast(inputDataStream->polarization1._channelised_voltage_G0.data() + input_offset),
              thrust::raw_pointer_cast(outputDataStream->G0.I.a().data() + output_offset),
              thrust::raw_pointer_cast(outputDataStream->G0.Q.a().data() + output_offset),
              thrust::raw_pointer_cast(outputDataStream->G0.U.a().data() + output_offset),
              thrust::raw_pointer_cast(outputDataStream->G0.V.a().data() + output_offset),
              _nchans, _naccumulate / _nBlocks
              );

      stokes_accumulate<<<1024, 1024, 0, _proc_stream>>>(
              thrust::raw_pointer_cast(inputDataStream->polarization0._channelised_voltage_G1.data() + input_offset),
              thrust::raw_pointer_cast(inputDataStream->polarization1._channelised_voltage_G1.data() + input_offset),
              thrust::raw_pointer_cast(outputDataStream->G1.I.a().data() + output_offset),
              thrust::raw_pointer_cast(outputDataStream->G1.Q.a().data() + output_offset),
              thrust::raw_pointer_cast(outputDataStream->G1.U.a().data() + output_offset),
              thrust::raw_pointer_cast(outputDataStream->G1.V.a().data() + output_offset),
              _nchans, _naccumulate / _nBlocks
              );
  }
  //thrust::fill(thrust::cuda::par.on(_proc_stream),outputDataStream->G0.I.a().begin(), outputDataStream->G0.I.a().end(), _call_count);
  //thrust::fill(thrust::cuda::par.on(_proc_stream),outputDataStream->G0.Q.a().begin(), outputDataStream->G0.Q.a().end(), _call_count);
  //thrust::fill(thrust::cuda::par.on(_proc_stream),outputDataStream->G0.U.a().begin(), outputDataStream->G0.U.a().end(), _call_count);
  //thrust::fill(thrust::cuda::par.on(_proc_stream),outputDataStream->G0.V.a().begin(), outputDataStream->G0.V.a().end(), _call_count);


 // thrust::fill(thrust::cuda::par.on(_proc_stream),outputDataStream->G1.I.a().begin(), outputDataStream->G1.I.a().end(), _call_count);
  //thrust::fill(thrust::cuda::par.on(_proc_stream),outputDataStream->G1.Q.a().begin(), outputDataStream->G1.Q.a().end(), _call_count);
  //thrust::fill(thrust::cuda::par.on(_proc_stream),outputDataStream->G1.U.a().begin(), outputDataStream->G1.U.a().end(), _call_count);
  //thrust::fill(thrust::cuda::par.on(_proc_stream),outputDataStream->G1.V.a().begin(), outputDataStream->G1.V.a().end(), _call_count);

  //  cudaDeviceSynchronize();
}

} // edd
} // effelsberg
} // psrdada_cpp

