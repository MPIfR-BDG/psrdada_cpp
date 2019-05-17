#include "psrdada_cpp/effelsberg/edd/GatedSpectrometer.cuh"
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

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

__global__ void gating(float* __restrict__ G0, float* __restrict__ G1, const uint64_t* __restrict__ sideChannelData,
                       size_t N, size_t heapSize, size_t bitpos,
                       size_t noOfSideChannels, size_t selectedSideChannel, const float* __restrict__ _baseLineN) {
  float baseLine = (*_baseLineN) / N;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; (i < N);
       i += blockDim.x * gridDim.x) {
    const float w = G0[i] - baseLine;
    const uint64_t sideChannelItem =
        sideChannelData[((i / heapSize) * (noOfSideChannels)) +
                        selectedSideChannel]; // Probably not optimal access as
                                              // same data is copied for several
                                              // threads, but maybe efficiently
                                              // handled by cache?

    const int bit_set = TEST_BIT(sideChannelItem, bitpos);
    G1[i] = w * bit_set + baseLine;
    G0[i] = w * (!bit_set) + baseLine;
  }
}


__global__ void countBitSet(const uint64_t *sideChannelData, size_t N, size_t
    bitpos, size_t noOfSideChannels, size_t selectedSideChannel, size_t
    *nBitsSet)
{
  // really not optimized reduction, but here only trivial array sizes.
  // run only in one block!
  __shared__ size_t x[1024];
  size_t ls = 0;

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; (i < N);
       i += blockDim.x * gridDim.x) {
    ls += TEST_BIT(sideChannelData[i * noOfSideChannels + selectedSideChannel], bitpos);
  }
  x[threadIdx.x] = ls;

  __syncthreads();
  for(int s = blockDim.x / 2; s > 0; s = s / 2)
  {
    if (threadIdx.x < s)
      x[threadIdx.x] += x[threadIdx.x + s];
    __syncthreads();
  }

  if(threadIdx.x == 0)
   nBitsSet[0] += x[threadIdx.x];
}


// blocksize for the array sum kernel
#define array_sum_Nthreads 1024

__global__ void array_sum(float *in, size_t N, float *out) {
  extern __shared__ float data[];

  size_t tid = threadIdx.x;

  float ls = 0;

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; (i < N);
       i += blockDim.x * gridDim.x) {
    ls += in[i]; // + in[i + blockDim.x];   // loading two elements increase the used bandwidth by ~10% but requires matching blocksize and size of input array
  }

  data[tid] = ls;
  __syncthreads();

  for (size_t i = blockDim.x / 2; i > 0; i /= 2) {
    if (tid < i) {
      data[tid] += data[tid + i];
    }
    __syncthreads();
  }

  // unroll last warp
  // if (tid < 32)
  //{
  //  warpReduce(data, tid);
  //}

  if (tid == 0) {
    out[blockIdx.x] = data[0];
  }
}


template <class HandlerType, typename IntegratedPowerType>
GatedSpectrometer<HandlerType, IntegratedPowerType>::GatedSpectrometer(
    const DadaBufferLayout &dadaBufferLayout,
    std::size_t selectedSideChannel, std::size_t selectedBit, std::size_t fft_length, std::size_t naccumulate,
    std::size_t nbits, float input_level, float output_level,
    HandlerType &handler) : _dadaBufferLayout(dadaBufferLayout),
      _selectedSideChannel(selectedSideChannel), _selectedBit(selectedBit),
      _fft_length(fft_length),
      _naccumulate(naccumulate), _nbits(nbits), _handler(handler), _fft_plan(0),
      _call_count(0), _nsamps_per_heap(4096) {

  // Sanity checks
  assert(((_nbits == 12) || (_nbits == 8)));
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

   _nsamps_per_buffer = _dadaBufferLayout.sizeOfData() * 8 / nbits;

  _nsamps_per_output_spectra = fft_length * naccumulate;
  int nBlocks;
  if (_nsamps_per_output_spectra <= _nsamps_per_buffer)
  { // one buffer block is used for one or multiple output spectra
    size_t N = _nsamps_per_buffer / _nsamps_per_output_spectra;
    // All data in one block has to be used
    assert(N * _nsamps_per_output_spectra == _nsamps_per_buffer);
    nBlocks = 1;
  }
  else
  { // multiple blocks are integrated intoone output
    size_t N =  _nsamps_per_output_spectra /  _nsamps_per_buffer;
    // All data in multiple blocks has to be used
    assert(N * _nsamps_per_buffer == _nsamps_per_output_spectra);
    nBlocks = N;
  }
  BOOST_LOG_TRIVIAL(debug) << "Integrating  " << _nsamps_per_output_spectra << " samples from " << nBlocks << " into one spectra.";

  _nchans = _fft_length / 2 + 1;
  int batch = _nsamps_per_buffer / _fft_length;
  float dof = 2 * _naccumulate;
  float scale =
      std::pow(input_level * std::sqrt(static_cast<float>(_nchans)), 2);
  float offset = scale * dof;
  float scaling = scale * std::sqrt(2 * dof) / output_level;
  BOOST_LOG_TRIVIAL(debug)
      << "Correction factors for 8-bit conversion: offset = " << offset
      << ", scaling = " << scaling;

  BOOST_LOG_TRIVIAL(debug) << "Generating FFT plan";
  int n[] = {static_cast<int>(_fft_length)};
  CUFFT_ERROR_CHECK(cufftPlanMany(&_fft_plan, 1, n, NULL, 1, _fft_length, NULL,
                                  1, _nchans, CUFFT_R2C, batch));
  cufftSetStream(_fft_plan, _proc_stream);

  BOOST_LOG_TRIVIAL(debug) << "Allocating memory";
  _raw_voltage_db.resize(_dadaBufferLayout.sizeOfData() / sizeof(uint64_t));
  _sideChannelData_db.resize(_dadaBufferLayout.getNSideChannels() * _dadaBufferLayout.getNHeaps());
  BOOST_LOG_TRIVIAL(debug) << "  Input voltages size (in 64-bit words): "
                           << _raw_voltage_db.size();
  _unpacked_voltage_G0.resize(_nsamps_per_buffer);
  _unpacked_voltage_G1.resize(_nsamps_per_buffer);

  _baseLineN.resize(array_sum_Nthreads);
  BOOST_LOG_TRIVIAL(debug) << "  Unpacked voltages size (in samples): "
                           << _unpacked_voltage_G0.size();
  _channelised_voltage.resize(_nchans * batch);
  BOOST_LOG_TRIVIAL(debug) << "  Channelised voltages size: "
                           << _channelised_voltage.size();
  _power_db.resize(_nchans * batch / (_naccumulate / nBlocks) * 2);  // hold on and off spectra to simplify output
  thrust::fill(_power_db.a().begin(), _power_db.a().end(), 0.);
  thrust::fill(_power_db.b().begin(), _power_db.b().end(), 0.);
  BOOST_LOG_TRIVIAL(debug) << "  Powers size: " << _power_db.size() / 2;

  _noOfBitSetsInSideChannel.resize( batch / (_naccumulate / nBlocks));
  thrust::fill(_noOfBitSetsInSideChannel.a().begin(), _noOfBitSetsInSideChannel.a().end(), 0L);
  thrust::fill(_noOfBitSetsInSideChannel.b().begin(), _noOfBitSetsInSideChannel.b().end(), 0L);
  BOOST_LOG_TRIVIAL(debug) << "  Bit set counrer size: " << _noOfBitSetsInSideChannel.size();

  // on the host both power are stored in the same data buffer together with
  // the number of bit sets
  _host_power_db.resize( _power_db.size() * sizeof(IntegratedPowerType) + 2 * sizeof(size_t) * _noOfBitSetsInSideChannel.size());

  CUDA_ERROR_CHECK(cudaStreamCreate(&_h2d_stream));
  CUDA_ERROR_CHECK(cudaStreamCreate(&_proc_stream));
  CUDA_ERROR_CHECK(cudaStreamCreate(&_d2h_stream));
  CUFFT_ERROR_CHECK(cufftSetStream(_fft_plan, _proc_stream));

  _unpacker.reset(new Unpacker(_proc_stream));
  _detector.reset(new DetectorAccumulator<IntegratedPowerType>(_nchans, _naccumulate / nBlocks, scaling,
                                          offset, _proc_stream));
} // constructor


template <class HandlerType, typename IntegratedPowerType>
GatedSpectrometer<HandlerType, IntegratedPowerType>::~GatedSpectrometer() {
  BOOST_LOG_TRIVIAL(debug) << "Destroying GatedSpectrometer";
  if (!_fft_plan)
    cufftDestroy(_fft_plan);
  cudaStreamDestroy(_h2d_stream);
  cudaStreamDestroy(_proc_stream);
  cudaStreamDestroy(_d2h_stream);
}


template <class HandlerType, typename IntegratedPowerType>
void GatedSpectrometer<HandlerType, IntegratedPowerType>::init(RawBytes &block) {
  BOOST_LOG_TRIVIAL(debug) << "GatedSpectrometer init called";
  std::stringstream headerInfo;
  headerInfo << "\n"
      << "# Gated spectrometer parameters: \n"
      << "fft_length               " << _fft_length << "\n"
      << "nchannels                " << _fft_length << "\n"
      << "naccumulate              " << _naccumulate << "\n"
      << "selected_side_channel    " << _selectedSideChannel << "\n"
      << "selected_bit             " << _selectedBit << "\n"
      << "output_bit_depth         " << sizeof(IntegratedPowerType) * 8;

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


template <class HandlerType, typename IntegratedPowerType>
void GatedSpectrometer<HandlerType, IntegratedPowerType>::process(
    thrust::device_vector<RawVoltageType> const &digitiser_raw,
    thrust::device_vector<uint64_t> const &sideChannelData,
    thrust::device_vector<IntegratedPowerType> &detected, thrust::device_vector<size_t> &noOfBitSet) {
  BOOST_LOG_TRIVIAL(debug) << "Unpacking raw voltages";
  switch (_nbits) {
  case 8:
    _unpacker->unpack<8>(digitiser_raw, _unpacked_voltage_G0);
    break;
  case 12:
    _unpacker->unpack<12>(digitiser_raw, _unpacked_voltage_G0);
    break;
  default:
    throw std::runtime_error("Unsupported number of bits");
  }
  BOOST_LOG_TRIVIAL(debug) << "Calculate baseline";
  psrdada_cpp::effelsberg::edd::array_sum<<<64, array_sum_Nthreads, array_sum_Nthreads * sizeof(float), _proc_stream>>>(thrust::raw_pointer_cast(_unpacked_voltage_G0.data()), _unpacked_voltage_G0.size(), thrust::raw_pointer_cast(_baseLineN.data()));
  psrdada_cpp::effelsberg::edd::array_sum<<<1, array_sum_Nthreads, array_sum_Nthreads * sizeof(float), _proc_stream>>>(thrust::raw_pointer_cast(_baseLineN.data()), _baseLineN.size(), thrust::raw_pointer_cast(_baseLineN.data()));

  BOOST_LOG_TRIVIAL(debug) << "Perform gating";
  gating<<<1024, 1024, 0, _proc_stream>>>(
      thrust::raw_pointer_cast(_unpacked_voltage_G0.data()),
      thrust::raw_pointer_cast(_unpacked_voltage_G1.data()),
      thrust::raw_pointer_cast(sideChannelData.data()),
      _unpacked_voltage_G0.size(), _dadaBufferLayout.getHeapSize(), _selectedBit, _dadaBufferLayout.getNSideChannels(),
      _selectedSideChannel, thrust::raw_pointer_cast(_baseLineN.data()));

  for (size_t i = 0; i < _noOfBitSetsInSideChannel.size(); i++)
  { // ToDo: Should be in one kernel call
    countBitSet<<<1, 1024, 0, _proc_stream>>>(thrust::raw_pointer_cast(sideChannelData.data() + i * sideChannelData.size() / _noOfBitSetsInSideChannel.size() ),
          sideChannelData.size() / _noOfBitSetsInSideChannel.size(), _selectedBit,
          _dadaBufferLayout.getNSideChannels(), _selectedBit,
          thrust::raw_pointer_cast(noOfBitSet.data() + i));
    
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_proc_stream));
  }

  BOOST_LOG_TRIVIAL(debug) << "Performing FFT 1";
  UnpackedVoltageType *_unpacked_voltage_ptr =
      thrust::raw_pointer_cast(_unpacked_voltage_G0.data());
  ChannelisedVoltageType *_channelised_voltage_ptr =
      thrust::raw_pointer_cast(_channelised_voltage.data());
  CUFFT_ERROR_CHECK(cufftExecR2C(_fft_plan, (cufftReal *)_unpacked_voltage_ptr,
                                 (cufftComplex *)_channelised_voltage_ptr));
  _detector->detect(_channelised_voltage, detected, 2, 0);

  BOOST_LOG_TRIVIAL(debug) << "Performing FFT 2";
  _unpacked_voltage_ptr = thrust::raw_pointer_cast(_unpacked_voltage_G1.data());
  CUFFT_ERROR_CHECK(cufftExecR2C(_fft_plan, (cufftReal *)_unpacked_voltage_ptr,
                                 (cufftComplex *)_channelised_voltage_ptr));

  _detector->detect(_channelised_voltage, detected, 2, 1);
  CUDA_ERROR_CHECK(cudaStreamSynchronize(_proc_stream));
  BOOST_LOG_TRIVIAL(debug) << "Exit processing";
} // process


template <class HandlerType, typename IntegratedPowerType>
bool GatedSpectrometer<HandlerType, IntegratedPowerType>::operator()(RawBytes &block) {
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
  _raw_voltage_db.swap();
  _sideChannelData_db.swap();

  BOOST_LOG_TRIVIAL(debug) << "   block.used_bytes() = " << block.used_bytes()
                           << ", dataBlockBytes = " << _dadaBufferLayout.sizeOfData() << "\n";

  CUDA_ERROR_CHECK(cudaMemcpyAsync(static_cast<void *>(_raw_voltage_db.a_ptr()),
                                   static_cast<void *>(block.ptr()),
                                   _dadaBufferLayout.sizeOfData() , cudaMemcpyHostToDevice,
                                   _h2d_stream));
  CUDA_ERROR_CHECK(cudaMemcpyAsync(
      static_cast<void *>(_sideChannelData_db.a_ptr()),
      static_cast<void *>(block.ptr() + _dadaBufferLayout.sizeOfData() + _dadaBufferLayout.sizeOfGap()),
      _dadaBufferLayout.sizeOfSideChannelData(), cudaMemcpyHostToDevice, _h2d_stream));
  BOOST_LOG_TRIVIAL(debug) << "First side channel item: 0x" <<   std::setw(12) << std::setfill('0') << std::hex <<  (reinterpret_cast<uint64_t*>(block.ptr() + _dadaBufferLayout.sizeOfData() + _dadaBufferLayout.sizeOfGap()))[0] << std::dec;


  if (_call_count == 1) {
    return false;
  }
  // process data

  // only if  a newblock is started the output buffer is swapped. Otherwise the
  // new data is added to it
  bool newBlock = false;
  if (((_call_count-1) * _nsamps_per_buffer) % _nsamps_per_output_spectra == 0) // _call_count -1 because this is the block number on the device
  {
    BOOST_LOG_TRIVIAL(debug) << "Starting new output block.";
    newBlock = true;
    _power_db.swap();
    _noOfBitSetsInSideChannel.swap();
    // move to specific stream!
    thrust::fill(thrust::cuda::par.on(_proc_stream),_power_db.a().begin(), _power_db.a().end(), 0.);
    thrust::fill(thrust::cuda::par.on(_proc_stream), _noOfBitSetsInSideChannel.a().begin(), _noOfBitSetsInSideChannel.a().end(), 0L);
  }

  process(_raw_voltage_db.b(), _sideChannelData_db.b(), _power_db.a(), _noOfBitSetsInSideChannel.a());
  CUDA_ERROR_CHECK(cudaStreamSynchronize(_proc_stream));

  if ((_call_count == 2) || (!newBlock)) {
    return false;
  }

  // copy data to host if block is finished
  CUDA_ERROR_CHECK(cudaStreamSynchronize(_d2h_stream));
  _host_power_db.swap();

  for (size_t i = 0; i < _noOfBitSetsInSideChannel.size(); i++)
  {
    size_t memOffset = 2 * i * (_nchans * sizeof(IntegratedPowerType) + sizeof(size_t));
    // copy 2x channel data
    CUDA_ERROR_CHECK(
        cudaMemcpyAsync(static_cast<void *>(_host_power_db.a_ptr() + memOffset) ,
                        static_cast<void *>(_power_db.b_ptr() + 2 * i * _nchans),
                        2 * _nchans * sizeof(IntegratedPowerType),
                        cudaMemcpyDeviceToHost, _d2h_stream));
    // copy noOf bit set data
    CUDA_ERROR_CHECK(
        cudaMemcpyAsync( static_cast<void *>(_host_power_db.a_ptr() + memOffset + 2 * _nchans * sizeof(IntegratedPowerType)),
          static_cast<void *>(_noOfBitSetsInSideChannel.b_ptr() + i ),
            1 * sizeof(size_t),
            cudaMemcpyDeviceToHost, _d2h_stream));
    BOOST_LOG_TRIVIAL(info) << " TOBR NOF BITS SET: " << _noOfBitSetsInSideChannel.b()[i]; 
  }

  BOOST_LOG_TRIVIAL(debug) << "Copy Data back to host";

  if (_call_count == 3) {
    return false;
  }

  // calculate off value
  BOOST_LOG_TRIVIAL(info) << "Buffer block: " << _call_count << " with " << _noOfBitSetsInSideChannel.size() << " output heaps:";
  for (size_t i = 0; i < _noOfBitSetsInSideChannel.size(); i++)
  {
    size_t memOffset = 2 * i * (_nchans * sizeof(IntegratedPowerType) + sizeof(size_t));

    size_t* on_values = reinterpret_cast<size_t*> (_host_power_db.b_ptr() + memOffset + 2 * _nchans * sizeof(IntegratedPowerType));
    *on_values *= _nsamps_per_heap;
    size_t* off_values = reinterpret_cast<size_t*> (_host_power_db.b_ptr() + memOffset + 2 * _nchans * sizeof(IntegratedPowerType) + sizeof(size_t));
    *off_values =  _nsamps_per_output_spectra - (*on_values);

    BOOST_LOG_TRIVIAL(info) << "    " << i << ": No of samples wo/w. bit set in side channel: " << *on_values << " / " << *off_values << std::endl;
  }

  // Wrap in a RawBytes object here;
  RawBytes bytes(reinterpret_cast<char *>(_host_power_db.b_ptr()),
                 _host_power_db.size(),
                 _host_power_db.size());
  BOOST_LOG_TRIVIAL(debug) << "Calling handler";
  // The handler can't do anything asynchronously without a copy here
  // as it would be unsafe (given that it does not own the memory it
  // is being passed).

  _handler(bytes);
  return false; //
} // operator ()

} // edd
} // effelsberg
} // psrdada_cpp

