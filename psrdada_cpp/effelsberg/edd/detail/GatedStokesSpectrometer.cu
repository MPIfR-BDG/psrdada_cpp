#include "psrdada_cpp/effelsberg/edd/GatedStokesSpectrometer.cuh"
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

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

// Reduce thread local vatiable v in shared array x, so that x[0]
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


// If one of the side channel items is lsot, then both are considered as lost
// here
__global__ void mergeSideChannels(uint64_t* __restrict__ A, uint64_t* __restrict__ B, size_t N)
{
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; (i < N);
       i += blockDim.x * gridDim.x)
  {
    uint64_t v = A[i] || B[i];
    A[i] = v;
    B[i] = v;
  }
}


__global__ void gating(float* __restrict__ G0,
        float* __restrict__ G1,
        const uint64_t* __restrict__ sideChannelData,
        size_t N, size_t heapSize, size_t bitpos,
        size_t noOfSideChannels, size_t selectedSideChannel,
        const float*  __restrict__ _baseLineG0,
        const float*  __restrict__ _baseLineG1,
        float* __restrict__ baseLineNG0,
        float* __restrict__ baseLineNG1,
        uint64_cu* stats_G0, uint64_cu* stats_G1) {
  // statistics values for samopels to G0, G1
  uint32_t _G0stats = 0;
  uint32_t _G1stats = 0;

  const float baseLineG0 = _baseLineG0[0];
  const float baseLineG1 = _baseLineG1[0];

  float baselineUpdateG0 = 0;
  float baselineUpdateG1 = 0;

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; (i < N);
       i += blockDim.x * gridDim.x) {
    const float v = G0[i];

    const uint64_t sideChannelItem = sideChannelData[((i / heapSize) * (noOfSideChannels)) +
                        selectedSideChannel];

    const unsigned int bit_set = TEST_BIT(sideChannelItem, bitpos);
    const unsigned int heap_lost = TEST_BIT(sideChannelItem, 63);
    G1[i] = (v - baseLineG1) * bit_set * (!heap_lost) + baseLineG1;
    G0[i] = (v - baseLineG0) * (!bit_set) *(!heap_lost) + baseLineG0;

    _G0stats += (!bit_set) *(!heap_lost);
    _G1stats += bit_set * (!heap_lost);

    baselineUpdateG1 += v * bit_set * (!heap_lost);
    baselineUpdateG0 += v * (!bit_set) *(!heap_lost);
  }

  __shared__ uint32_t x[1024];

  // Reduce G0, G1
  sum_reduce<uint32_t>(x, _G0stats);
  if(threadIdx.x == 0) {
    atomicAdd(stats_G0,  (uint64_cu) x[threadIdx.x]);
  }
  __syncthreads();

  sum_reduce<uint32_t>(x, _G1stats);
  if(threadIdx.x == 0) {
    atomicAdd(stats_G1,  (uint64_cu) x[threadIdx.x]);
  }
  __syncthreads();

  //reuse shared array
  float *y = (float*) x;
  //update the baseline array
  sum_reduce<float>(y, baselineUpdateG0);
  if(threadIdx.x == 0) {
    atomicAdd(baseLineNG0, y[threadIdx.x]);
  }
  __syncthreads();

  sum_reduce<float>(y, baselineUpdateG1);
  if(threadIdx.x == 0) {
    atomicAdd(baseLineNG1, y[threadIdx.x]);
  }
  __syncthreads();
}



// Updates the baselines of the gates for the polarization set for the next
// block
// only few output blocks per input block thus execution on only one thread.
// Important is that the execution is async on the GPU.
__global__ void update_baselines(float*  __restrict__ baseLineG0,
        float*  __restrict__ baseLineG1,
        float* __restrict__ baseLineNG0,
        float* __restrict__ baseLineNG1,
        uint64_cu* stats_G0, uint64_cu* stats_G1,
        size_t N)
{
    size_t NG0 = 0;
    size_t NG1 = 0;

    for (size_t i =0; i < N; i++)
    {
       NG0 += stats_G0[i];
       NG1 += stats_G1[i];
    }

    baseLineG0[0] = baseLineNG0[0] / NG0;
    baseLineG1[0] = baseLineNG1[0] / NG1;
    baseLineNG0[0] = 0;
    baseLineNG1[0] = 0;
}





template <class HandlerType, class InputType>
GatedStokesSpectrometer<HandlerType>::GatedStokesSpectrometer(
    const DadaBufferLayout &dadaBufferLayout,
    std::size_t selectedSideChannel, std::size_t selectedBit, std::size_t fft_length, std::size_t naccumulate,
    std::size_t nbits, float input_level, float output_level,
    HandlerType &handler) : _dadaBufferLayout(dadaBufferLayout),
      _selectedSideChannel(selectedSideChannel), _selectedBit(selectedBit),
      _fft_length(fft_length),
      _naccumulate(naccumulate), _nbits(nbits), _handler(handler), _fft_plan(0),
      _call_count(0), _nsamps_per_heap(4096), _processing_efficiency(0.){

  // Sanity checks
  assert(((_nbits == 12) || (_nbits == 8)));
  assert(_naccumulate > 0);

  // check for any device errors
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());

  BOOST_LOG_TRIVIAL(info)
      << "Creating new GatedStokesSpectrometer instance with parameters: \n"
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
  polarization0._raw_voltage.resize(_dadaBufferLayout.sizeOfData() / sizeof(uint64_t));
  polarization1._raw_voltage.resize(_dadaBufferLayout.sizeOfData() / sizeof(uint64_t));
  polarization0._sideChannelData.resize(_dadaBufferLayout.getNSideChannels() * _dadaBufferLayout.getNHeaps());
  polarization1._sideChannelData.resize(_dadaBufferLayout.getNSideChannels() * _dadaBufferLayout.getNHeaps());
  BOOST_LOG_TRIVIAL(debug) << "  Input voltages size (in 64-bit words): "
                           << polarization0._raw_voltage.size();
  _unpacked_voltage_G0.resize(_nsamps_per_buffer);
  _unpacked_voltage_G1.resize(_nsamps_per_buffer);

  BOOST_LOG_TRIVIAL(debug) << "  Unpacked voltages size (in samples): "
                           << _unpacked_voltage_G0.size();
  polarization0.resize(_nchans * batch);
  polarization1.resize(_nchans * batch);
  BOOST_LOG_TRIVIAL(debug) << "  Channelised voltages size: "
                           << polarization0._channelised_voltage_G0.size();

   stokes_G0.resize(_nchans, batch / (_naccumulate / nBlocks));
   stokes_G1.resize(_nchans, batch / (_naccumulate / nBlocks));

  // on the host full output is stored together with sci data in one buffer
  _host_power_db.resize( 8 * (_nchans * sizeof(IntegratedPowerType) + sizeof(size_t)) * batch / (_naccumulate / nBlocks));

  CUDA_ERROR_CHECK(cudaStreamCreate(&_h2d_stream));
  CUDA_ERROR_CHECK(cudaStreamCreate(&_proc_stream));
  CUDA_ERROR_CHECK(cudaStreamCreate(&_d2h_stream));
  CUFFT_ERROR_CHECK(cufftSetStream(_fft_plan, _proc_stream));

  _unpacker.reset(new Unpacker(_proc_stream));
} // constructor



template <class HandlerType, class InputType>
GatedStokesSpectrometer<HandlerType>::~GatedStokesSpectrometer() {
  BOOST_LOG_TRIVIAL(debug) << "Destroying GatedStokesSpectrometer";
  if (!_fft_plan)
    cufftDestroy(_fft_plan);
  cudaStreamDestroy(_h2d_stream);
  cudaStreamDestroy(_proc_stream);
  cudaStreamDestroy(_d2h_stream);
}



template <class HandlerType, class InputType>
void GatedStokesSpectrometer<HandlerType>::init(RawBytes &block) {
  BOOST_LOG_TRIVIAL(debug) << "GatedStokesSpectrometer init called";
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



template <class HandlerType, class InputType>
void GatedStokesSpectrometer<HandlerType>::gated_fft(
        PolarizationData &data,
  thrust::device_vector<uint64_cu> &_noOfBitSetsIn_G0,
  thrust::device_vector<uint64_cu> &_noOfBitSetsIn_G1
        )
{
  BOOST_LOG_TRIVIAL(debug) << "Unpacking raw voltages";
  switch (_nbits) {
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
  UnpackedVoltageType *_unpacked_voltage_ptr =
      thrust::raw_pointer_cast(_unpacked_voltage_G0.data());
  ChannelisedVoltageType *_channelised_voltage_ptr =
      thrust::raw_pointer_cast(data._channelised_voltage_G0.data());
  CUFFT_ERROR_CHECK(cufftExecR2C(_fft_plan, (cufftReal *)_unpacked_voltage_ptr,
                                 (cufftComplex *)_channelised_voltage_ptr));

  BOOST_LOG_TRIVIAL(debug) << "Performing FFT 2";
  _unpacked_voltage_ptr = thrust::raw_pointer_cast(_unpacked_voltage_G1.data());
  _channelised_voltage_ptr = thrust::raw_pointer_cast(data._channelised_voltage_G1.data());
  CUFFT_ERROR_CHECK(cufftExecR2C(_fft_plan, (cufftReal *)_unpacked_voltage_ptr,
                                 (cufftComplex *)_channelised_voltage_ptr));

  CUDA_ERROR_CHECK(cudaStreamSynchronize(_proc_stream));
  BOOST_LOG_TRIVIAL(debug) << "Exit processing";
} // process


template <class HandlerType, class InputType>
bool GatedStokesSpectrometer<HandlerType>::operator()(RawBytes &block) {
    ++_call_count;
    BOOST_LOG_TRIVIAL(debug) << "GatedStokesSpectrometer operator() called (count = "
                             << _call_count << ")";
    if (block.used_bytes() != _dadaBufferLayout.getBufferSize()) {
      // Stop on unexpected buffer size
      BOOST_LOG_TRIVIAL(error) << "Unexpected Buffer Size - Got "
                               << block.used_bytes() << " byte, expected "
                               << _dadaBufferLayout.getBufferSize() << " byte)";
      CUDA_ERROR_CHECK(cudaDeviceSynchronize());
      cudaProfilerStop();
      return true;
    }

    // Copy data to device
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_h2d_stream));
    polarization0.swap();
    polarization1.swap();

    BOOST_LOG_TRIVIAL(debug) << "   block.used_bytes() = " <<
        block.used_bytes() << ", dataBlockBytes = " <<
        _dadaBufferLayout.sizeOfData() << "\n";

    // Copy the data with stride to the GPU:
    // CPU: P1P2P1P2P1P2 ...
    // GPU: P1P1P1 ... P2P2P2 ...
    // If this is a bottleneck the gating kernel could sort the layout out
    // during copy
    int heapsize_bytes = _nsamps_per_heap * _nbits / 8;
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


  if (_call_count == 1) {
    return false;
  }

  // process data
  // check if new outblock is started:  _call_count -1 because this is the block number on the device
  bool newBlock = (((_call_count-1) * _nsamps_per_buffer) % _nsamps_per_output_spectra == 0);

  // only if  a newblock is started the output buffer is swapped. Otherwise the
  // new data is added to it
  if (newBlock)
  {
      BOOST_LOG_TRIVIAL(debug) << "Starting new output block.";
      stokes_G0.swap();
      stokes_G1.swap();
      stokes_G0.reset(_proc_stream);
      stokes_G1.reset(_proc_stream);
  }

  mergeSideChannels<<<1024, 1024, 0, _proc_stream>>>(thrust::raw_pointer_cast(polarization0._sideChannelData.a().data()),
          thrust::raw_pointer_cast(polarization1._sideChannelData.a().data()), polarization1._sideChannelData.a().size());

  gated_fft(polarization0, stokes_G0._noOfBitSets.a(), stokes_G1._noOfBitSets.a());
  gated_fft(polarization1, stokes_G0._noOfBitSets.a(), stokes_G1._noOfBitSets.a());

  stokes_accumulate<<<1024, 1024, 0, _proc_stream>>>(
          thrust::raw_pointer_cast(polarization0._channelised_voltage_G0.data()),
          thrust::raw_pointer_cast(polarization1._channelised_voltage_G0.data()),
          thrust::raw_pointer_cast(stokes_G0.I.a().data()),
          thrust::raw_pointer_cast(stokes_G0.Q.a().data()),
          thrust::raw_pointer_cast(stokes_G0.U.a().data()),
          thrust::raw_pointer_cast(stokes_G0.V.a().data()),
          _nchans, _naccumulate
          );

  stokes_accumulate<<<1024, 1024, 0, _proc_stream>>>(
          thrust::raw_pointer_cast(polarization0._channelised_voltage_G1.data()),
          thrust::raw_pointer_cast(polarization1._channelised_voltage_G1.data()),
          thrust::raw_pointer_cast(stokes_G1.I.a().data()),
          thrust::raw_pointer_cast(stokes_G1.Q.a().data()),
          thrust::raw_pointer_cast(stokes_G1.U.a().data()),
          thrust::raw_pointer_cast(stokes_G1.V.a().data()),
          _nchans, _naccumulate
          );


  CUDA_ERROR_CHECK(cudaStreamSynchronize(_proc_stream));

  if ((_call_count == 2) || (!newBlock)) {
    return false;
  }

  // copy data to host if block is finished
  CUDA_ERROR_CHECK(cudaStreamSynchronize(_d2h_stream));
  _host_power_db.swap();
  // OUTPUT MEMORY LAYOUT:
  // I G0, IG1,Q G0, QG1, U G0,UG1,V G0,VG1, 8xSCI, ...

  for (size_t i = 0; i < stokes_G0._noOfBitSets.size(); i++)
  {
    size_t memslicesize = (_nchans * sizeof(IntegratedPowerType));
    size_t memOffset = 8 * i * (memslicesize +  + sizeof(size_t));
    // Copy  II QQ UU VV
    CUDA_ERROR_CHECK(
        cudaMemcpyAsync(static_cast<void *>(_host_power_db.a_ptr() + memOffset) ,
                        static_cast<void *>(stokes_G0.I.b_ptr() + i * memslicesize),
                        _nchans * sizeof(IntegratedPowerType),
                        cudaMemcpyDeviceToHost, _d2h_stream));

    CUDA_ERROR_CHECK(
        cudaMemcpyAsync(static_cast<void *>(_host_power_db.a_ptr() + memOffset + 1 * memslicesize) ,
                        static_cast<void *>(stokes_G1.I.b_ptr() + i * memslicesize),
                        _nchans * sizeof(IntegratedPowerType),
                        cudaMemcpyDeviceToHost, _d2h_stream));

    CUDA_ERROR_CHECK(
        cudaMemcpyAsync(static_cast<void *>(_host_power_db.a_ptr() + memOffset + 2 * memslicesize) ,
                        static_cast<void *>(stokes_G0.Q.b_ptr() + i * memslicesize),
                        _nchans * sizeof(IntegratedPowerType),
                        cudaMemcpyDeviceToHost, _d2h_stream));

    CUDA_ERROR_CHECK(
        cudaMemcpyAsync(static_cast<void *>(_host_power_db.a_ptr() + memOffset + 3 * memslicesize) ,
                        static_cast<void *>(stokes_G1.Q.b_ptr() + i * memslicesize),
                        _nchans * sizeof(IntegratedPowerType),
                        cudaMemcpyDeviceToHost, _d2h_stream));

    CUDA_ERROR_CHECK(
        cudaMemcpyAsync(static_cast<void *>(_host_power_db.a_ptr() + memOffset + 4 * memslicesize) ,
                        static_cast<void *>(stokes_G0.U.b_ptr() + i * memslicesize),
                        _nchans * sizeof(IntegratedPowerType),
                        cudaMemcpyDeviceToHost, _d2h_stream));

    CUDA_ERROR_CHECK(
        cudaMemcpyAsync(static_cast<void *>(_host_power_db.a_ptr() + memOffset + 5 * memslicesize) ,
                        static_cast<void *>(stokes_G1.U.b_ptr() + i * memslicesize),
                        _nchans * sizeof(IntegratedPowerType),
                        cudaMemcpyDeviceToHost, _d2h_stream));

    CUDA_ERROR_CHECK(
        cudaMemcpyAsync(static_cast<void *>(_host_power_db.a_ptr() + memOffset + 6 * memslicesize) ,
                        static_cast<void *>(stokes_G0.V.b_ptr() + i * memslicesize),
                        _nchans * sizeof(IntegratedPowerType),
                        cudaMemcpyDeviceToHost, _d2h_stream));

    CUDA_ERROR_CHECK(
        cudaMemcpyAsync(static_cast<void *>(_host_power_db.a_ptr() + memOffset + 7 * memslicesize) ,
                        static_cast<void *>(stokes_G1.V.b_ptr() + i * memslicesize),
                        _nchans * sizeof(IntegratedPowerType),
                        cudaMemcpyDeviceToHost, _d2h_stream));

    // Copy SCI
    CUDA_ERROR_CHECK(
        cudaMemcpyAsync( static_cast<void *>(_host_power_db.a_ptr() + memOffset + 8 * memslicesize),
          static_cast<void *>(stokes_G0._noOfBitSets.b_ptr() + i ),
            1 * sizeof(size_t),
            cudaMemcpyDeviceToHost, _d2h_stream));
    CUDA_ERROR_CHECK(
        cudaMemcpyAsync( static_cast<void *>(_host_power_db.a_ptr() + memOffset + 8 * memslicesize + 1 * sizeof(size_t)),
          static_cast<void *>(stokes_G1._noOfBitSets.b_ptr() + i ),
            1 * sizeof(size_t),
            cudaMemcpyDeviceToHost, _d2h_stream));
    CUDA_ERROR_CHECK(
        cudaMemcpyAsync( static_cast<void *>(_host_power_db.a_ptr() + memOffset + 8 * memslicesize + 2 * sizeof(size_t)),
          static_cast<void *>(stokes_G0._noOfBitSets.b_ptr() + i ),
            1 * sizeof(size_t),
            cudaMemcpyDeviceToHost, _d2h_stream));
    CUDA_ERROR_CHECK(
        cudaMemcpyAsync( static_cast<void *>(_host_power_db.a_ptr() + memOffset + 8 * memslicesize + 3 * sizeof(size_t)),
          static_cast<void *>(stokes_G1._noOfBitSets.b_ptr() + i ),
            1 * sizeof(size_t),
            cudaMemcpyDeviceToHost, _d2h_stream));
    CUDA_ERROR_CHECK(
        cudaMemcpyAsync( static_cast<void *>(_host_power_db.a_ptr() + memOffset + 8 * memslicesize + 4 * sizeof(size_t)),
          static_cast<void *>(stokes_G0._noOfBitSets.b_ptr() + i ),
            1 * sizeof(size_t),
            cudaMemcpyDeviceToHost, _d2h_stream));
    CUDA_ERROR_CHECK(
        cudaMemcpyAsync( static_cast<void *>(_host_power_db.a_ptr() + memOffset + 8 * memslicesize + 5 * sizeof(size_t)),
          static_cast<void *>(stokes_G1._noOfBitSets.b_ptr() + i ),
            1 * sizeof(size_t),
            cudaMemcpyDeviceToHost, _d2h_stream));
    CUDA_ERROR_CHECK(
        cudaMemcpyAsync( static_cast<void *>(_host_power_db.a_ptr() + memOffset + 8 * memslicesize + 6 * sizeof(size_t)),
          static_cast<void *>(stokes_G0._noOfBitSets.b_ptr() + i ),
            1 * sizeof(size_t),
            cudaMemcpyDeviceToHost, _d2h_stream));
    CUDA_ERROR_CHECK(
        cudaMemcpyAsync( static_cast<void *>(_host_power_db.a_ptr() + memOffset + 8 * memslicesize + 7 * sizeof(size_t)),
          static_cast<void *>(stokes_G1._noOfBitSets.b_ptr() + i ),
            1 * sizeof(size_t),
            cudaMemcpyDeviceToHost, _d2h_stream));

  }

  BOOST_LOG_TRIVIAL(debug) << "Copy Data back to host";

  if (_call_count == 3) {
    return false;
  }

  // calculate off value
  //BOOST_LOG_TRIVIAL(info) << "Buffer block: " << _call_count-3 << " with " << _noOfBitSetsIn_G0.size() << "x2 output heaps:";
  //size_t total_samples_lost = 0;
  //for (size_t i = 0; i < _noOfBitSetsIn_G0.size(); i++)
  //{
  //  size_t memOffset = 2 * i * (_nchans * sizeof(IntegratedPowerType) + sizeof(size_t));

  //  size_t* on_values = reinterpret_cast<size_t*> (_host_power_db.b_ptr() + memOffset + 2 * _nchans * sizeof(IntegratedPowerType));
  //  size_t* off_values = reinterpret_cast<size_t*> (_host_power_db.b_ptr() + memOffset + 2 * _nchans * sizeof(IntegratedPowerType) + sizeof(size_t));

  //  size_t samples_lost = _nsamps_per_output_spectra - (*on_values) - (*off_values);
  //  total_samples_lost += samples_lost;

  //  BOOST_LOG_TRIVIAL(info) << "    Heap " << i << ":\n"
  //    <<"                            Samples with  bit set  : " << *on_values << std::endl
  //    <<"                            Samples without bit set: " << *off_values << std::endl
  //    <<"                            Samples lost           : " << samples_lost << " out of " << _nsamps_per_output_spectra << std::endl;
  //}
  //double efficiency = 1. - double(total_samples_lost) / (_nsamps_per_output_spectra * _noOfBitSetsIn_G0.size());
  //double prev_average = _processing_efficiency / (_call_count- 3 - 1);
  //_processing_efficiency += efficiency;
  //double average = _processing_efficiency / (_call_count-3);
  //BOOST_LOG_TRIVIAL(info) << "Total processing efficiency of this buffer block:" << std::setprecision(6) << efficiency << ". Run average: " << average << " (Trend: " << std::showpos << (average - prev_average) << ")";

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

