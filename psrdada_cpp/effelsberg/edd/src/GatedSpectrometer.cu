#include "psrdada_cpp/effelsberg/edd/GatedSpectrometer.cuh"


namespace psrdada_cpp {
namespace effelsberg {
namespace edd {


__global__ void mergeSideChannels(uint64_t* __restrict__ A, uint64_t*
        __restrict__ B, size_t N)
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



/**
 * @brief calculate stokes IQUV from two complex valuies for each polarization
 */
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


void PolarizationData::resize(size_t rawVolttageBufferBytes, size_t nsidechannelitems, size_t channelized_samples)
{
    _raw_voltage.resize(rawVolttageBufferBytes / sizeof(uint64_t));
    BOOST_LOG_TRIVIAL(debug) << "  Input voltages size (in 64-bit words): " << _raw_voltage.size();

    _baseLineG0.resize(1);
    _baseLineG0_update.resize(1);
    _baseLineG1.resize(1);
    _baseLineG1_update.resize(1);
    _channelised_voltage_G0.resize(channelized_samples);
    _channelised_voltage_G1.resize(channelized_samples);
    _sideChannelData.resize(nsidechannelitems);
    BOOST_LOG_TRIVIAL(debug) << "  Channelised voltages size: " << _channelised_voltage_G0.size();
}


SinglePolarizationInput::SinglePolarizationInput(size_t fft_length, size_t nbits, const DadaBufferLayout
        &dadaBufferLayout) : PolarizationData(nbits), _fft_length(fft_length), _dadaBufferLayout(dadaBufferLayout)
{

  size_t nsamps_per_buffer = _dadaBufferLayout.sizeOfData() * 8 / nbits;
  size_t _batch = nsamps_per_buffer / _fft_length;

    resize(_dadaBufferLayout.sizeOfData(), _dadaBufferLayout.getNSideChannels() * _dadaBufferLayout.getNHeaps(), (_fft_length / 2 + 1) * _batch);
};


size_t SinglePolarizationInput::getSamplesPerInputPolarization()
{
    return _dadaBufferLayout.sizeOfData() * 8 / _nbits;
}


void PolarizationData::swap()
{
    _raw_voltage.swap();
    _sideChannelData.swap();
}


void SinglePolarizationInput::getFromBlock(RawBytes &block, cudaStream_t &_h2d_stream)
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


DualPolarizationInput::DualPolarizationInput(size_t fft_length, size_t nbits, const DadaBufferLayout
        &dadaBufferLayout) : _fft_length(fft_length),
    polarization0(nbits),
    polarization1(nbits),
    _dadaBufferLayout(dadaBufferLayout)
{

  size_t nsamps_per_buffer = _dadaBufferLayout.sizeOfData() * 8 / nbits;
  size_t _batch = nsamps_per_buffer / _fft_length / 2;

    polarization0.resize(_dadaBufferLayout.sizeOfData() / 2, _dadaBufferLayout.getNSideChannels() * _dadaBufferLayout.getNHeaps() / 2, (_fft_length / 2 + 1) * _batch);
    polarization1.resize(_dadaBufferLayout.sizeOfData() / 2, _dadaBufferLayout.getNSideChannels() * _dadaBufferLayout.getNHeaps() / 2, (_fft_length / 2 + 1) * _batch);
};


void DualPolarizationInput::swap()
{
    polarization0.swap();
    polarization1.swap();
}


size_t DualPolarizationInput::getSamplesPerInputPolarization()
{
    return _dadaBufferLayout.sizeOfData() * 8 / polarization0._nbits / 2;
}


void DualPolarizationInput::getFromBlock(RawBytes &block, cudaStream_t &_h2d_stream)
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
    static_cast<void *>(block.ptr() + heapsize_bytes),
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



PowerSpectrumOutput::PowerSpectrumOutput(size_t size, size_t blocks)
{
    BOOST_LOG_TRIVIAL(debug) << "Setting size of power spectrum output size = " << size << ", blocks =  " << blocks;
   data.resize(size * blocks);
   _noOfBitSets.resize(blocks);
}


void PowerSpectrumOutput::swap(cudaStream_t &_proc_stream)
{
    data.swap();
    _noOfBitSets.swap();
    thrust::fill(thrust::cuda::par.on(_proc_stream), data.a().begin(), data.a().end(), 0.);
    thrust::fill(thrust::cuda::par.on(_proc_stream), _noOfBitSets.a().begin(), _noOfBitSets.a().end(), 0L);
}


GatedPowerSpectrumOutput::GatedPowerSpectrumOutput(size_t nchans, size_t
        blocks) : OutputDataStream(nchans, blocks), G0(nchans, blocks),
G1(nchans, blocks)
{
  // on the host both power are stored in the same data buffer together with
  // the number of bit sets
  _host_power.resize( 2 * ( _nchans * sizeof(IntegratedPowerType) + sizeof(size_t) ) * G0._noOfBitSets.size());
}


/// Swap output buffers
void GatedPowerSpectrumOutput::swap(cudaStream_t &_proc_stream)
{
    G0.swap(_proc_stream);
    G1.swap(_proc_stream);
    _host_power.swap();
}


void GatedPowerSpectrumOutput::data2Host(cudaStream_t &_d2h_stream)
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
                        static_cast<void *>(G0.data.b_ptr() + i * _nchans),
                        _nchans * sizeof(IntegratedPowerType),
                        cudaMemcpyDeviceToHost, _d2h_stream));

    CUDA_ERROR_CHECK(
        cudaMemcpyAsync(static_cast<void *>(_host_power.a_ptr() + memOffset + 1 * memslicesize) ,
                        static_cast<void *>(G1.data.b_ptr() + i * _nchans),
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


void FullStokesOutput::swap(cudaStream_t &_proc_stream)
{
    I.swap();
    Q.swap();
    U.swap();
    V.swap();
    _noOfBitSets.swap();
    thrust::fill(thrust::cuda::par.on(_proc_stream), I.a().begin(), I.a().end(), 0.);
    thrust::fill(thrust::cuda::par.on(_proc_stream), Q.a().begin(), Q.a().end(), 0.);
    thrust::fill(thrust::cuda::par.on(_proc_stream), U.a().begin(), U.a().end(), 0.);
    thrust::fill(thrust::cuda::par.on(_proc_stream), V.a().begin(), V.a().end(), 0.);
    thrust::fill(thrust::cuda::par.on(_proc_stream), _noOfBitSets.a().begin(), _noOfBitSets.a().end(), 0L);
}


FullStokesOutput::FullStokesOutput(size_t size, size_t blocks)
{
    I.resize(size * blocks);
    Q.resize(size * blocks);
    U.resize(size * blocks);
    V.resize(size * blocks);
    _noOfBitSets.resize(blocks);
}



GatedFullStokesOutput::GatedFullStokesOutput(size_t nchans, size_t blocks): OutputDataStream(nchans, blocks), G0(nchans, blocks),
G1(nchans, blocks)
{
    BOOST_LOG_TRIVIAL(debug) << "Output with " << _blocks << " blocks a " << _nchans << " channels";
    _host_power.resize( 8 * ( _nchans * sizeof(IntegratedPowerType) + sizeof(size_t) ) * _blocks);
    BOOST_LOG_TRIVIAL(debug) << "Allocated " << _host_power.size() << " bytes.";
};


void GatedFullStokesOutput::swap(cudaStream_t &_proc_stream)
{
    G0.swap(_proc_stream);
    G1.swap(_proc_stream);
    _host_power.swap();
}


void GatedFullStokesOutput::data2Host(cudaStream_t &_d2h_stream)
{
for (size_t i = 0; i < G0._noOfBitSets.size(); i++)
{
    size_t memslicesize = (_nchans * sizeof(IntegratedPowerType));
    size_t memOffset = 8 * i * (memslicesize + sizeof(size_t));
    // Copy  II QQ UU VV
    CUDA_ERROR_CHECK(
        cudaMemcpyAsync(static_cast<void *>(_host_power.a_ptr() + memOffset) ,
                        static_cast<void *>(G0.I.b_ptr() + i * _nchans),
                        _nchans * sizeof(IntegratedPowerType),
                        cudaMemcpyDeviceToHost, _d2h_stream));

    CUDA_ERROR_CHECK(
        cudaMemcpyAsync(static_cast<void *>(_host_power.a_ptr() + memOffset + 1 * memslicesize) ,
                        static_cast<void *>(G1.I.b_ptr() + i * _nchans),
                        _nchans * sizeof(IntegratedPowerType),
                        cudaMemcpyDeviceToHost, _d2h_stream));

    CUDA_ERROR_CHECK(
        cudaMemcpyAsync(static_cast<void *>(_host_power.a_ptr() + memOffset + 2 * memslicesize) ,
                        static_cast<void *>(G0.Q.b_ptr() + i * _nchans),
                        _nchans * sizeof(IntegratedPowerType),
                        cudaMemcpyDeviceToHost, _d2h_stream));

    CUDA_ERROR_CHECK(
        cudaMemcpyAsync(static_cast<void *>(_host_power.a_ptr() + memOffset + 3 * memslicesize) ,
                        static_cast<void *>(G1.Q.b_ptr() + i * _nchans),
                        _nchans * sizeof(IntegratedPowerType),
                        cudaMemcpyDeviceToHost, _d2h_stream));

    CUDA_ERROR_CHECK(
        cudaMemcpyAsync(static_cast<void *>(_host_power.a_ptr() + memOffset + 4 * memslicesize) ,
                        static_cast<void *>(G0.U.b_ptr() + i * _nchans),
                        _nchans * sizeof(IntegratedPowerType),
                        cudaMemcpyDeviceToHost, _d2h_stream));

    CUDA_ERROR_CHECK(
        cudaMemcpyAsync(static_cast<void *>(_host_power.a_ptr() + memOffset + 5 * memslicesize) ,
                        static_cast<void *>(G1.U.b_ptr() + i * _nchans),
                        _nchans * sizeof(IntegratedPowerType),
                        cudaMemcpyDeviceToHost, _d2h_stream));

    CUDA_ERROR_CHECK(
        cudaMemcpyAsync(static_cast<void *>(_host_power.a_ptr() + memOffset + 6 * memslicesize) ,
                        static_cast<void *>(G0.V.b_ptr() + i * _nchans),
                        _nchans * sizeof(IntegratedPowerType),
                        cudaMemcpyDeviceToHost, _d2h_stream));

    CUDA_ERROR_CHECK(
        cudaMemcpyAsync(static_cast<void *>(_host_power.a_ptr() + memOffset + 7 * memslicesize) ,
                        static_cast<void *>(G1.V.b_ptr() + i * _nchans),
                        _nchans * sizeof(IntegratedPowerType),
                        cudaMemcpyDeviceToHost, _d2h_stream));

    // Copy SCI
    CUDA_ERROR_CHECK(
        cudaMemcpyAsync( static_cast<void *>(_host_power.a_ptr() + memOffset + 8 * memslicesize),
          static_cast<void *>(G0._noOfBitSets.b_ptr() + i ),
            1 * sizeof(size_t),
            cudaMemcpyDeviceToHost, _d2h_stream));
    CUDA_ERROR_CHECK(
        cudaMemcpyAsync( static_cast<void *>(_host_power.a_ptr() + memOffset + 8 * memslicesize + 1 * sizeof(size_t)),
          static_cast<void *>(G1._noOfBitSets.b_ptr() + i ),
            1 * sizeof(size_t),
            cudaMemcpyDeviceToHost, _d2h_stream));
    CUDA_ERROR_CHECK(
        cudaMemcpyAsync( static_cast<void *>(_host_power.a_ptr() + memOffset + 8 * memslicesize + 2 * sizeof(size_t)),
          static_cast<void *>(G0._noOfBitSets.b_ptr() + i ),
            1 * sizeof(size_t),
            cudaMemcpyDeviceToHost, _d2h_stream));
    CUDA_ERROR_CHECK(
        cudaMemcpyAsync( static_cast<void *>(_host_power.a_ptr() + memOffset + 8 * memslicesize + 3 * sizeof(size_t)),
          static_cast<void *>(G1._noOfBitSets.b_ptr() + i ),
            1 * sizeof(size_t),
            cudaMemcpyDeviceToHost, _d2h_stream));
    CUDA_ERROR_CHECK(
        cudaMemcpyAsync( static_cast<void *>(_host_power.a_ptr() + memOffset + 8 * memslicesize + 4 * sizeof(size_t)),
          static_cast<void *>(G0._noOfBitSets.b_ptr() + i ),
            1 * sizeof(size_t),
            cudaMemcpyDeviceToHost, _d2h_stream));
    CUDA_ERROR_CHECK(
        cudaMemcpyAsync( static_cast<void *>(_host_power.a_ptr() + memOffset + 8 * memslicesize + 5 * sizeof(size_t)),
          static_cast<void *>(G1._noOfBitSets.b_ptr() + i ),
            1 * sizeof(size_t),
            cudaMemcpyDeviceToHost, _d2h_stream));
    CUDA_ERROR_CHECK(
        cudaMemcpyAsync( static_cast<void *>(_host_power.a_ptr() + memOffset + 8 * memslicesize + 6 * sizeof(size_t)),
          static_cast<void *>(G0._noOfBitSets.b_ptr() + i ),
            1 * sizeof(size_t),
            cudaMemcpyDeviceToHost, _d2h_stream));
    CUDA_ERROR_CHECK(
        cudaMemcpyAsync( static_cast<void *>(_host_power.a_ptr() + memOffset + 8 * memslicesize + 7 * sizeof(size_t)),
          static_cast<void *>(G1._noOfBitSets.b_ptr() + i ),
            1 * sizeof(size_t),
            cudaMemcpyDeviceToHost, _d2h_stream));
  }
}


}}} // namespace
