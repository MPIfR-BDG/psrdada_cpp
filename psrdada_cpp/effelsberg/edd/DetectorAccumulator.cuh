#ifndef PSRDADA_CPP_EFFELSBERG_EDD_DETECTORACCUMULATOR_CUH
#define PSRDADA_CPP_EFFELSBERG_EDD_DETECTORACCUMULATOR_CUH

#include "psrdada_cpp/common.hpp"
#include <thrust/device_vector.h>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace kernels {


// template argument unused but needed as nvcc gets otherwise confused.
template <typename T>
__global__
void detect_and_accumulate(float2 const* __restrict__ in, int8_t* __restrict__ out,
    int nchans, int nsamps, int naccumulate, float scale, float offset)
{
    // grid stride loop over output array to keep 
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; (i < nsamps * nchans / naccumulate); i += blockDim.x * gridDim.x)
    {
      double sum = 0.0f;
      size_t currentOutputSpectra = i / nchans;
      size_t currentChannel = i % nchans;

      for (size_t j = 0; j < naccumulate; j++)
      {
        float2 tmp = in[ j * nchans + currentOutputSpectra * nchans * naccumulate + currentChannel];
        double x = tmp.x * tmp.x;
        double y = tmp.y * tmp.y;
        sum += x + y;
      }
      out[i] = (int8_t) ((sum - offset)/scale);
    }

}


template <typename T>
__global__
void detect_and_accumulate(float2 const* __restrict__ in, float* __restrict__ out,
    int nchans, int nsamps, int naccumulate, float scale, float offset)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; (i < nsamps * nchans / naccumulate); i += blockDim.x * gridDim.x)
    {
      double sum = 0;
      size_t currentOutputSpectra = i / nchans;
      size_t currentChannel = i % nchans;

      for (size_t j = 0; j < naccumulate; j++)
      {
        float2 tmp = in[ j * nchans + currentOutputSpectra * nchans * naccumulate + currentChannel];
        double x = tmp.x * tmp.x;
        double y = tmp.y * tmp.y;
        sum += x + y;
      }
      out[i] = sum;
    }
}

} // namespace kernels



template <typename T>
class DetectorAccumulator
{
public:
    typedef thrust::device_vector<float2> InputType;
    typedef thrust::device_vector<T> OutputType;

public:
    DetectorAccumulator(DetectorAccumulator const&) = delete;


  DetectorAccumulator(
      int nchans, int tscrunch, float scale,
      float offset, cudaStream_t stream)
      : _nchans(nchans)
      , _tscrunch(tscrunch)
      , _scale(scale)
      , _offset(offset)
      , _stream(stream)
  {

  }

  ~DetectorAccumulator()
  {

  }

  void detect(InputType const& input, OutputType& output)
  {
      assert(input.size() % (_nchans * _tscrunch) == 0 /* Input is not a multiple of _nchans * _tscrunch*/);
      output.resize(input.size()/_tscrunch);
      int nsamps = input.size() / _nchans;
      float2 const* input_ptr = thrust::raw_pointer_cast(input.data());
      T * output_ptr = thrust::raw_pointer_cast(output.data());
      kernels::detect_and_accumulate<T> <<<1024, 1024, 0, _stream>>>(
          input_ptr, output_ptr, _nchans, nsamps, _tscrunch, _scale, _offset);
  }



private:
    int _nchans;
    int _tscrunch;
    float _scale;
    float _offset;
    cudaStream_t _stream;
};

} //namespace edd
} //namespace effelsberg
} //namespace psrdada_cpp

#endif // PSRDADA_CPP_EFFELSBERG_EDD_DETECTORACCUMULATOR_CUH



