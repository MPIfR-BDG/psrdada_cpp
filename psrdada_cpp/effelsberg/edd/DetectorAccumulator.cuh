#ifndef PSRDADA_CPP_EFFELSBERG_EDD_DETECTORACCUMULATOR_CUH
#define PSRDADA_CPP_EFFELSBERG_EDD_DETECTORACCUMULATOR_CUH

#include "psrdada_cpp/common.hpp"
#include <thrust/device_vector.h>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace kernels {

template<typename T>
__global__
void detect_and_accumulate(float2 const* __restrict__ in, int8_t* __restrict__ out,
    int nchans, int nsamps, int naccumulate, float scale, float offset)
{
    for (int block_idx = blockIdx.x; block_idx < nsamps/naccumulate; block_idx += gridDim.x)
    {
        int read_offset = block_idx * naccumulate * nchans;
        int write_offset = block_idx * nchans;
        for (int chan_idx = threadIdx.x; chan_idx < nchans; chan_idx += blockDim.x)
        {
            float sum = 0.0f;
            for (int ii=0; ii < naccumulate; ++ii)
            {
                float2 tmp = in[read_offset + chan_idx + ii*nchans];
                float x = tmp.x * tmp.x;
                float y = tmp.y * tmp.y;
                sum += x + y;
            }
            out[write_offset + chan_idx] = (int8_t) ((sum - offset)/scale);
        }
    }
}

template<typename T>
__global__
void detect_and_accumulate(float2 const* __restrict__ in, float* __restrict__ out,
    int nchans, int nsamps, int naccumulate, float scale, float offset)
{
    for (int block_idx = blockIdx.x; block_idx < nsamps/naccumulate; block_idx += gridDim.x)
    {
        int read_offset = block_idx * naccumulate * nchans;
        int write_offset = block_idx * nchans;
        for (int chan_idx = threadIdx.x; chan_idx < nchans; chan_idx += blockDim.x)
        {
            double sum = 0.0;
            for (int ii=0; ii < naccumulate; ++ii)
            {
                float2 tmp = in[read_offset + chan_idx + ii*nchans];
                double x = tmp.x * tmp.x;
                double y = tmp.y * tmp.y;
                sum += x + y;
            }
            out[write_offset + chan_idx] = (float) sum;
        }
    }
}


}



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



