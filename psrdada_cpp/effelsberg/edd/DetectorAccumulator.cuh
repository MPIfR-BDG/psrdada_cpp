#ifndef PSRDADA_CPP_EFFELSBERG_EDD_DETECTORACCUMULATOR_CUH
#define PSRDADA_CPP_EFFELSBERG_EDD_DETECTORACCUMULATOR_CUH

#include "psrdada_cpp/common.hpp"
#include <thrust/device_vector.h>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace kernels {

__global__
void detect_and_accumulate(float2 const* __restrict__ in, int8_t* __restrict__ out,
    int nchans, int nsamps, int naccumulate, float scale, float offset);

}

class DetectorAccumulator
{
public:
    typedef thrust::device_vector<float2> InputType;
    typedef thrust::device_vector<int8_t> OutputType;

public:
    DetectorAccumulator(int nchans, int tscrunch, float scale, float offset, cudaStream_t stream);
    ~DetectorAccumulator();
    DetectorAccumulator(DetectorAccumulator const&) = delete;
    void detect(InputType const& input, OutputType& output);

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



