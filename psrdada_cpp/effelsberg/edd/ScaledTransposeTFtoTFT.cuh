#ifndef PSRDADA_CPP_EFFELSBERG_EDD_SCALEDTRANSPOSETFTOTFT_CUH
#define PSRDADA_CPP_EFFELSBERG_EDD_SCALEDTRANSPOSETFTOTFT_CUH

#include "psrdada_cpp/common.hpp"
#include <thrust/device_vector.h>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace kernels {

__global__
void tf_to_tft_transpose(
    float2 const* __restrict__ input, 
    char2* __restrict__ output, 
    const int nchans, 
    const int nsamps, 
    const int nsamps_per_packet,
    const int nsamps_per_load,
    const float scale,
    const float offset);

} // namespace kernels

class ScaledTransposeTFtoTFT
{
public:
    typedef thrust::device_vector<float2> InputType;
    typedef thrust::device_vector<char2> OutputType;

public:
    ScaledTransposeTFtoTFT(int nchans, int nsamps_per_packet, float scale, float offset, cudaStream_t stream);
    ~ScaledTransposeTFtoTFT();
    ScaledTransposeTFtoTFT(ScaledTransposeTFtoTFT const&) = delete;
    void transpose(InputType const& input, OutputType& output);

private:
    int _nchans;
    int _nsamps_per_packet;
    float _scale;
    float _offset;
    cudaStream_t _stream;
};

} //namespace edd
} //namespace effelsberg
} //namespace psrdada_cpp

#endif // PSRDADA_CPP_EFFELSBERG_EDD_SCALEDTRANSPOSETFTOTFT_CUH



