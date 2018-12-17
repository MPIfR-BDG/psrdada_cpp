#ifndef PSRDADA_CPP_EFFELSBERG_PAF_UNPACKER_CUH
#define PSRDADA_CPP_EFFELSBERG_PAF_UNPACKER_CUH

#include "psrdada_cpp/common.hpp"
#include <thrust/device_vector.h>

namespace psrdada_cpp {
namespace effelsberg {
namespace paf {
namespace kernels {

__global__
void unpack_paf_float32(uint64_t const* __restrict__ in, float2* __restrict__ out, int n);

}

class Unpacker
{
public:
    typedef thrust::device_vector<int64_t> InputType;
    typedef thrust::device_vector<float2> OutputType;

public:

    Unpacker(cudaStream_t stream);
    ~Unpacker();
    Unpacker(Unpacker const&) = delete;

    void unpack(InputType const& input, OutputType& output);

private:
    cudaStream_t _stream;
};

} //namespace paf
} //namespace effelsberg
} //namespace psrdada_cpp

#endif // PSRDADA_CPP_EFFELSBERG_PAF_UNPACKER_CUH



