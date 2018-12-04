#ifndef PSRDADA_CPP_EFFELSBERG_EDD_UNPACKER_CUH
#define PSRDADA_CPP_EFFELSBERG_EDD_UNPACKER_CUH

#include "psrdada_cpp/common.hpp"
#include <thrust/device_vector.h>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace kernels {

__global__
void unpack_edd_12bit_to_float32(uint64_t* __restrict__ in, float* __restrict__ out, int n);

__global__
void unpack_edd_8bit_to_float32(uint64_t* __restrict__ in, float* __restrict__ out, int n);

}

class Unpacker
{
public:
    typedef thrust::device_vector<uint64_t> InputType;
    typedef thrust::device_vector<float> OutputType;

public:

    Unpacker(cudaStream_t stream);
    ~Unpacker();
    Unpacker(Unpacker const&) = delete;

    template <int Nbits>
    void unpack(InputType const& input, OutputType& output);

private:
    cudaStream_t _copy_stream;
};

} //namespace edd
} //namespace effelsberg
} //namespace psrdada_cpp

#endif // PSRDADA_CPP_EFFELSBERG_EDD_UNPACKER_CUH



