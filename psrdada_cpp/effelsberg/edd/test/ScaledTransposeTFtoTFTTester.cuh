#ifndef PSRDADA_CPP_EFFELSBERG_EDD_SCALEDTRANSPOSETFTOTFTTESTER_CUH
#define PSRDADA_CPP_EFFELSBERG_EDD_SCALEDTRANSPOSETFTOTFTTESTER_CUH

#include "psrdada_cpp/effelsberg/edd/ScaledTransposeTFtoTFT.cuh"
#include "thrust/host_vector.h"
#include <gtest/gtest.h>
#include <vector>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace test {

class ScaledTransposeTFtoTFTTester: public ::testing::Test
{
public:
    typedef thrust::host_vector<float2> InputType;
    typedef thrust::host_vector<char2> OutputType;

protected:
    void SetUp() override;
    void TearDown() override;

public:
    ScaledTransposeTFtoTFTTester();
    ~ScaledTransposeTFtoTFTTester();

protected:
    void transpose_c_reference(
        InputType const& input,
        OutputType& output,
        const int nchans, 
        const int nsamps, 
        const int nsamps_per_packet,
        const float scale,
        const float offset);

    void compare_against_host(
        ScaledTransposeTFtoTFT::OutputType const& gpu_output,
        OutputType const& host_output);

protected:
    cudaStream_t _stream;
};

} //namespace test
} //namespace edd
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_EFFELSBERG_EDD_SCALEDTRANSPOSETFTOTFTTESTER_CUH
