#ifndef PSRDADA_CPP_EFFELSBERG_PAF_UNPACKERTESTER_CUH
#define PSRDADA_CPP_EFFELSBERG_PAF_UNPACKERTESTER_CUH

#include "psrdada_cpp/effelsberg/paf/Unpacker.cuh"
#include "thrust/host_vector.h"
#include <gtest/gtest.h>
#include <vector>

namespace psrdada_cpp {
namespace effelsberg {
namespace paf {
namespace test {

class UnpackerTester: public ::testing::Test
{
public:
    typedef thrust::host_vector<int64_t> InputType;
    typedef thrust::host_vector<float2> OutputType;

protected:
    void SetUp() override;
    void TearDown() override;

public:
    UnpackerTester();
    ~UnpackerTester();

protected:
    void unpacker_c_reference(
        InputType const& input,
        OutputType& output);

    void compare_against_host(
        Unpacker::OutputType const& gpu_output,
        OutputType const& host_output);

protected:
    cudaStream_t _stream;
};

} //namespace test
} //namespace paf
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_EFFELSBERG_PAF_UNPACKERTESTER_CUH
