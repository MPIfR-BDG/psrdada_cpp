#ifndef PSRDADA_CPP_EFFELSBERG_EDD_UNPACKERTESTER_CUH
#define PSRDADA_CPP_EFFELSBERG_EDD_UNPACKERTESTER_CUH

#include "psrdada_cpp/effelsberg/edd/Unpacker.cuh"
#include <gtest/gtest.h>
#include <vector>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace test {

class UnpackerTester: public ::testing::Test
{
public:
    typedef std::vector<uint64_t> InputType;
    typedef std::vector<float> OutputType;

protected:
    void SetUp() override;
    void TearDown() override;

public:
    UnpackerTester();
    ~UnpackerTester();

protected:
    void unpacker_12_to_32_c_reference(
        InputType const& input,
        OutputType& output);

    void unpacker_8_to_32_c_reference(
        InputType const& input,
        OutputType& output);

    void compare_against_host(
        Unpacker::OutputType const& gpu_output,
        OutputType const& host_output);

protected:
    cudaStream_t _stream;
};

} //namespace test
} //namespace edd
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_EFFELSBERG_EDD_UNPACKERTESTER_CUH
