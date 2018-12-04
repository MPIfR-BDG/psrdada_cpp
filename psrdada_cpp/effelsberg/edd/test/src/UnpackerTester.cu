#include "psrdada_cpp/effelsberg/edd/test/UnpackerTester.cuh"
#include "psrdada_cpp/cuda_utils.hpp"

#define BSWAP64(x) ((0xFF00000000000000 & x) >> 56) | \
                   ((0x00FF000000000000 & x) >> 40) | \
                   ((0x0000FF0000000000 & x) >> 24) | \
                   ((0x000000FF00000000 & x) >>  8) | \
                   ((0x00000000FF000000 & x) <<  8) | \
                   ((0x0000000000FF0000 & x) << 24) | \
                   ((0x000000000000FF00 & x) << 40) | \
                   ((0x00000000000000FF & x) << 56)

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace test {

UnpackerTester::UnpackerTester()
    : ::testing::Test()
    , _stream(0)
{

}

UnpackerTester::~UnpackerTester()
{

}

void UnpackerTester::SetUp()
{
    CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
}

void UnpackerTester::TearDown()
{
    CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
}

void UnpackerTester::unpacker_12_to_32_c_reference(
    InputType const& input,
    OutputType& output)
{

}

void UnpackerTester::unpacker_8_to_32_c_reference(
    InputType const& input,
    OutputType& output)
{

}

void UnpackerTester::compare_against_host(
    Unpacker::OutputType const& gpu_output,
    OutputType const& host_output)
{

}

TEST_F(UnpackerTester, 12_bit_unpack_test)
{

}

TEST_F(UnpackerTester, 8_bit_unpack_test)
{

}


} //namespace test
} //namespace edd
} //namespace meerkat
} //namespace psrdada_cpp