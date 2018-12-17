#include "psrdada_cpp/effelsberg/paf/test/UnpackerTester.cuh"
#include "psrdada_cpp/cuda_utils.hpp"
#include <random>

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
namespace paf {
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


void UnpackerTester::unpacker_c_reference(
    InputType const& input,
    OutputType& output)
{
    /*...*/
}

void UnpackerTester::compare_against_host(
    Unpacker::OutputType const& gpu_output,
    OutputType const& host_output)
{
    OutputType copy_from_gpu = gpu_output;
    ASSERT_EQ(host_output.size(), copy_from_gpu.size());
    for (std::size_t ii = 0; ii < host_output.size(); ++ii)
    {
	   ASSERT_EQ(host_output[ii].x, copy_from_gpu[ii].x);
       ASSERT_EQ(host_output[ii].y, copy_from_gpu[ii].y);
    }
}

TEST_F(UnpackerTester, paf_unpack_test)
{
    std::size_t n = 1024 * 3;
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(1,1<<31);
    InputType host_input(n);
    for (int ii = 0; ii < n; ++ii)
    {
        host_input[ii] = distribution(generator);
    }
    Unpacker::InputType gpu_input = host_input;
    Unpacker::OutputType gpu_output;
    OutputType host_output;
    Unpacker unpacker(_stream);
    unpacker.unpack<12>(gpu_input, gpu_output);
    unpacker_c_reference(host_input, host_output);
    compare_against_host(gpu_output, host_output);
}


} //namespace test
} //namespace paf
} //namespace meerkat
} //namespace psrdada_cpp
