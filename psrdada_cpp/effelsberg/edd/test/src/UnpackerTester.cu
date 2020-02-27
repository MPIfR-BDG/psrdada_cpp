#include "psrdada_cpp/effelsberg/edd/test/UnpackerTester.cuh"
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
    uint64_t val, rest;
    assert(input.size() % 3 == 0 /*Input must be a multiple of 3 for 12-bit unpacking*/);
    output.reserve(input.size() * 64 / 12);
    for (std::size_t ii = 0; ii < input.size(); ii += 3)
    {
        val = be64toh(input[ii]);
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0xFFF0000000000000 & val) <<  0) >> 52));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x000FFF0000000000 & val) << 12) >> 52));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x000000FFF0000000 & val) << 24) >> 52));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x000000000FFF0000 & val) << 36) >> 52));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x000000000000FFF0 & val) << 48) >> 52));
        rest = ( 0x000000000000000F & val) << 60; // 4 bits rest.
        val = be64toh(input[ii + 1]);
        output.push_back(static_cast<float>(
            static_cast<int64_t>(((0xFF00000000000000 & val) >>  4) | rest) >> 52));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x00FFF00000000000 & val) <<  8) >> 52));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x00000FFF00000000 & val) << 20) >> 52));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x00000000FFF00000 & val) << 32) >> 52));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x00000000000FFF00 & val) << 44) >> 52));
        rest = ( 0x00000000000000FF & val) << 56; // 8 bits rest.
        val = be64toh(input[ii + 2]);
        output.push_back(static_cast<float>(
            static_cast<int64_t>(((0xF000000000000000 & val) >>  8) | rest) >> 52));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x0FFF000000000000 & val) <<  4) >> 52));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x0000FFF000000000 & val) << 16) >> 52));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x0000000FFF000000 & val) << 28) >> 52));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x0000000000FFF000 & val) << 40) >> 52));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x0000000000000FFF & val) << 52) >> 52));
    }
    assert(output.size() == input.size() * 64 / 12 /*Output is not the expected size*/);
}

void UnpackerTester::unpacker_10_to_32_c_reference(
    InputType const& input,
    OutputType& output)
{
    uint64_t val, rest;
    assert(input.size() % 5 == 0 /*Input must be a multiple of 5 for 10-bit unpacking*/);
    output.reserve(input.size() * 64 / 10);
    for (std::size_t ii = 0; ii < input.size(); ii += 5)
    {
        val = be64toh(input[ii]);
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0xFFC0000000000000 & val) <<  0) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x003FF00000000000 & val) << 10) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x00000FFC00000000 & val) << 20) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x00000003FF000000 & val) << 30) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x0000000000FFC000 & val) << 40) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x0000000000003FF0 & val) << 50) >> 54));
        rest =          ( 0x000000000000000F & val) << 60; // 4 bits rest.
        val = be64toh(input[ii + 1]);
        output.push_back(static_cast<float>(
            static_cast<int64_t>(((0xFC00000000000000 & val) >> 4) | rest) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x03FF000000000000 & val) <<  6) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x0000FFC000000000 & val) << 16) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x0000003FF0000000 & val) << 26) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x000000000FFC0000 & val) << 36) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x000000000003FF00 & val) << 46) >> 54));
        rest = ( 0x00000000000000FF & val) << 56; // 8 bits rest.
        val = be64toh(input[ii + 2]);
        output.push_back(static_cast<float>(
            static_cast<int64_t>(((0xC000000000000000 & val) >> 8) | rest) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x3FF0000000000000 & val) <<  2) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x000FFC0000000000 & val) << 12) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x000003FF00000000 & val) << 22) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x00000000FFC00000 & val) << 32) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x00000000003FF000 & val) << 42) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x0000000000000FFC & val) << 52) >> 54));
        rest = ( 0x0000000000000003 & val) << 62; // 2 bits rest.
        val = be64toh(input[ii + 3]);
        output.push_back(static_cast<float>(
            static_cast<int64_t>(((0xFF00000000000000 & val) >> 2) | rest) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x00FFC00000000000 & val) <<  8) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x00003FF000000000 & val) << 18) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x0000000FFC000000 & val) << 28) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x0000000003FF0000 & val) << 38) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x000000000000FFC0 & val) << 48) >> 54));
        rest = ( 0x000000000000003F & val) << 58; // 6 bits rest.
        val = be64toh(input[ii + 4]);
        output.push_back(static_cast<float>(
            static_cast<int64_t>(((0xF000000000000000 & val) >> 6) | rest) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x0FFC000000000000 & val) <<  4) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x0003FF0000000000 & val) << 14) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x000000FFC0000000 & val) << 24) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x000000003FF00000 & val) << 34) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x00000000000FFC00 & val) << 44) >> 54));
        output.push_back(static_cast<float>(
            static_cast<int64_t>(( 0x00000000000003FF & val) << 54) >> 54));
    }
}

void UnpackerTester::unpacker_8_to_32_c_reference(
    InputType const& input,
    OutputType& output)
{
    output.reserve(input.size() * 8);
    for (uint64_t val: input)
    {
        val = be64toh(val);
        output.push_back(static_cast<float>(
            static_cast<int64_t>((0xFF00000000000000 & val) <<  0) >> 56));
        output.push_back(static_cast<float>(
            static_cast<int64_t>((0x00FF000000000000 & val) <<  8) >> 56));
        output.push_back(static_cast<float>(
            static_cast<int64_t>((0x0000FF0000000000 & val) << 16) >> 56));
        output.push_back(static_cast<float>(
            static_cast<int64_t>((0x000000FF00000000 & val) << 24) >> 56));
        output.push_back(static_cast<float>(
            static_cast<int64_t>((0x00000000FF000000 & val) << 32) >> 56));
        output.push_back(static_cast<float>(
            static_cast<int64_t>((0x0000000000FF0000 & val) << 40) >> 56));
        output.push_back(static_cast<float>(
            static_cast<int64_t>((0x000000000000FF00 & val) << 48) >> 56));
        output.push_back(static_cast<float>(
            static_cast<int64_t>((0x00000000000000FF & val) << 56) >> 56));
    }
}

void UnpackerTester::compare_against_host(
    Unpacker::OutputType const& gpu_output,
    OutputType const& host_output)
{
    OutputType copy_from_gpu = gpu_output;
    ASSERT_EQ(host_output.size(), copy_from_gpu.size());
    for (std::size_t ii = 0; ii < host_output.size(); ++ii)
    {
	ASSERT_EQ(host_output[ii], copy_from_gpu[ii]);
    }
}

TEST_F(UnpackerTester, 12_bit_unpack_test)
{
    std::size_t n = 1024 * 3;
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(1,1<<31);
    InputType host_input(n);
    for (size_t ii = 0; ii < n; ++ii)
    {
        host_input[ii] = distribution(generator);
    }
    Unpacker::InputType gpu_input = host_input;
    Unpacker::OutputType gpu_output;
    gpu_output.resize(host_input.size() * sizeof(host_input[0]) * 8 / 12);
    OutputType host_output;
    Unpacker unpacker(_stream);
    unpacker.unpack<12>(gpu_input, gpu_output);
    unpacker_12_to_32_c_reference(host_input, host_output);
    compare_against_host(gpu_output, host_output);
}

TEST_F(UnpackerTester, 8_bit_unpack_test)
{
    std::size_t n = 512 * 8;
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(1,1<<31);
    InputType host_input(n);
    for (size_t ii = 0; ii < n; ++ii)
    {
        host_input[ii] = distribution(generator);
    }
    Unpacker::InputType gpu_input = host_input;
    Unpacker::OutputType gpu_output;
    gpu_output.resize(host_input.size() * sizeof(host_input[0]) * 8 / 8);
    OutputType host_output;
    Unpacker unpacker(_stream);
    unpacker.unpack<8>(gpu_input, gpu_output);
    unpacker_8_to_32_c_reference(host_input, host_output);
    compare_against_host(gpu_output, host_output);
}


} //namespace test
} //namespace edd
} //namespace meerkat
} //namespace psrdada_cpp
