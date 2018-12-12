#include "psrdada_cpp/effelsberg/edd/test/ChanneliserTester.cuh"
#include "psrdada_cpp/effelsberg/edd/Channeliser.cuh"
#include "psrdada_cpp/dada_null_sink.hpp"
#include "psrdada_cpp/double_host_buffer.cuh"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include <vector>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace test {

ChanneliserTester::ChanneliserTester()
    : ::testing::Test()
{

}

ChanneliserTester::~ChanneliserTester()
{

}

void ChanneliserTester::SetUp()
{
}

void ChanneliserTester::TearDown()
{
}


       std::size_t buffer_bytes,
        std::size_t fft_length,
        std::size_t nbits,
        float input_level,
        float output_level,
        HandlerType& handler

void ChanneliserTester::performance_test(std::size_t fft_length, std::size_t nbits)
{
    std::size_t input_block_bytes = fft_length * 8192 * 1024 * nbits / 8;
   
    DoublePinnedHostBuffer<char> input_block;
    input_block.resize(input_block_bytes);	
    RawBytes input_raw_bytes(input_block.a_ptr(), input_block_bytes, input_block_bytes);
    std::vector<char> header_block(4096);
    RawBytes header_raw_bytes(header_block.data(), 4096, 4096);
    NullSink null_sink;
    Channeliser<NullSink> channeliser(input_block_bytes, fft_length, nbits, 16.0f, 16.0f, null_sink);
    spectrometer.init(header_raw_bytes);
    for (int ii = 0; ii < 100; ++ii)
    {
        channeliser(input_raw_bytes);
    }
}


TEST_F(ChanneliserTester, simple_exec_test)
{
    performance_test(1024, 16, 128, 12);
}

} //namespace test
} //namespace edd
} //namespace meerkat
} //namespace psrdada_cpp
