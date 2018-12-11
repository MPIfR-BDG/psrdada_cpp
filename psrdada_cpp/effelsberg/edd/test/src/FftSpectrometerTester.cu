#include "psrdada_cpp/effelsberg/edd/test/FftSpectrometerTester.cuh"
#include "psrdada_cpp/effelsberg/edd/FftSpectrometer.cuh"
#include "psrdada_cpp/dada_null_sink.hpp"
#include "psrdada_cpp/double_host_buffer.cuh"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include <vector>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace test {

FftSpectrometerTester::FftSpectrometerTester()
    : ::testing::Test()
{

}

FftSpectrometerTester::~FftSpectrometerTester()
{

}

void FftSpectrometerTester::SetUp()
{
}

void FftSpectrometerTester::TearDown()
{
}

void FftSpectrometerTester::performance_test(
    std::size_t fft_length, std::size_t tscrunch, 
    std::size_t nsamps_out, std::size_t nbits)
{
    std::size_t input_block_bytes = tscrunch * fft_length * nsamps_out * nbits/8;
   
    DoublePinnedHostBuffer<char> input_block;
    input_block.resize(input_block_bytes);	
    RawBytes input_raw_bytes(input_block.a_ptr(), input_block_bytes, input_block_bytes);
    std::vector<char> header_block(4096);
    RawBytes header_raw_bytes(header_block.data(), 4096, 4096);
    NullSink null_sink;
    FftSpectrometer<NullSink> spectrometer(input_block_bytes, fft_length, tscrunch, nbits, 1.0f, 0.0f, null_sink);
    spectrometer.init(header_raw_bytes);
    for (int ii = 0; ii < 100; ++ii)
    {
        spectrometer(input_raw_bytes);
    }
}


TEST_F(FftSpectrometerTester, simple_exec_test)
{
    performance_test(1024, 16, 128, 12);
}

} //namespace test
} //namespace edd
} //namespace meerkat
} //namespace psrdada_cpp
