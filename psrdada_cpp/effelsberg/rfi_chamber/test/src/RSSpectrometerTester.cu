#include "psrdada_cpp/effelsberg/rfi_chamber/test/RSSpectrometerTester.cuh"
#include "psrdada_cpp/effelsberg/rfi_chamber/RSSpectrometer.cuh"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <random>
#include <cmath>
#include <complex>

namespace psrdada_cpp {
namespace effelsberg {
namespace rfi_chamber {
namespace test {

RSSpectrometerTester::RSSpectrometerTester()
    : ::testing::Test()
{

}

RSSpectrometerTester::~RSSpectrometerTester()
{


}

void RSSpectrometerTester::SetUp()
{

}

void RSSpectrometerTester::TearDown()
{

}

TEST_F(RSSpectrometerTester, test_dc_power)
{
    std::size_t input_nchans = 1<<15;
    std::size_t fft_length = 1<<10;
    std::size_t naccumulate = 10;
    std::size_t nskip = 0;
    std::size_t nspec_per_block = 4;
    std::size_t block_bytes = input_nchans * fft_length * nspec_per_block * sizeof(short2);
    std::vector<char> vheader(4096);
    //std::vector<char> vdata(block_bytes);
    char* ptr;
    cudaMallocHost((void**)&ptr, block_bytes);
    RawBytes header_block(vheader.data(), vheader.size(), vheader.size());
    //RawBytes data_block((char*) vdata.data(), vdata.size(), vdata.size());
    RawBytes data_block((char*) ptr, block_bytes, block_bytes);


    short2* s2ptr = reinterpret_cast<short2*>(ptr);
    for (std::size_t ii = 0;
        ii < input_nchans * fft_length * nspec_per_block;
        ii += input_nchans)
    {
        for (std::size_t chan_idx = 0; chan_idx < input_nchans; ++chan_idx)
        {
            s2ptr[ii + chan_idx].x = chan_idx;
            s2ptr[ii + chan_idx].y = 0;
        }
    }

    RSSpectrometer spectrometer(input_nchans, fft_length, naccumulate, nskip, "/tmp/dc_power_test.bin");
    spectrometer.init(header_block);
    bool retval;
    for (int ii=0; ii < 20; ++ii)
    {
        if (spectrometer(data_block))
        {
            break;
        }
    }
    cudaFreeHost(ptr);
}


TEST_F(RSSpectrometerTester, test_extreme_nchans)
{
    std::size_t input_nchans = 1<<15;
    std::size_t fft_length = 65536;
    std::size_t naccumulate = 4;
    std::size_t nskip = 5;
    std::size_t block_bytes = input_nchans * fft_length * 4 * sizeof(short2);
    std::vector<char> vheader(4096);
    //std::vector<char> vdata(block_bytes);
    char* ptr;
    cudaMallocHost((void**)&ptr, block_bytes);
    RawBytes header_block(vheader.data(), vheader.size(), vheader.size());
    //RawBytes data_block((char*) vdata.data(), vdata.size(), vdata.size());
    RawBytes data_block((char*) ptr, block_bytes, block_bytes);
    RSSpectrometer spectrometer(input_nchans, fft_length, naccumulate, nskip, "/dev/null");
    spectrometer.init(header_block);
    bool retval;
    for (int ii=0; ii < 20; ++ii)
    {
        if (spectrometer(data_block))
        {
            break;
        }
    }
    cudaFreeHost(ptr);
}

} //namespace test
} //namespace rfi_chamber
} //namespace effeslberg
} //namespace psrdada_cpp

