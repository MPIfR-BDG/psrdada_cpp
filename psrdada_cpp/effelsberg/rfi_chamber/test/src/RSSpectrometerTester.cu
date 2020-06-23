#include "psrdada_cpp/effelsberg/rfi_chamber/test/RSSpectrometerTester.cuh"
#include "psrdada_cpp/effelsberg/rfi_chamber/RSSpectrometer.cuh"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <arpa/inet.h>
#include <random>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>

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

void RSSpectrometerTester::run_dc_power_test(std::size_t input_nchans, std::size_t fft_length, std::size_t naccumulate)
{
    std::size_t nskip = 0;
    std::size_t nspec_per_block = 4;
    std::size_t block_bytes = input_nchans * fft_length * nspec_per_block * sizeof(short2);
    std::size_t nblocks = naccumulate / nspec_per_block + 1;
    std::vector<char> vheader(4096);
    char* ptr;
    cudaMallocHost((void**)&ptr, block_bytes);
    RawBytes header_block(vheader.data(), vheader.size(), vheader.size());
    RawBytes data_block((char*) ptr, block_bytes, block_bytes);


    short2* s2ptr = reinterpret_cast<short2*>(ptr);
    for (std::size_t ii = 0;
        ii < input_nchans * fft_length * nspec_per_block;
        ii += input_nchans)
    {
        for (std::size_t chan_idx = 0; chan_idx < input_nchans; ++chan_idx)
        {
            s2ptr[ii + chan_idx].x = htons(static_cast<unsigned short>(chan_idx));
            s2ptr[ii + chan_idx].y = 0;
        }
    }
    RSSpectrometer spectrometer(input_nchans, fft_length, naccumulate, nskip, "/tmp/dc_power_test.bin");
    spectrometer.init(header_block);
    for (int ii=0; ii < nblocks; ++ii)
    {
        if (spectrometer(data_block))
        {
            break;
        }
    }
    cudaFreeHost(ptr);


    std::vector<float> acc_spec(input_nchans * fft_length);

    std::ifstream infile;
    infile.open("/tmp/dc_power_test.bin", std::ifstream::in | std::ifstream::binary);
    if (!infile.is_open())
    {
        std::stringstream stream;
        stream << "Could not open file " << "/tmp/dc_power_test.bin";
        throw std::runtime_error(stream.str().c_str());
    }
    infile.read(reinterpret_cast<char*>(acc_spec.data()), input_nchans * fft_length * sizeof(float));
    infile.close();

    for (std::size_t input_chan_idx = 0; input_chan_idx < input_nchans; ++input_chan_idx)
    {
        float expected_peak = naccumulate * pow(input_chan_idx * fft_length, 2);
        for (std::size_t fft_idx = 0; fft_idx < fft_length; ++fft_idx)
        {
            if (fft_idx == fft_length/2)
            {
                ASSERT_NEAR(acc_spec[input_chan_idx*fft_length + fft_idx], expected_peak, expected_peak * 0.00001f);
            }
            else
            {
                ASSERT_NEAR(acc_spec[input_chan_idx*fft_length + fft_idx], 0.0f, 0.00001f);
            }

        }
    }
    EXPECT_EQ(0, remove("/tmp/dc_power_test.bin")) << "Error deleting file '/tmp/dc_power_test.bin'";
}

TEST_F(RSSpectrometerTester, test_dc_power_1024_chan)
{
    run_dc_power_test(1<<15, 1<<10, 10);
}


TEST_F(RSSpectrometerTester, test_dc_power_16384_chan)
{
    run_dc_power_test(1<<15, 1<<14, 10);
}

} //namespace test
} //namespace rfi_chamber
} //namespace effeslberg
} //namespace psrdada_cpp

