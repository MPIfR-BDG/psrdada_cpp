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

TEST_F(RSSpectrometerTester, test_exec)
{
    std::size_t input_nchans = 1<<15;
    std::size_t fft_length = 65536;
    std::size_t naccumulate = 100;
    std::size_t nskip = 5;

    std::size_t block_bytes = input_nchans * fft_length * 4 * sizeof(short2);

    std::vector<char> vheader(4096);
    //std::vector<char> vdata(block_bytes);
    char* ptr;
    cudaMallocHost((void**)&ptr, block_bytes);

    //std::vector<char,thrust::system::cuda::experimental::pinned_allocator<char> > vdata(block_bytes);


    RawBytes header_block(vheader.data(), vheader.size(), vheader.size());
    //RawBytes data_block((char*) vdata.data(), vdata.size(), vdata.size());
    RawBytes data_block((char*) ptr, block_bytes, block_bytes);
    RSSpectrometer spectrometer(input_nchans, fft_length, naccumulate, nskip, "/tmp/dump.bin");
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

