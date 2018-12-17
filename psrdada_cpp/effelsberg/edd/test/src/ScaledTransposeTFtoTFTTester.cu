#include "psrdada_cpp/effelsberg/edd/test/ScaledTransposeTFtoTFTTester.cuh"
#include "psrdada_cpp/cuda_utils.hpp"
#include <random>
#include <cmath>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace test {

ScaledTransposeTFtoTFTTester::ScaledTransposeTFtoTFTTester()
    : ::testing::Test()
    , _stream(0)
{

}

ScaledTransposeTFtoTFTTester::~ScaledTransposeTFtoTFTTester()
{

}

void ScaledTransposeTFtoTFTTester::SetUp()
{
    CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
}

void ScaledTransposeTFtoTFTTester::TearDown()
{
    CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
}

void ScaledTransposeTFtoTFTTester::transpose_c_reference(
        InputType const& input,
        OutputType& output,
        const int nchans, 
        const int nsamps, 
        const int nsamps_per_packet,
        const float scale,
        const float offset)
{
    int nsamples = input.size() / nchans;
    int outer_t_dim = nsamps / nsamps_per_packet;
    output.resize(input.size());
    for (int outer_t_idx = 0; outer_t_idx < outer_t_dim; ++outer_t_idx)
    {
        for (int chan_idx = 0; chan_idx < nchans; ++chan_idx)
        {
            for (int inner_t_idx = 0; inner_t_idx < nsamps_per_packet; ++inner_t_idx)
            {
                int load_idx = outer_t_idx * nchans * nsamps_per_packet + inner_t_idx * nchans + chan_idx;
                float2 val = input[load_idx];
                char2 out_val;
                out_val.x = (char)((val.x - offset)/scale);
                out_val.y = (char)((val.y - offset)/scale);
                int store_idx = outer_t_idx * nchans * nsamps_per_packet + chan_idx * nsamps_per_packet + inner_t_idx;
                output[store_idx] = out_val;
            }
        }
    }
}

void ScaledTransposeTFtoTFTTester::compare_against_host(
    ScaledTransposeTFtoTFT::OutputType const& gpu_output,
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

TEST_F(ScaledTransposeTFtoTFTTester, counter_test)
{
    int nchans = 16;
    int nsamps_per_packet = 8192;
    float stdev = 64.0f;
    float scale = 4.0f;
    int nsamps = nsamps_per_packet * 1024;
    int n = nchans * nsamps;
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, stdev);
    InputType host_input(n);
    for (int ii = 0; ii < n; ++ii)
    {
        host_input[ii].x = distribution(generator);
	    host_input[ii].y = distribution(generator);
    }
    ScaledTransposeTFtoTFT::InputType gpu_input = host_input;
    ScaledTransposeTFtoTFT::OutputType gpu_output;
    OutputType host_output;
    ScaledTransposeTFtoTFT transposer(nchans, nsamps_per_packet, scale, 0.0, _stream);
    transposer.transpose(gpu_input, gpu_output);
    transpose_c_reference(host_input, host_output, nchans, nsamps, nsamps_per_packet, scale, 0.0);
    compare_against_host(gpu_output, host_output);
}

} //namespace test
} //namespace edd
} //namespace meerkat
} //namespace psrdada_cpp
