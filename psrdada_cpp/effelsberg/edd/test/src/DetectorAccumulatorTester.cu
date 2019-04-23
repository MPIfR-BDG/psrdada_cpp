#include "psrdada_cpp/effelsberg/edd/test/DetectorAccumulatorTester.cuh"
#include "psrdada_cpp/cuda_utils.hpp"
#include <random>
#include <cmath>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace test {

DetectorAccumulatorTester::DetectorAccumulatorTester()
    : ::testing::Test()
    , _stream(0)
{

}

DetectorAccumulatorTester::~DetectorAccumulatorTester()
{

}

void DetectorAccumulatorTester::SetUp()
{
    CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
}

void DetectorAccumulatorTester::TearDown()
{
    CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
}

void DetectorAccumulatorTester::detect_c_reference(
    InputType const& input,
    OutputType& output,
    int nchans,
    int tscrunch,
    float scale,
    float offset)
{
    int nsamples = input.size() / nchans;
    int nsamples_out = nsamples / tscrunch;
    output.resize(nsamples_out * nchans);
    for (int chan_idx = 0; chan_idx < nchans; ++chan_idx)
    {
        for (int output_sample_idx = 0;
            output_sample_idx < nsamples_out;
            ++output_sample_idx)
        {
            double value = 0.0f;
            for (int input_sample_offset=0;
                input_sample_offset < tscrunch;
                ++input_sample_offset)
            {
                int input_sample_idx = output_sample_idx * tscrunch + input_sample_offset;
                float2 x = input[input_sample_idx * nchans + chan_idx];
                value += ((x.x * x.x) + (x.y * x.y));
            }
            output[output_sample_idx * nchans + chan_idx] = (int8_t)((value - offset) / scale);
        }
    }
}

void DetectorAccumulatorTester::compare_against_host(
    DetectorAccumulator<int8_t>::OutputType const& gpu_output,
    OutputType const& host_output)
{
    OutputType copy_from_gpu = gpu_output;
    ASSERT_EQ(host_output.size(), copy_from_gpu.size());
    for (std::size_t ii = 0; ii < host_output.size(); ++ii)
    {
	ASSERT_EQ(host_output[ii], copy_from_gpu[ii]);
    }
}

TEST_F(DetectorAccumulatorTester, noise_test)
{
    int nchans = 512;
    int tscrunch = 16;
    float stdev = 15.0;
    float scale = std::sqrt(stdev * tscrunch);
    int n = nchans * tscrunch * 16;
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, stdev);
    InputType host_input(n);
    for (int ii = 0; ii < n; ++ii)
    {
        host_input[ii].x = distribution(generator);
	host_input[ii].y = distribution(generator);
    }
    DetectorAccumulator<int8_t>::InputType gpu_input = host_input;
    DetectorAccumulator<int8_t>::OutputType gpu_output;
    gpu_output.resize(gpu_input.size() / tscrunch );
    OutputType host_output;
    DetectorAccumulator<int8_t> detector(nchans, tscrunch, scale, 0.0, _stream);
    detector.detect(gpu_input, gpu_output);
    detect_c_reference(host_input, host_output, nchans, tscrunch, scale, 0.0);
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_stream));
    compare_against_host(gpu_output, host_output);
}

} //namespace test
} //namespace edd
} //namespace meerkat
} //namespace psrdada_cpp
