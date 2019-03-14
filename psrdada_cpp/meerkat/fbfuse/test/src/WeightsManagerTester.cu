#include "psrdada_cpp/meerkat/fbfuse/test/WeightsManagerTester.cuh"
#include "psrdada_cpp/meerkat/fbfuse/fbfuse_constants.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "thrust/host_vector.h"

#define TWOPI 6.283185307179586f

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

WeightsManagerTester::WeightsManagerTester()
    : ::testing::Test()
    , _stream(0)
{

}

WeightsManagerTester::~WeightsManagerTester()
{


}

void WeightsManagerTester::SetUp()
{
    _config.centre_frequency(1.4e9);
    _config.bandwidth(56.0e6);
    CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
}

void WeightsManagerTester::TearDown()
{
    CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
}

void WeightsManagerTester::calc_weights_c_reference(
    thrust::host_vector<float2> const& delay_models,
    thrust::host_vector<char2>& weights,
    std::vector<float> const& channel_frequencies,
    int nantennas,
    int nbeams,
    int nchans,
    float tstart,
    float tstep,
    int ntsteps)
{
    float2 weight;
    char2 compressed_weight;
    for (int antenna_idx = 0; antenna_idx < nantennas; ++antenna_idx)
    {
        for (int beam_idx = 0; beam_idx < nbeams; ++beam_idx)
        {
            float2 delay_model = delay_models[beam_idx * nantennas + antenna_idx];
            for (int chan_idx = 0; chan_idx < nchans; ++chan_idx)
            {
                float frequency = channel_frequencies[chan_idx];
                for (int time_idx = 0; time_idx < ntsteps; ++time_idx)
                {
                    float t = tstart + time_idx * tstep;
                    float phase = (t * delay_model.x + delay_model.y) * frequency;
                    sincosf(TWOPI * phase, &weight.y, &weight.x);
                    compressed_weight.x = (char) round(weight.x * 127.0f);
                    compressed_weight.y = (char) round(weight.y * 127.0f);
                    int output_idx = nantennas * ( nbeams *
                        ( time_idx * nchans + chan_idx ) + beam_idx ) + antenna_idx;
                    weights[output_idx] = compressed_weight;
                }
            }
        }
    }
}

void WeightsManagerTester::compare_against_host(
    DelayVectorType const& delays,
    WeightsVectorType const& weights,
    TimeType epoch)
{
    // Implicit device to host copies
    thrust::host_vector<float2> host_delays = delays;
    thrust::host_vector<char2> cuda_weights = weights;
    thrust::host_vector<char2> c_weights(cuda_weights.size());
    calc_weights_c_reference(host_delays, c_weights,
        _config.channel_frequencies(), _config.cb_nantennas(),
        _config.cb_nbeams(), _config.channel_frequencies().size(),
        epoch, 0.0, 1);
    for (size_t ii = 0; ii < cuda_weights.size(); ++ii)
    {
        ASSERT_EQ(c_weights[ii].x, cuda_weights[ii].x);
        ASSERT_EQ(c_weights[ii].y, cuda_weights[ii].y);
    }
}

TEST_F(WeightsManagerTester, test_zero_value)
{
    // This is always the size of the delay array
    std::size_t delays_size = FBFUSE_CB_NBEAMS * FBFUSE_CB_NANTENNAS;
    WeightsManager weights_manager(_config, _stream);
    // This is a thrust::device_vector<float2>
    DelayVectorType delays(delays_size, {0.0, 0.0});
    TimeType epoch = 0.0;
    // First try everything with only zeros
    auto const& weights = weights_manager.weights(delays, epoch);
    compare_against_host(delays, weights, epoch);
}

TEST_F(WeightsManagerTester, test_real_value)
{
    // This is always the size of the delay array
    std::size_t delays_size = FBFUSE_CB_NBEAMS * FBFUSE_CB_NANTENNAS;
    WeightsManager weights_manager(_config, _stream);
    // This is a thrust::device_vector<float2>
    DelayVectorType delays(delays_size, {1e-11f, 1e-10f});
    TimeType epoch = 10.0;
    // First try everything with only zeros
    auto const& weights = weights_manager.weights(delays, epoch);
    compare_against_host(delays, weights, epoch);
}

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

