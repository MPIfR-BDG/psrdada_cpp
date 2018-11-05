#include "psrdada_cpp/meerkat/fbfuse/test/CoherentBeamformerTester.cuh"
#include "psrdada_cpp/meerkat/fbfuse/fbfuse_constants.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include <random>
#include <cmath>
#include <complex>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

CoherentBeamformerTester::CoherentBeamformerTester()
    : ::testing::Test()
    , _stream(0)
{

}

CoherentBeamformerTester::~CoherentBeamformerTester()
{


}

void CoherentBeamformerTester::SetUp()
{
    CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
}

void CoherentBeamformerTester::TearDown()
{
    CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
}

void CoherentBeamformerTester::beamformer_c_reference(
    HostVoltageVectorType const& ftpa_voltages,
    HostWeightsVectorType const& fbpa_weights,
    HostPowerVectorType& tbtf_powers,
    int nchannels,
    int naccumulate,
    int nsamples,
    int nbeams,
    int nantennas,
    int npol,
    float scale,
    float offset)
{
    float xx,yy,xy,yx;
    double power_sum = 0.0;
    double power_sq_sum = 0.0;
    std::size_t count = 0;
    for (int channel_idx = 0; channel_idx < nchannels; ++channel_idx)
    {
        BOOST_LOG_TRIVIAL(debug) << "Beamformer C reference: "
        << static_cast<int>(100.0f * (channel_idx + 1.0f) / nchannels)
        << "% complete";
        for (int sample_idx = 0; sample_idx < nsamples; sample_idx+=naccumulate)
        {
            for (int beam_idx = 0; beam_idx < nbeams; ++beam_idx)
            {
                float power = 0.0f;
                for (int sample_offset = 0; sample_offset < naccumulate; ++sample_offset)
                {
                    for (int pol_idx = 0; pol_idx < npol; ++pol_idx)
                    {
                        float2 accumulator = {0,0};
                        for (int antenna_idx = 0; antenna_idx < nantennas; ++antenna_idx)
                        {
                            int ftpa_voltages_idx = nantennas * npol * nsamples * channel_idx
                            + nantennas * npol * (sample_idx + sample_offset)
                            + nantennas * pol_idx
                            + antenna_idx;
                            char2 datum = ftpa_voltages[ftpa_voltages_idx];

                            int fbpa_weights_idx = nantennas * nbeams * channel_idx
                            + nantennas * beam_idx
                            + antenna_idx;
                            char2 weight = fbpa_weights[fbpa_weights_idx];

                            xx = datum.x * weight.x;
                            yy = datum.y * weight.y;
                            xy = datum.x * weight.y;
                            yx = datum.y * weight.x;
                            accumulator.x += xx - yy;
                            accumulator.y += xy + yx;
                        }
                        float r = accumulator.x;
                        float i = accumulator.y;
                        power += r*r + i*i;
                    }
                }
                int tf_size = FBFUSE_CB_NSAMPLES_PER_HEAP * nchannels;
                int btf_size = nbeams * tf_size;
                int output_sample_idx = sample_idx / naccumulate;
                int tbtf_powers_idx = (output_sample_idx / FBFUSE_CB_NSAMPLES_PER_HEAP * btf_size
                    + beam_idx * tf_size
                    + (output_sample_idx % FBFUSE_CB_NSAMPLES_PER_HEAP) * nchannels
                    + channel_idx);
                //int btf_powers_idx = beam_idx * nsamples/naccumulate * nchannels
                //+ sample_idx/naccumulate * nchannels
                //+ channel_idx;
                power_sum += power;
                power_sq_sum += power * power;
                ++count;
                //btf_powers[btf_powers_idx] = (int8_t) ((power - offset)/scale);
                tbtf_powers[tbtf_powers_idx] = (int8_t) ((power - offset)/scale);
            }
        }
    }
    double power_mean = power_sum / count;
    BOOST_LOG_TRIVIAL(debug) << "Average power level: " << power_mean;
    BOOST_LOG_TRIVIAL(debug) << "Power variance: " << power_sq_sum / count - power_mean * power_mean;
}

void CoherentBeamformerTester::compare_against_host(
    DeviceVoltageVectorType const& ftpa_voltages_gpu,
    DeviceWeightsVectorType const& fbpa_weights_gpu,
    DevicePowerVectorType& btf_powers_gpu,
    int nsamples)
{
    HostVoltageVectorType ftpa_voltages_host = ftpa_voltages_gpu;
    HostWeightsVectorType fbpa_weights_host = fbpa_weights_gpu;
    HostPowerVectorType btf_powers_cuda = btf_powers_gpu;
    HostPowerVectorType btf_powers_host(btf_powers_gpu.size());
    beamformer_c_reference(ftpa_voltages_host,
        fbpa_weights_host,
        btf_powers_host,
        _config.nchans(),
        _config.cb_tscrunch(),
        nsamples,
        _config.cb_nbeams(),
        _config.cb_nantennas(),
        _config.npol(),
        _config.cb_power_scaling(),
        _config.cb_power_offset());
    for (int ii = 0; ii < btf_powers_host.size(); ++ii)
    {
        std::cout << (int) btf_powers_cuda[ii] << ", ";
	    //std::cout << (int) btf_powers_host[ii] << ", " << (int) btf_powers_cuda[ii]
	    //<< ", (" << (int)btf_powers_host[ii] - (int)btf_powers_cuda[ii] << ");";
        ASSERT_TRUE(std::abs(static_cast<int>(btf_powers_host[ii]) - btf_powers_cuda[ii]) <= 1);
    }
    //std::cout << "\n";
}

TEST_F(CoherentBeamformerTester, representative_noise_test)
{
    const float input_level = 32.0f;
    const double pi = std::acos(-1);
    _config.input_level(input_level);
    _config.output_level(32.0f);
    std::default_random_engine generator;
    std::normal_distribution<float> normal_dist(0.0,input_level);
    std::uniform_real_distribution<float> uniform_dist(0.0,2*pi);

    CoherentBeamformer coherent_beamformer(_config);
    std::size_t ntimestamps = 32;
    std::size_t input_size = (ntimestamps * _config.cb_nantennas()
        * _config.nchans() * _config.nsamples_per_heap() * _config.npol());
    int nsamples = _config.nsamples_per_heap() * ntimestamps;
    std::size_t weights_size = _config.cb_nantennas() * _config.nchans() * _config.cb_nbeams();
    HostVoltageVectorType ftpa_voltages_host(input_size);
    for (int ii = 0; ii < ftpa_voltages_host.size(); ++ii)
    {
        ftpa_voltages_host[ii].x = static_cast<char>(std::lround(normal_dist(generator)));
        ftpa_voltages_host[ii].y = static_cast<char>(std::lround(normal_dist(generator)));
    }
    HostWeightsVectorType fbpa_weights_host(weights_size);
    for (int ii = 0; ii < fbpa_weights_host.size(); ++ii)
    {
        // Build complex weight as C * exp(i * theta).
        std::complex<double> val = 127.0f * std::exp(std::complex<float>(0.0f, uniform_dist(generator)));
        fbpa_weights_host[ii].x = static_cast<char>(std::lround(val.real()));
        fbpa_weights_host[ii].y = static_cast<char>(std::lround(val.imag()));
    }
    DeviceVoltageVectorType ftpa_voltages_gpu = ftpa_voltages_host;
    DeviceWeightsVectorType fbpa_weights_gpu = fbpa_weights_host;
    DevicePowerVectorType btf_powers_gpu;
    coherent_beamformer.beamform(ftpa_voltages_gpu, fbpa_weights_gpu, btf_powers_gpu, _stream);
    compare_against_host(ftpa_voltages_gpu, fbpa_weights_gpu, btf_powers_gpu, nsamples);
}

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

