#include "psrdada_cpp/meerkat/fbfuse/test/SplitTransposeTester.cuh"
#include "psrdada_cpp/meerkat/fbfuse/fbfuse_constants.hpp"
#include "psrdada_cpp/cuda_utils.hpp"

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

SplitTransposeTester::SplitTransposeTester()
    : ::testing::Test()
    , _stream(0)
{

}

SplitTransposeTester::~SplitTransposeTester()
{


}

void SplitTransposeTester::SetUp()
{
    CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
}

void SplitTransposeTester::TearDown()
{
    CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
}

void SplitTransposeTester::transpose_c_reference(
    HostVoltageType const& input,
    HostVoltageType& output,
    int total_nantennas,
    int used_nantennas,
    int start_antenna,
    int nchans,
    int ntimestamps)
{
    //TAFTP to FTPA
    //Input dimensions
    int tp = _config.nsamples_per_heap() * _config.npol();
    int ftp = nchans * tp;
    int aftp = total_nantennas * ftp;

    //Output dimensions
    int pa = _config.npol() * used_nantennas;
    int tpa = _config.nsamples_per_heap() * ntimestamps * pa;
    output.resize(nchans * tpa);

    for (int timestamp_idx = 0; timestamp_idx < ntimestamps; ++timestamp_idx)
    {
        for (int antenna_idx = 0; antenna_idx < used_nantennas; ++antenna_idx)
        {
            int input_antenna_idx = antenna_idx + start_antenna;
            for (int chan_idx = 0; chan_idx < nchans; ++chan_idx)
            {
                for (int samp_idx = 0; samp_idx < _config.nsamples_per_heap(); ++samp_idx)
                {
                    for (int pol_idx = 0; pol_idx < _config.npol(); ++pol_idx)
                    {
                        int input_idx = (timestamp_idx * aftp + input_antenna_idx
                            * ftp + chan_idx * tp + samp_idx * _config.npol() + pol_idx);
                        int output_sample_idx = timestamp_idx * _config.nsamples_per_heap() + samp_idx;
                        int output_idx = (chan_idx * tpa + output_sample_idx * pa
                            + pol_idx * used_nantennas + antenna_idx);
                        output[output_idx] = input[input_idx];
                    }
                }
            }
        }
    }
}

void SplitTransposeTester::compare_against_host(
    DeviceVoltageType const& gpu_input,
    DeviceVoltageType const& gpu_output,
    std::size_t ntimestamps)
{
    HostVoltageType host_input = gpu_input;
    HostVoltageType host_output;
    HostVoltageType cuda_output = gpu_output;
    transpose_c_reference(host_input, host_output,
        _config.total_nantennas(), _config.cb_nantennas(),
        _config.cb_antenna_offset(), _config.nchans(),
        ntimestamps);
    for (int ii = 0; ii < host_output.size(); ++ii)
    {
        ASSERT_EQ(host_output[ii].x, cuda_output[ii].x);
        ASSERT_EQ(host_output[ii].y, cuda_output[ii].y);
    }
}

TEST_F(SplitTransposeTester, cycling_prime_test)
{
    SplitTranspose split_transpose(_config);
    std::size_t ntimestamps = 12;
    std::size_t input_size = (ntimestamps * _config.total_nantennas()
        * _config.nchans() * _config.nsamples_per_heap() * _config.npol());

    DeviceVoltageType host_gpu_input(input_size);
    for (int ii = 0; ii < input_size; ++ii)
    {
        host_gpu_input[ii] = ii % 113;
    }
    DeviceVoltageType gpu_input = host_gpu_input;
    DeviceVoltageType gpu_output;
    split_transpose.transpose(gpu_input, gpu_output, _stream);
    compare_against_host(gpu_input, gpu_output, ntimestamps);
}

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

