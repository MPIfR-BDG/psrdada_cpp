#include "psrdada_cpp/meerkat/fbfuse/test/BeamBandpassGeneratorTester.hpp"
#include <vector>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

BeamBandpassGeneratorTester::BeamBandpassGeneratorTester()
    : ::testing::Test()
{

}

BeamBandpassGeneratorTester::~BeamBandpassGeneratorTester()
{

}

void BeamBandpassGeneratorTester::SetUp()
{

}

void BeamBandpassGeneratorTester::TearDown()
{

}

struct TestHandler
{
    TestHandler(std::vector<ChannelStatistics> const& expectation)
    : _expectation(expectation)
    , operator_called(false)
    {

    }

    void operator()(RawBytes& block)
    {
        ChannelStatistics* stats_ptr = reinterpret_cast<ChannelStatistics*>(block.ptr());
        std::vector<ChannelStatistics> stats(stats_ptr,
            stats_ptr + block.used_bytes() / sizeof(ChannelStatistics));
        ASSERT_EQ(stats.size(), _expectation.size());
        for (int ii = 0; ii < stats.size(); ++ii)
        {
            EXPECT_NEAR(stats[ii].mean, _expectation[ii].mean, 0.0001f);
            EXPECT_NEAR(stats[ii].variance, _expectation[ii].variance, 0.0001f);
        }
        operator_called = true;
    }

    std::vector<ChannelStatistics> const& _expectation;
    bool operator_called;
};

TEST_F(BeamBandpassGeneratorTester, all_zeros_single_accumulate)
{
    const unsigned int nbeams = 192;
    const unsigned int nchans_per_subband = 64;
    const unsigned int nsubbands = 1;
    const unsigned int heap_size = 8192;
    const unsigned int nbuffer_acc = 1;
    const unsigned int nheap_groups = 32;
    const std::size_t bytes = nheap_groups * nbeams * nsubbands * heap_size;
    std::vector<ChannelStatistics> expectation(
        nbeams * nsubbands * nchans_per_subband, {0.0f, 0.0f});
    TestHandler handler(expectation);
    BeamBandpassGenerator<TestHandler> bandpass_generator(
        nbeams, nchans_per_subband, nsubbands,
        heap_size, nbuffer_acc, handler);
    std::vector<char> buffer(bytes, 0);
    RawBytes block(buffer.data(), bytes, bytes);
    bandpass_generator(block);
    ASSERT_TRUE(handler.operator_called);
}

TEST_F(BeamBandpassGeneratorTester, mean_equals_channel_id_var_zero)
{
    const unsigned int nbeams = 192;
    const unsigned int nchans_per_subband = 64;
    const unsigned int nsubbands = 1;
    const unsigned int heap_size = 8192;
    const unsigned int nbuffer_acc = 1;
    const unsigned int nheap_groups = 32;
    const std::size_t bytes = nheap_groups * nbeams * nsubbands * heap_size;

    std::vector<ChannelStatistics> expectation(
        nbeams * nsubbands * nchans_per_subband);
    for (int jj = 0; jj < nbeams; ++jj)
    {
        for (int kk = 0; kk < nchans_per_subband * nsubbands; ++kk)
        {
            expectation[jj * nchans_per_subband * nsubbands + kk].mean = static_cast<float>(kk);
            expectation[jj * nchans_per_subband * nsubbands + kk].variance = 0.0f;
	}
    }
    TestHandler handler(expectation);
    BeamBandpassGenerator<TestHandler> bandpass_generator(
        nbeams, nchans_per_subband, nsubbands,
        heap_size, nbuffer_acc, handler);
    std::vector<char> buffer(bytes, 0);
    const unsigned int nsamps_per_heap = heap_size / nchans_per_subband;
    RawBytes block(buffer.data(), bytes, bytes);
    for (int ii = 0; ii< bytes; ii += heap_size * nsubbands)
    {
        for (int subband_idx = 0; subband_idx < nsubbands; ++subband_idx)
        {
            for (int samp_idx = 0; samp_idx < nsamps_per_heap; ++samp_idx)
            {
                for (int chan_idx = 0; chan_idx < nchans_per_subband; ++chan_idx)
                {
                    std::size_t index = ii + subband_idx * heap_size + samp_idx * nchans_per_subband + chan_idx;
                    buffer[index] = subband_idx * nchans_per_subband + chan_idx;
                }
            }
        }
    }
    bandpass_generator(block);
    ASSERT_TRUE(handler.operator_called);
}

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

