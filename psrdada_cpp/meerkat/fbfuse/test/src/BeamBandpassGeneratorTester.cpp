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

void BeamBandpassGeneratorTester::operator()(RawBytes& block)
{
    ChannelStatistics* stats_ptr = reinterpret_cast<ChannelStatistics*>(block.ptr());
    std::vector<ChannelStatistics> stats(stats_ptr,
        stats_ptr + block.used_bytes() / sizeof(ChannelStatistics));
    for (auto const& stat: stats)
    {
        ASSERT_EQ(stat.mean, 0.0f);
        ASSERT_EQ(stat.variance, 0.0f);
    }
}

TEST_F(BeamBandpassGeneratorTester, all_zeros_single_accumulate)
{
    const unsigned int nbeams = 192;
    const unsigned int nchans_per_subband = 64;
    const unsigned int nsubbands = 1;
    const unsigned int heap_size = 8192;
    const unsigned int nbuffer_acc = 1;
    const unsigned int nheap_groups = 32;
    const std::size_t bytes = nheap_groups * nbeams * nsubbands * heap_size;
    BeamBandpassGenerator<BeamBandpassGeneratorTester> bandpass_generator(
        nbeams, nchans_per_subband, nsubbands,
        heap_size, nbuffer_acc, *this);
    std::vector<char> buffer(bytes, 0);
    RawBytes block(buffer.data(), bytes, bytes);
    bandpass_generator(block);
}

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

