#include "psrdada_cpp/effelsberg/edd/GatedSpectrometer.cuh"

#include "psrdada_cpp/dada_null_sink.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "gtest/gtest.h"

#include "thrust/device_vector.h"
#include "thrust/extrema.h"

namespace {

TEST(BitManipulationMacros, SetBit_TestBit) {
  for (int i = 0; i < 64; i++) {
    int64_t v = 0;
    SET_BIT(v, i);

    for (int j = 0; j < 64; j++) {
      if (j == i)
        EXPECT_EQ(TEST_BIT(v, j), 1);
      else
        EXPECT_EQ(TEST_BIT(v, j), 0);
    }
  }
}


TEST(GatedSpectrometer, ParameterSanity) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  psrdada_cpp::NullSink sink;

  // 8 or 12 bit sampling
  EXPECT_DEATH(psrdada_cpp::effelsberg::edd::GatedSpectrometer<decltype(sink)>(
                   0, 0, 0, 0, 4096, 0, 0, 0, 0, 0, sink),
               "_nbits == 8");
  // naccumulate > 0
  EXPECT_DEATH(psrdada_cpp::effelsberg::edd::GatedSpectrometer<decltype(sink)>(
                   0, 0, 0, 0, 4096, 0, 0, 8, 0, 0, sink),
               "_naccumulate");

  // selected side channel
  EXPECT_DEATH(psrdada_cpp::effelsberg::edd::GatedSpectrometer<decltype(sink)>(
                   0, 1, 2, 0, 4096, 0, 1, 8, 0, 0, sink),
               "nSideChannels");

  // selected bit
  EXPECT_DEATH(psrdada_cpp::effelsberg::edd::GatedSpectrometer<decltype(sink)>(
                   0, 2, 1, 65, 4096, 0, 1, 8, 0, 0, sink),
               "selectedBit");

  // valid construction
  psrdada_cpp::effelsberg::edd::GatedSpectrometer<decltype(sink)> a(
      4096 * 4096, 2, 1, 63, 4096, 1024, 1, 8, 100., 100., sink);
}
} // namespace


TEST(GatedSpectrometer, GatingKernel)
{
  size_t blockSize = 1024;
  size_t nBlocks = 16;

  thrust::device_vector<float> G0(blockSize * nBlocks);
  thrust::device_vector<float> G1(blockSize * nBlocks);
  thrust::device_vector<int64_t> _sideChannelData(nBlocks);

  thrust::fill(G0.begin(), G0.end(), 42);
  thrust::fill(G1.begin(), G1.end(), 23);
  thrust::fill(_sideChannelData.begin(), _sideChannelData.end(), 0);

  // everything to G0
  {
    const int64_t *sideCD =
        (int64_t *)(thrust::raw_pointer_cast(_sideChannelData.data()));
    psrdada_cpp::effelsberg::edd::gating<<<1024, 1024>>>(
          thrust::raw_pointer_cast(G0.data()),
          thrust::raw_pointer_cast(G1.data()), sideCD,
          G0.size(), blockSize, 0, 1,
          0);
    thrust::pair<thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator> minmax;
    minmax = thrust::minmax_element(G0.begin(), G0.end());
    EXPECT_EQ(*minmax.first, 42);
    EXPECT_EQ(*minmax.second, 42);

    minmax = thrust::minmax_element(G1.begin(), G1.end());
    EXPECT_EQ(*minmax.first, 0);
    EXPECT_EQ(*minmax.second, 0);
  }

  // everything to G1
  {
    thrust::fill(_sideChannelData.begin(), _sideChannelData.end(), 1L);
    const int64_t *sideCD =
        (int64_t *)(thrust::raw_pointer_cast(_sideChannelData.data()));
    psrdada_cpp::effelsberg::edd::gating<<<1024, 1024>>>(
          thrust::raw_pointer_cast(G0.data()),
          thrust::raw_pointer_cast(G1.data()), sideCD,
          G0.size(), blockSize, 0, 1,
          0);
    thrust::pair<thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator> minmax;
    minmax = thrust::minmax_element(G0.begin(), G0.end());
    EXPECT_EQ(*minmax.first, 0);
    EXPECT_EQ(*minmax.second, 0);

    minmax = thrust::minmax_element(G1.begin(), G1.end());
    EXPECT_EQ(*minmax.first, 42);
    EXPECT_EQ(*minmax.second, 42);
  }
}


TEST(GatedSpectrometer, countBitSet) {
  size_t nBlocks = 16;
  thrust::device_vector<int64_t> _sideChannelData(nBlocks);
  thrust::fill(_sideChannelData.begin(), _sideChannelData.end(), 0);
  const int64_t *sideCD =
      (int64_t *)(thrust::raw_pointer_cast(_sideChannelData.data()));

  thrust::device_vector<int> res(1);

  // test 1 side channel
  res[0] = 0;
  psrdada_cpp::effelsberg::edd::
      countBitSet<<<(_sideChannelData.size() + 255) / 256, 256>>>(
    sideCD, nBlocks, 0, 1, 0, thrust::raw_pointer_cast(res.data()));
  EXPECT_EQ(res[0], 0);

  res[0] = 0;
  thrust::fill(_sideChannelData.begin(), _sideChannelData.end(), 1L);
  psrdada_cpp::effelsberg::edd::countBitSet<<<(_sideChannelData.size() + 255) / 256, 256>>>(
    sideCD, nBlocks, 0, 1, 0, thrust::raw_pointer_cast(res.data()));
  EXPECT_EQ(res[0], nBlocks);

  // test mult side channels w stride.
  res[0] = 0;
  int nSideChannels = 4;
  thrust::fill(_sideChannelData.begin(), _sideChannelData.end(), 0);
  for (size_t i = 2; i < _sideChannelData.size(); i += nSideChannels)
    _sideChannelData[i] = 1L;
  psrdada_cpp::effelsberg::edd::countBitSet<<<(_sideChannelData.size() + 255) / 256, 256>>>(
    sideCD, nBlocks, 0, nSideChannels, 3,
    thrust::raw_pointer_cast(res.data()));
  EXPECT_EQ(res[0], 0);

  res[0] = 0;
  psrdada_cpp::effelsberg::edd::countBitSet<<<(_sideChannelData.size() + 255) / 256, 256>>>(
    sideCD, nBlocks, 0, nSideChannels, 2,
    thrust::raw_pointer_cast(res.data()));
  EXPECT_EQ(res[0], nBlocks / nSideChannels);
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}

