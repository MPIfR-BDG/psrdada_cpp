#include "psrdada_cpp/effelsberg/edd/GatedSpectrometer.cuh"

#include "psrdada_cpp/dada_null_sink.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "gtest/gtest.h"

#include "thrust/device_vector.h"
#include "thrust/extrema.h"

namespace {

TEST(GatedSpectrometer, BitManipulationMacros) {
  for (int i = 0; i < 64; i++) {
    uint64_t v = 0;
    SET_BIT(v, i);

    for (int j = 0; j < 64; j++) {
      if (j == i)
        EXPECT_EQ(TEST_BIT(v, j), 1);
      else
        EXPECT_EQ(TEST_BIT(v, j), 0);
    }
  }
}

//
//TEST(GatedSpectrometer, ParameterSanity) {
//  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
//  psrdada_cpp::NullSink sink;
//
//  // 8 or 12 bit sampling
//  EXPECT_DEATH(psrdada_cpp::effelsberg::edd::GatedSpectrometer<decltype(sink),int8_t > (0, 0, 0, 0, 4096, 0, 0, 0, 0, 0, sink),
//               "_nbits == 8");
//  // naccumulate > 0
//  EXPECT_DEATH(psrdada_cpp::effelsberg::edd::GatedSpectrometer<decltype(sink),int8_t > (0, 0, 0, 0, 4096, 0, 0, 8, 0, 0, sink),
//               "_naccumulate");
//
//  // selected side channel
//  EXPECT_DEATH(psrdada_cpp::effelsberg::edd::GatedSpectrometer<decltype(sink),int8_t > (0, 1, 2, 0, 4096, 0, 1, 8, 0, 0, sink),
//               "nSideChannels");
//
//  // selected bit
//  EXPECT_DEATH(psrdada_cpp::effelsberg::edd::GatedSpectrometer<decltype(sink),int8_t > (0, 2, 1, 65, 4096, 0, 1, 8, 0, 0, sink),
//               "selectedBit");
//
//  // valid construction
//  psrdada_cpp::effelsberg::edd::GatedSpectrometer<decltype(sink), int8_t> a(
//      4096 * 4096, 2, 1, 63, 4096, 1024, 1, 8, 100., 100., sink);
//}
} // namespace


TEST(GatedSpectrometer, GatingKernel)
{
  size_t blockSize = 1024;
  size_t nBlocks = 16 * 1024;

  thrust::device_vector<float> G0(blockSize * nBlocks);
  thrust::device_vector<float> G1(blockSize * nBlocks);
  thrust::device_vector<uint64_t> _sideChannelData(nBlocks);
  thrust::device_vector<psrdada_cpp::effelsberg::edd::uint64_cu> _nG0(1);
  thrust::device_vector<psrdada_cpp::effelsberg::edd::uint64_cu> _nG1(1);
  thrust::device_vector<float> baseLine(1);

  thrust::fill(G0.begin(), G0.end(), 42);
  thrust::fill(G1.begin(), G1.end(), 23);
  thrust::fill(_sideChannelData.begin(), _sideChannelData.end(), 0);

  // everything to G0
  {
    thrust::fill(_nG0.begin(), _nG0.end(), 0);
    thrust::fill(_nG1.begin(), _nG1.end(), 0);
    baseLine[0] = 0.;
    const uint64_t *sideCD =
        (uint64_t *)(thrust::raw_pointer_cast(_sideChannelData.data()));
    psrdada_cpp::effelsberg::edd::gating<<<1024, 1024>>>(
          thrust::raw_pointer_cast(G0.data()),
          thrust::raw_pointer_cast(G1.data()), sideCD,
          G0.size(), blockSize, 0, 1,
          0, thrust::raw_pointer_cast(baseLine.data()),
          thrust::raw_pointer_cast(_nG0.data()),
          thrust::raw_pointer_cast(_nG1.data())
          );
    thrust::pair<thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator> minmax;
    minmax = thrust::minmax_element(G0.begin(), G0.end());
    EXPECT_EQ(*minmax.first, 42);
    EXPECT_EQ(*minmax.second, 42);

    minmax = thrust::minmax_element(G1.begin(), G1.end());
    EXPECT_EQ(*minmax.first, 0);
    EXPECT_EQ(*minmax.second, 0);

    EXPECT_EQ(_nG0[0], G0.size());
    EXPECT_EQ(_nG1[0], 0u);
  }

  // everything to G1 // with baseline -5
  {
    thrust::fill(_nG0.begin(), _nG0.end(), 0);
    thrust::fill(_nG1.begin(), _nG1.end(), 0);
    baseLine[0] = -5. * G0.size();
    thrust::fill(_sideChannelData.begin(), _sideChannelData.end(), 1L);
    const uint64_t *sideCD =
        (uint64_t *)(thrust::raw_pointer_cast(_sideChannelData.data()));
    psrdada_cpp::effelsberg::edd::gating<<<1024, 1024>>>(
          thrust::raw_pointer_cast(G0.data()),
          thrust::raw_pointer_cast(G1.data()), sideCD,
          G0.size(), blockSize, 0, 1,
          0, thrust::raw_pointer_cast(baseLine.data()),
          thrust::raw_pointer_cast(_nG0.data()),
          thrust::raw_pointer_cast(_nG1.data())
          );
    thrust::pair<thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator> minmax;
    minmax = thrust::minmax_element(G0.begin(), G0.end());
    EXPECT_EQ(*minmax.first, -5.);
    EXPECT_EQ(*minmax.second, -5.);

    minmax = thrust::minmax_element(G1.begin(), G1.end());
    EXPECT_EQ(*minmax.first, 42);
    EXPECT_EQ(*minmax.second, 42);

    EXPECT_EQ(_nG0[0], 0u);
    EXPECT_EQ(_nG1[0], G1.size());
  }
}


TEST(GatedSpectrometer, countBitSet) {
  size_t nBlocks = 100000;
  int nSideChannels = 4;
  thrust::device_vector<uint64_t> _sideChannelData(nBlocks * nSideChannels);
  thrust::fill(_sideChannelData.begin(), _sideChannelData.end(), 0L);
  const uint64_t *sideCD =
      (uint64_t *)(thrust::raw_pointer_cast(_sideChannelData.data()));

  thrust::device_vector<size_t> res(1);

  // test 1 side channel
  res[0] = 0;
  psrdada_cpp::effelsberg::edd::
      countBitSet<<<1, 1024>>>(
    sideCD, nBlocks, 0, 1, 0, thrust::raw_pointer_cast(res.data()));

  EXPECT_EQ(res[0], 0u);

  res[0] = 0;
  thrust::fill(_sideChannelData.begin(), _sideChannelData.end(), 1L);
  psrdada_cpp::effelsberg::edd::countBitSet<<<1, 1024>>>(
    sideCD, nBlocks, 0, 1, 0, thrust::raw_pointer_cast(res.data()));
  EXPECT_EQ(res[0], nBlocks);

  // test mult side channels w stride.
  res[0] = 0;

  thrust::fill(_sideChannelData.begin(), _sideChannelData.end(), 0L);
  for (size_t i = 2; i < _sideChannelData.size(); i += nSideChannels)
    _sideChannelData[i] = 1L;
  psrdada_cpp::effelsberg::edd::countBitSet<<<1, 1024>>>(
    sideCD, nBlocks, 0, nSideChannels, 3,
    thrust::raw_pointer_cast(res.data()));
  cudaDeviceSynchronize();
  EXPECT_EQ(0U, res[0]);

  res[0] = 0;
  psrdada_cpp::effelsberg::edd::countBitSet<<<1, 1024>>>(
    sideCD, nBlocks, 0, nSideChannels, 2,
    thrust::raw_pointer_cast(res.data()));
  cudaDeviceSynchronize();
  EXPECT_EQ(nBlocks, res[0]);
}


TEST(GatedSpectrometer, array_sum) {

  const size_t NBLOCKS = 16 * 32;
  const size_t NTHREADS = 1024;

  size_t inputLength = 1 << 22 + 1 ;
  size_t dataLength = inputLength;
  ////zero pad input array
  //if (inputLength % (2 * NTHREADS) != 0)
  //  dataLength = (inputLength / (2 * NTHREADS) + 1) * 2 * NTHREADS;
  thrust::device_vector<float> data(dataLength);
  thrust::fill(data.begin(), data.begin() + inputLength, 1);
  //thrust::fill(data.begin() + inputLength, data.end(), 0);
  thrust::device_vector<float> blr(NTHREADS * 2);

  thrust::fill(blr.begin(), blr.end(), 0);

  psrdada_cpp::effelsberg::edd::array_sum<<<NBLOCKS, NTHREADS, NTHREADS* sizeof(float)>>>(thrust::raw_pointer_cast(data.data()), data.size(), thrust::raw_pointer_cast(blr.data()));
  psrdada_cpp::effelsberg::edd::array_sum<<<1, NTHREADS, NTHREADS* sizeof(float)>>>(thrust::raw_pointer_cast(blr.data()), blr.size(), thrust::raw_pointer_cast(blr.data()));

  EXPECT_EQ(size_t(blr[0]), inputLength);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}

