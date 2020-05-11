#include "psrdada_cpp/effelsberg/edd/GatedSpectrometer.cuh"

#include "psrdada_cpp/dada_null_sink.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "gtest/gtest.h"

#include "thrust/device_vector.h"
#include "thrust/extrema.h"


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
//TEST(GatedSpectrometer, stokes_IQUV)
//{
//    float I,Q,U,V;
//    // No field
//    psrdada_cpp::effelsberg::edd::stokes_IQUV((float2){0.0f,0.0f}, (float2){0.0f,0.0f}, I, Q, U, V);
//    EXPECT_FLOAT_EQ(I, 0);
//    EXPECT_FLOAT_EQ(Q, 0);
//    EXPECT_FLOAT_EQ(U, 0);
//    EXPECT_FLOAT_EQ(V, 0);
//
//    // For p1 = Ex, p2 = Ey
//    // horizontal polarized
//    psrdada_cpp::effelsberg::edd::stokes_IQUV((float2){1.0f,0.0f}, (float2){0.0f,0.0f}, I, Q, U, V);
//    EXPECT_FLOAT_EQ(I, 1);
//    EXPECT_FLOAT_EQ(Q, 1);
//    EXPECT_FLOAT_EQ(U, 0);
//    EXPECT_FLOAT_EQ(V, 0);
//
//    // vertical polarized
//    psrdada_cpp::effelsberg::edd::stokes_IQUV((float2){0.0f,0.0f}, (float2){1.0f,0.0f}, I, Q, U, V);
//    EXPECT_FLOAT_EQ(I, 1);
//    EXPECT_FLOAT_EQ(Q, -1);
//    EXPECT_FLOAT_EQ(U, 0);
//    EXPECT_FLOAT_EQ(V, 0);
//
//    //linear +45 deg.
//    psrdada_cpp::effelsberg::edd::stokes_IQUV((float2){1.0f/std::sqrt(2),0.0f}, (float2){1.0f/std::sqrt(2),0.0f}, I, Q, U, V);
//    EXPECT_FLOAT_EQ(I, 1);
//    EXPECT_FLOAT_EQ(Q, 0);
//    EXPECT_FLOAT_EQ(U, 1);
//    EXPECT_FLOAT_EQ(V, 0);
//
//    //linear -45 deg.
//    psrdada_cpp::effelsberg::edd::stokes_IQUV((float2){-1.0f/std::sqrt(2),0.0f}, (float2){1.0f/std::sqrt(2),0.0f}, I, Q, U, V);
//    EXPECT_FLOAT_EQ(I, 1);
//    EXPECT_FLOAT_EQ(Q, 0);
//    EXPECT_FLOAT_EQ(U, -1);
//    EXPECT_FLOAT_EQ(V, 0);
//
//    //left circular
//    psrdada_cpp::effelsberg::edd::stokes_IQUV((float2){.0f,1.0f/std::sqrt(2)}, (float2){1.0f/std::sqrt(2),.0f}, I, Q, U, V);
//    EXPECT_FLOAT_EQ(I, 1);
//    EXPECT_FLOAT_EQ(Q, 0);
//    EXPECT_FLOAT_EQ(U, 0);
//    EXPECT_FLOAT_EQ(V, -1);
//
//    // right circular
//    psrdada_cpp::effelsberg::edd::stokes_IQUV((float2){.0f,-1.0f/std::sqrt(2)}, (float2){1.0f/std::sqrt(2),.0f}, I, Q, U, V);
//    EXPECT_FLOAT_EQ(I, 1);
//    EXPECT_FLOAT_EQ(Q, 0);
//    EXPECT_FLOAT_EQ(U, 0);
//    EXPECT_FLOAT_EQ(V, 1);
//}
//
//
//TEST(GatedSpectrometer, stokes_accumulate)
//{
//    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
//    size_t nchans = 8 * 1024 * 1024 + 1;
//    size_t naccumulate = 5;
//
//    thrust::device_vector<float2> P0(nchans * naccumulate);
//    thrust::device_vector<float2> P1(nchans * naccumulate);
//    thrust::fill(P0.begin(), P0.end(), (float2){0, 0});
//    thrust::fill(P1.begin(), P1.end(), (float2){0, 0});
//    thrust::device_vector<float> I(nchans);
//    thrust::device_vector<float> Q(nchans);
//    thrust::device_vector<float> U(nchans);
//    thrust::device_vector<float> V(nchans);
//    thrust::fill(I.begin(), I.end(), 0);
//    thrust::fill(Q.begin(), Q.end(), 0);
//    thrust::fill(U.begin(), U.end(), 0);
//    thrust::fill(V.begin(), V.end(), 0);
//
//    // This channel should be left circular polarized
//    size_t idx0 = 23;
//    for (int k = 0; k< naccumulate; k++)
//    {
//        size_t idx = idx0 + k * nchans;
//        P0[idx] = (float2){0.0f, 1.0f/std::sqrt(2)};
//        P1[idx] = (float2){1.0f/std::sqrt(2),0.0f};
//    }
//
//    psrdada_cpp::effelsberg::edd::stokes_accumulate<<<1024, 1024>>>(
//          thrust::raw_pointer_cast(P0.data()),
//          thrust::raw_pointer_cast(P1.data()),
//          thrust::raw_pointer_cast(I.data()),
//          thrust::raw_pointer_cast(Q.data()),
//          thrust::raw_pointer_cast(U.data()),
//          thrust::raw_pointer_cast(V.data()),
//          nchans,
//          naccumulate
//            );
//
//    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
//    thrust::pair<thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator> minmax;
//
//    minmax = thrust::minmax_element(I.begin(), I.end());
//    EXPECT_FLOAT_EQ(*minmax.first, 0);
//    EXPECT_FLOAT_EQ(*minmax.second, naccumulate);
//
//    minmax = thrust::minmax_element(Q.begin(), Q.end());
//    EXPECT_FLOAT_EQ(*minmax.first, 0);
//    EXPECT_FLOAT_EQ(*minmax.second, 0);
//
//    minmax = thrust::minmax_element(U.begin(), U.end());
//    EXPECT_FLOAT_EQ(*minmax.first, 0);
//    EXPECT_FLOAT_EQ(*minmax.second, 0);
//
//    minmax = thrust::minmax_element(V.begin(), V.end());
//    EXPECT_FLOAT_EQ(*minmax.first, -1. * naccumulate);
//    EXPECT_FLOAT_EQ(*minmax.second, 0);
//}
//


TEST(GatedSpectrometer, GatingKernel)
{
  const size_t blockSize = 1024;
  const size_t nBlocks = 16 * 1024;

  thrust::device_vector<float> G0(blockSize * nBlocks);
  thrust::device_vector<float> G1(blockSize * nBlocks);
  thrust::device_vector<uint64_t> _sideChannelData(nBlocks);
  thrust::device_vector<psrdada_cpp::effelsberg::edd::uint64_cu> _nG0(nBlocks);
  thrust::device_vector<psrdada_cpp::effelsberg::edd::uint64_cu> _nG1(nBlocks);
  thrust::device_vector<float> baseLineG0(1);
  thrust::device_vector<float> baseLineG1(1);

  thrust::device_vector<float> baseLineG0_update(1);
  thrust::device_vector<float> baseLineG1_update(1);
  thrust::fill(G0.begin(), G0.end(), 42);
  thrust::fill(G1.begin(), G1.end(), 23);
  thrust::fill(_sideChannelData.begin(), _sideChannelData.end(), 0);

  // everything to G0
  {
    thrust::fill(_nG0.begin(), _nG0.end(), 0);
    thrust::fill(_nG1.begin(), _nG1.end(), 0);
    baseLineG0[0] = -3;
    baseLineG1[0] = -4;
    baseLineG0_update[0] = 0;
    baseLineG1_update[0] = 0;

    const uint64_t *sideCD =
        (uint64_t *)(thrust::raw_pointer_cast(_sideChannelData.data()));
    psrdada_cpp::effelsberg::edd::gating<<<1024 , 1024>>>(
          thrust::raw_pointer_cast(G0.data()),
          thrust::raw_pointer_cast(G1.data()), sideCD,
          G0.size(), blockSize, 0, 1,
          0,
          thrust::raw_pointer_cast(baseLineG0.data()),
          thrust::raw_pointer_cast(baseLineG1.data()),
          thrust::raw_pointer_cast(baseLineG0_update.data()),
          thrust::raw_pointer_cast(baseLineG1_update.data()),
          thrust::raw_pointer_cast(_nG0.data()),
          thrust::raw_pointer_cast(_nG1.data())
          );

    thrust::pair<thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator> minmax;
    minmax = thrust::minmax_element(G0.begin(), G0.end());
    EXPECT_EQ(*minmax.first, 42);
    EXPECT_EQ(*minmax.second, 42);

    minmax = thrust::minmax_element(G1.begin(), G1.end());
    EXPECT_EQ(*minmax.first, -4.);
    EXPECT_EQ(*minmax.second, -4.);

    EXPECT_EQ(_nG0[0], G0.size());
    EXPECT_EQ(_nG1[0], 0u);

    EXPECT_FLOAT_EQ(42.f, baseLineG0_update[0] / (_nG0[0] + 1E-121));
    EXPECT_FLOAT_EQ(0.f, baseLineG1_update[0] / (_nG1[0] + 1E-121));
  }

  // everything to G1 // with baseline -5
  {
    thrust::fill(_nG0.begin(), _nG0.end(), 0);
    thrust::fill(_nG1.begin(), _nG1.end(), 0);
    baseLineG0[0] = 5.;
    baseLineG1[0] = -2;
    baseLineG0_update[0] = 0;
    baseLineG1_update[0] = 0;

    thrust::fill(_sideChannelData.begin(), _sideChannelData.end(), 1L);
    const uint64_t *sideCD =
        (uint64_t *)(thrust::raw_pointer_cast(_sideChannelData.data()));
    psrdada_cpp::effelsberg::edd::gating<<<1024, 1024>>>(
          thrust::raw_pointer_cast(G0.data()),
          thrust::raw_pointer_cast(G1.data()), sideCD,
          G0.size(), blockSize, 0, 1,
          0,
          thrust::raw_pointer_cast(baseLineG0.data()),
          thrust::raw_pointer_cast(baseLineG1.data()),
          thrust::raw_pointer_cast(baseLineG0_update.data()),
          thrust::raw_pointer_cast(baseLineG1_update.data()),
          thrust::raw_pointer_cast(_nG0.data()),
          thrust::raw_pointer_cast(_nG1.data())
          );
    thrust::pair<thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator> minmax;
    minmax = thrust::minmax_element(G0.begin(), G0.end());
    EXPECT_EQ(*minmax.first, 5.);
    EXPECT_EQ(*minmax.second, 5.);

    minmax = thrust::minmax_element(G1.begin(), G1.end());
    EXPECT_EQ(*minmax.first, 42);
    EXPECT_EQ(*minmax.second, 42);

    EXPECT_EQ(_nG0[0], 0u);
    EXPECT_EQ(_nG1[0], G1.size());

    EXPECT_FLOAT_EQ(0.f, baseLineG0_update[0] / (_nG0[0] + 1E-121));
    EXPECT_FLOAT_EQ(42.f, baseLineG1_update[0] / (_nG1[0] + 1E-121));
  }
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
