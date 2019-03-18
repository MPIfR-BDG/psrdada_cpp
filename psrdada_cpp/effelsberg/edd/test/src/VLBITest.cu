#include "gtest/gtest.h"

#include "psrdada_cpp/cuda_utils.hpp"
#include <random>
#include "psrdada_cpp/effelsberg/edd/VLBI.cuh"
#include "thrust/extrema.h"

TEST(VLBITest, 2_bit_pack_test)
{
    std::size_t n = 1024;
    thrust::device_vector<float>  input(n);
    thrust::device_vector<uint32_t>  output(n);
    {
      thrust::fill(input.begin(), input.end(), 0);
      thrust::fill(output.begin(), output.end(), 5);

      psrdada_cpp::effelsberg::edd::pack_2bit(input, output, 0, 3);

      EXPECT_EQ(output.size(), input.size() / 16);
      thrust::pair<thrust::device_vector<uint32_t>::iterator, thrust::device_vector<uint32_t>::iterator> minmax;
      minmax = thrust::minmax_element(output.begin(), output.end());
      EXPECT_EQ(0, *minmax.first);
      EXPECT_EQ(0, *minmax.second);
    }

    {
      thrust::fill(input.begin(), input.end(), 1);
      thrust::fill(output.begin(), output.end(), 5);

      psrdada_cpp::effelsberg::edd::pack_2bit(input, output, 0, 3);
      thrust::pair<thrust::device_vector<uint32_t>::iterator, thrust::device_vector<uint32_t>::iterator> minmax;
      minmax = thrust::minmax_element(output.begin(), output.end());

      EXPECT_EQ((uint32_t)0b0101010101010101010101010101010101010101, *minmax.first);
      EXPECT_EQ((uint32_t)0b0101010101010101010101010101010101010101, *minmax.second);
    }

    {
      thrust::fill(input.begin(), input.end(), 2);
      thrust::fill(output.begin(), output.end(), 5);

      psrdada_cpp::effelsberg::edd::pack_2bit(input, output, 0, 3);
      thrust::pair<thrust::device_vector<uint32_t>::iterator, thrust::device_vector<uint32_t>::iterator> minmax;
      minmax = thrust::minmax_element(output.begin(), output.end());

      EXPECT_EQ((uint32_t)0b1010101010101010101010101010101010101010, *minmax.first);
      EXPECT_EQ((uint32_t)0b1010101010101010101010101010101010101010, *minmax.second);
    }

    {
      thrust::fill(input.begin(), input.end(), 3);
      thrust::fill(output.begin(), output.end(), 5);

      psrdada_cpp::effelsberg::edd::pack_2bit(input, output, 0, 3);
      thrust::pair<thrust::device_vector<uint32_t>::iterator, thrust::device_vector<uint32_t>::iterator> minmax;
      minmax = thrust::minmax_element(output.begin(), output.end());

      EXPECT_EQ((uint32_t)0b1111111111111111111111111111111111111111, *minmax.first);
      EXPECT_EQ((uint32_t)0b1111111111111111111111111111111111111111, *minmax.second);
    }

    {
      thrust::fill(input.begin(), input.end(), 4);
      thrust::fill(output.begin(), output.end(), 5);

      psrdada_cpp::effelsberg::edd::pack_2bit(input, output, 0, 3);
      thrust::pair<thrust::device_vector<uint32_t>::iterator, thrust::device_vector<uint32_t>::iterator> minmax;
      minmax = thrust::minmax_element(output.begin(), output.end());

      EXPECT_EQ((uint32_t)0b1111111111111111111111111111111111111111, *minmax.first);
      EXPECT_EQ((uint32_t)0b1111111111111111111111111111111111111111, *minmax.second);
    }

}

//int main(int argc, char **argv) {
//  ::testing::InitGoogleTest(&argc, argv);
//
//  return RUN_ALL_TESTS();
//}
