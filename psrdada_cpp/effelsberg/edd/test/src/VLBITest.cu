#include "gtest/gtest.h"

#include <time.h>
#include <stdlib.h>

#include "psrdada_cpp/effelsberg/edd/VLBI.cuh"

#include "psrdada_cpp/cuda_utils.hpp"
#include "thrust/extrema.h"

TEST(VLBITest, 2_bit_pack_test)
{
    std::size_t n = 1024;
    thrust::device_vector<float>  input(n);
    thrust::device_vector<uint32_t>  output(n / 16);

    {
      float minV = -2;
      float maxV = 2;

      srand (time(NULL));
      for (int i =0; i < input.size(); i++)
      {
        input[i] = ((float(rand()) / RAND_MAX) - 0.5) * 2.5 * (maxV-minV) + maxV + minV;
      }

      thrust::fill(output.begin(), output.end(), 5);
      psrdada_cpp::effelsberg::edd::pack_2bit(input, output, minV, maxV);

      float step = (maxV - minV) / 3;
      float L2 = minV + step;
      float L3 = minV + 2 * step;
      float L4 = minV + 3 * step;

      for(int i = 0; i < input.size() / 16; i++)
      {
          uint64_t of = output[i];
          for (size_t j =0; j< 16; j++)
          {
            int a = ((of >> (j *2)) & 3);
            int k = i * 16 + j;
            if (input[k] >= L4)
              EXPECT_EQ(a, 3);
            else if (input[k] >= L3)
              EXPECT_EQ(a, 2);
            else if (input[k] >= L2)
              EXPECT_EQ(a, 1);
            else
              EXPECT_EQ(a, 0);
          }
      }
    }
}

//int main(int argc, char **argv) {
//  ::testing::InitGoogleTest(&argc, argv);
//
//  return RUN_ALL_TESTS();
//}
