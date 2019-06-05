#include <time.h>
#include <stdlib.h>
#include "psrdada_cpp/effelsberg/edd/Packer.cuh"

#include "gtest/gtest.h"



class PackerTest: public ::testing::Test
{
  protected:
    thrust::device_vector<float>  input;
    thrust::device_vector<uint32_t>  output;
    float minV;
    float maxV;
    cudaStream_t stream;

    void SetUp() override {
      input.resize(1024);
      minV = -2;
      maxV = 2;

      srand (time(NULL));
      for (int i =0; i < input.size(); i++)
      {
        input[i] = ((float(rand()) / RAND_MAX) - 0.5) * 2.5 * (maxV-minV) + maxV + minV;
      }

      cudaStreamCreate(&stream);
    }

    void TearDown()
    {
      cudaStreamDestroy(stream);
    }

    void checkOutputSize(unsigned int bit_depth)
    {
      //SCOPED_TRACE("Input Bitdepth: " << bit_depth );
      EXPECT_EQ(output.size(), input.size() / (32 / bit_depth));
    }


    void checkOutputValues(unsigned int bit_depth)
    {

      float step = (maxV - minV) /  ((1 << bit_depth) - 1);

      const size_t nbp = 32 / bit_depth;
      for(int i = 0; i < input.size() / nbp; i++)
      {
          uint32_t of = output[i];
          for (size_t j =0; j< nbp; j++)
          {
            uint32_t a = ((of >> (j * bit_depth)) & ((1 << bit_depth) - 1));
            int k = i * nbp + j;

            if (input[k] <= minV)
              EXPECT_EQ(0, int (a)) << "input[ " << k << "] = " << input[k];
            else if (input[k] >= maxV)
              EXPECT_EQ(((1 << bit_depth) - 1), int (a)) << "input[ " << k << "] = " << input[k];
            else
              EXPECT_EQ(int((input[k] - minV) / step), int(a)) << "input[ " << k << "] = " << input[k];
          }
      }
    }
    };

TEST_F(PackerTest, 2bit)
{
  psrdada_cpp::effelsberg::edd::pack<2>(input, output, minV, maxV, stream);
  checkOutputSize(2);
  float step = (maxV - minV) / 3;
  float L2 = minV + step;
  float L3 = minV + 2 * step;
  float L4 = minV + 3 * step;

  const size_t nbp = 16; // 16 samples per output value
  for(int i = 0; i < input.size() / nbp; i++)
  {
      uint32_t of = output[i];
      for (size_t j =0; j< nbp; j++)
      {
        uint32_t a = ((of >> (j *2)) & 3);
        int k = i * nbp + j;
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


TEST_F(PackerTest, 4bit)
{
  psrdada_cpp::effelsberg::edd::pack<4>(input, output, minV, maxV, stream);
  checkOutputSize(4);
  checkOutputValues(4);
}

TEST_F(PackerTest, 8bit)
{
  psrdada_cpp::effelsberg::edd::pack<8>(input, output, minV, maxV, stream);
  checkOutputSize(8);
  checkOutputValues(8);
}

TEST_F(PackerTest, 16bit)
{
  psrdada_cpp::effelsberg::edd::pack<16>(input, output, minV, maxV, stream);
  checkOutputSize(16);
  checkOutputValues(16);
}
