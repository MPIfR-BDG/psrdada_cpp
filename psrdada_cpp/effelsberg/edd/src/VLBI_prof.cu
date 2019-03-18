#include <thrust/device_vector.h>
#include <cuda_profiler_api.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include "psrdada_cpp/effelsberg/edd/VLBI.cuh"


struct GenRand
{
    __device__
    float operator () (int idx)
    {
        thrust::default_random_engine randEng;
        thrust::uniform_real_distribution<float> uniDist;
        randEng.discard(idx);
        return uniDist(randEng);
    }
};


int main()
{

  const size_t n = 1024 * 1024 * 1024 / 4;
    thrust::device_vector<float>  input(n);
    thrust::device_vector<uint32_t>  output(n);
  
    thrust::fill(input.begin(), input.end(), .5);
    thrust::fill(output.begin(), output.end(), 5);

    cudaDeviceSynchronize();
    psrdada_cpp::effelsberg::edd::pack_2bit(input, output, 0, 1, 0);
    cudaDeviceSynchronize();
    std::cout << input[0] << std::endl; 
    std::cout << input[1] << std::endl; 
    std::cout << input[2] << std::endl; 
    std::cout << input[3] << std::endl; 
    std::cout << (int) output[0] << std::endl; 
    for (size_t i = 0; i<10; i++)
      std::cout << i <<": " << output[i] << std::endl; 
    cudaProfilerStop();
}
