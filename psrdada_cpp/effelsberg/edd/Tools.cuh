#ifndef PSRDADA_CPP_EFFELSBERG_EDD_TOOLS_CUH
#define PSRDADA_CPP_EFFELSBERG_EDD_TOOLS_CUH

// collection of multi purpose kernels

// blocksize for the array sum kernel
#define array_sum_Nthreads 1024

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
/**
   * @brief      Sums all elements of an input array.
   *
   * @detail     The results is stored in an array with one value per launch
   *             block. Full reduction thus requires two kernel launches.
   *
   * @param      in. Input array.
   * @param      N. Size of input array.
   * @param      out. Output array.
   * */
__global__ void array_sum(float *in, size_t N, float *out);


/// Calculates 1/N \sum (x_i - 1/N offset)**2 per block
/// To calculate the std dev sum partial results of block using array sum
__global__ void scaled_square_offset_sum(float *in, size_t N, float* offset, float *out);

} // edd
} // effelsberg
} // psrdada_cpp

#endif //PSRDADA_CPP_EFFELSBERG_EDD_TOOLS_CUH
