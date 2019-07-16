#ifndef PSRDADA_CPP_EFFELSBERG_EDD_TOOLS_CUH
#define PSRDADA_CPP_EFFELSBERG_EDD_TOOLS_CUH

#include <cstdint>

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
__global__ void scaled_square_residual_sum(float *in, size_t N, float* offset, float *out);


/// Create a bit mask with 1 between first and lastBit (inclusive), zero otherwise;
uint32_t bitMask(uint32_t firstBit, uint32_t lastBit);

/// Squeeze a value into the specified bitrange of the target
void setBitsWithValue(uint32_t &target, uint32_t firstBit, uint32_t lastBit, uint32_t value);

/// Get numerical value from the specified bits in the target
uint32_t getBitsValue(const uint32_t &target, uint32_t firstBit, uint32_t lastBit);

} // edd
} // effelsberg
} // psrdada_cpp

#endif //PSRDADA_CPP_EFFELSBERG_EDD_TOOLS_CUH
