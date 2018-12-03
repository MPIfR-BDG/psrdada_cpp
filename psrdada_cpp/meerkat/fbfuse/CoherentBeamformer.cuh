#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_COHERENTBEAMFORMER_HPP
#define PSRDADA_CPP_MEERKAT_FBFUSE_COHERENTBEAMFORMER_HPP

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/common.hpp"
#include "thrust/device_vector.h"
#include "cuda.h"

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace kernels {

/**
 * @brief      2 char4 instances...
 */
struct char4x2
{
    char4 x;
    char4 y;
};


/**
 * @brief      4 char2 instances...
 */
struct char2x4
{
    char2 x;
    char2 y;
    char2 z;
    char2 w;
};

/**
 * @brief      Wrapper for the DP4A int8 fused multiply add instruction
 *
 * @param      c     The output value
 * @param[in]  a     An integer composed of 4 chars
 * @param[in]  b     An integer composed of 4 chars
 *
 * @detail     If we treat a and b like to char4 instances, then the dp4a
 *             instruction performs the following:
 *
 *             c = (a.x * b.x) + (a.y * b.y) + (a.z * b.z) + (a.w * b.w)
 *
 * @note       The assembly instruction that underpins this operation (dp4a.s32.s32)
 *             is only available on compute 6.1 cards (GP102, GP104 and GP106 architectures).
 *             To use this properly the code must be *explicitly* compiled for 6.1 architectures
 *             using gencode. The PTX JIT compiler will not use this optimisation and the
 *             function will default to using standard small integer math (slow).
 */
__forceinline__ __device__
void dp4a(int &c, const int &a, const int &b);

/**
 * @brief      Transpose an int2 from a char2x4 to a char4x2.
 *
 * @param      input  The value to transpose
 *
 * @note       This is used to go from (for 4 sequential antennas):
 *
 *             [[real, imag],
 *              [real, imag],
 *              [real, imag],
 *              [real, imag]]
 *
 *             to
 *
 *             [[real, real, real, real],
 *              [imag, imag, imag, imag]]
 */
__forceinline__ __device__
int2 int2_transpose(int2 const &input);

/**
 * @brief      The coherent beamforming kernel
 *
 * @param      aptf_voltages  The aptf voltages (8 int8 complex values packed into int2)
 * @param      apbf_weights   The apbf weights (8 int8 complex values packed into int2)
 * @param      ftb_powers     The ftb powers
 * @param[in]  output_scale   The output scaling
 * @param[in]  output_offset  The output offset
 * @param[in]  nsamples       The number of samples in the aptf_voltages
 */
__global__
void bf_aptf_general_k(
    int2 const* __restrict__ aptf_voltages,
    int2 const* __restrict__ apbf_weights,
    int8_t* __restrict__ ftb_powers,
    float output_scale,
    float output_offset,
    int nsamples);

} //namespace kernels

/**
 * @brief      Class for coherent beamformer.
 */
class CoherentBeamformer
{
public:
    // FTPA order
    typedef thrust::device_vector<char2> VoltageVectorType;
    // TBTF order
    typedef thrust::device_vector<int8_t> PowerVectorType;
    // FBA order (assuming equal weight per polarisation)
    typedef thrust::device_vector<char2> WeightsVectorType;

public:
    /**
     * @brief      Constructs a CoherentBeamformer object.
     *
     * @param      config  The pipeline configuration
     */
    CoherentBeamformer(PipelineConfig const& config);
    ~CoherentBeamformer();
    CoherentBeamformer(CoherentBeamformer const&) = delete;

    /**
     * @brief      Form coherent beams
     *
     * @param      input    Input array of 8-bit voltages in FTPA order
     * @param      weights  8-bit beamforming weights in FTA order
     * @param      output   Output array of 8-bit powers in TBTF order
     * @param[in]  stream   The CUDA stream to use for processing
     */
    void beamform(VoltageVectorType const& input,
        WeightsVectorType const& weights,
        PowerVectorType& output,
        cudaStream_t stream);

private:
    PipelineConfig const& _config;
    std::size_t _size_per_sample;
    std::size_t _expected_weights_size;
};

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_COHERENTBEAMFORMER_HPP
