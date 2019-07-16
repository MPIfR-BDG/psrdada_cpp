#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_INCOHERENTBEAMFORMER_HPP
#define PSRDADA_CPP_MEERKAT_FBFUSE_INCOHERENTBEAMFORMER_HPP

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/common.hpp"
#include "thrust/device_vector.h"
#include "cuda.h"

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace kernels {

__global__
void icbf_taftp_general_k(
    char4 const* __restrict__ taftp_voltages,
    int8_t* __restrict__ tf_powers,
    float output_scale,
    float output_offset,
    int nsamples);

} //namespace kernels

/**
 * @brief      Class for incoherent beamforming.
 */
class IncoherentBeamformer
{
public:
    // TAFTP order
    typedef thrust::device_vector<char2> VoltageVectorType;
    // TF order
    typedef thrust::device_vector<int8_t> PowerVectorType;

public:
    /**
     * @brief      Constructs an instance of the IncoherentBeamformer class
     *
     * @param      config  The pipeline config
     */
    IncoherentBeamformer(PipelineConfig const& config);
    ~IncoherentBeamformer();
    IncoherentBeamformer(IncoherentBeamformer const&) = delete;

    /**
     * @brief      Form incoherent beams
     *
     * @param      input   Input array of 8-bit voltages in TAFTP order
     * @param      output  Output array of 8-bit powers in TF order
     * @param[in]  stream  The CUDA stream to use for processing
     */
    void beamform(VoltageVectorType const& input,
        PowerVectorType& output,
        cudaStream_t stream);

private:
    PipelineConfig const& _config;
    std::size_t _size_per_aftp_block;
};

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_INCOHERENTBEAMFORMER_HPP
