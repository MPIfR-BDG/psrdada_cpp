#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_WEIGHTSMANAGER_HPP
#define PSRDADA_CPP_MEERKAT_FBFUSE_WEIGHTSMANAGER_HPP

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/meerkat/fbfuse/DelayManager.cuh"
#include <thrust/device_vector.h>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace kernels {

__global__
void generate_weights_k(
    float2 const * __restrict__ delay_models,
    char2 * __restrict__ weights,
    float const * __restrict__ channel_frequencies,
    int nantennas,
    int nbeams,
    int nchans,
    float tstart,
    float tstep,
    int ntsteps);

} //namespace kernels

class WeightsManager
{
public:
    typedef char2 WeightsType;
    typedef thrust::device_vector<WeightsType> WeightsVectorType;
    typedef float FreqType;
    typedef thrust::device_vector<FreqType> FreqVectorType;
    typedef float TimeType;
    typedef DelayManager::DelayVectorType DelayVectorType;

public:

    /**
     * @brief      Create a new weights mananger object
     *
     * @param      config          The pipeline configuration
     */
    WeightsManager(PipelineConfig const& config, cudaStream_t stream);
    ~WeightsManager();
    WeightsManager(WeightsManager const&) = delete;

    /**
     * @brief      Calculate beamforming weights for a given epock
     *
     * @param[in]  delays A vector containing delay models for all beams (produced by
     *             an instance of DelayManager)
     *
     * @param[in]  epoch  The epoch at which to evaluate the given delay models
     *
     * @detail     No check is performed here on whether the provided epoch is in
     *             the bounds of the current delay polynomial. The assumption here
     *             is the we are running real time and as such there is no large
     *             latency between the reception of a set of delay models and a
     *             request for weights from that model. If this is not the case
     *             a more sophisticated mechanism for control of the delays into
     *             the beamformer will have to be developed.
     *
     * @note       This function is not thread-safe!!! Competing calls will overwrite
     *             the memory of the _weights object.
     *
     * @return     A thrust device vector containing the generated weights
     */
    WeightsVectorType const& weights(DelayVectorType const& delays, TimeType epoch);

private:
    PipelineConfig const& _config;
    cudaStream_t _stream;
    WeightsVectorType _weights;
    FreqVectorType _channel_frequencies;
};

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif // PSRDADA_CPP_MEERKAT_FBFUSE_WEIGHTSMANAGER_HPP


