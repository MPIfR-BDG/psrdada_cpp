#include "psrdada_cpp/meerkat/fbfuse/WeightsManager.cuh"
#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/meerkat/fbfuse/DelayManager.cuh"
#include "psrdada_cpp/cuda_utils.hpp"
#include <thrust/device_vector.h>

#define TWOPI 6.283185307179586f

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
    int ntsteps)
{

    //for each loaded delay poly we can produce multiple epochs for one antenna, one beam, all frequencies and both pols
    //Different blocks should handle different beams (as antennas are on the inner dimension of the output product)

    //Basics of this kernel:
    //
    // gridDim.x is used for beams (there is a loop if you want to limit the grid size)
    // gridDim.y is used for channels (there is a loop if you want to limit the grid size)
    // blockDim.x is used for antennas (there is a loop if you want to limit the grid size)
    //
    // Time steps are handled in a the inner loop. As antennas are on the inner dimension of
    // both the input and the output array, all reads and writes should be coalesced.
    const int weights_per_beam = nantennas;
    const int weights_per_channel = weights_per_beam * nbeams;
    const int weights_per_time_step = weights_per_channel * nchans;

    float2 weight;
    char2 compressed_weight;
    //This isn't really needed as there will never be more than 64 antennas
    //However this makes this fucntion more flexible with smaller blocks

    for (int chan_idx = blockIdx.y; chan_idx < nchans; chan_idx += gridDim.y)
    {
        float frequency = channel_frequencies[chan_idx];
        int chan_offset = chan_idx * weights_per_channel; // correct

        for (int beam_idx = blockIdx.x; beam_idx < nbeams; beam_idx += gridDim.x)
        {
            int beam_offset = chan_offset + beam_idx * weights_per_beam; // correct

            for (int antenna_idx = threadIdx.x; antenna_idx < nantennas; antenna_idx+=blockDim.x)
            {
                float2 delay_model = delay_models[beam_idx * nantennas + antenna_idx]; // correct

                int antenna_offset = beam_offset + antenna_idx;

                for (int time_idx = threadIdx.y; time_idx < ntsteps; time_idx+=blockDim.y)
                {
                    //Calculates epoch offset
                    float t = tstart + time_idx * tstep;
                    float phase = (t * delay_model.x + delay_model.y) * frequency;
                    //This is possible as the magnitude of the weight is 1
                    //If we ever have to implement scalar weightings, this
                    //must change.
                    __sincosf(TWOPI * phase, &weight.y, &weight.x);
                    compressed_weight.x = (char) __float2int_rn(weight.x * 127.0f);
                    compressed_weight.y = (char) __float2int_rn(weight.y * 127.0f);
                    int output_idx = time_idx * weights_per_time_step + antenna_offset;
                    weights[output_idx] = compressed_weight;
                }
            }
        }
    }
}

} //namespace kernels


WeightsManager::WeightsManager(PipelineConfig const& config, DelayManager& delay_manager, cudaStream_t stream)
    : _config(config)
    , _delay_manager(delay_manager)
    , _stream(stream)
{
    std::size_t nbeams = _config.cb_nbeams();
    std::size_t nantennas = _config.cb_nantennas();
    BOOST_LOG_TRIVIAL(debug) << "Constructing WeightsManager instance to hold weights for "
    << nbeams << " beams and " << nantennas << " antennas";
    _weights.resize(nbeams * nantennas);
    // This should be an implicit copy to the device
    BOOST_LOG_TRIVIAL(debug) << "Copying channel frequencies to the GPU";
    _channel_frequencies = _config.channel_frequencies();
}

WeightsManager::~WeightsManager()
{
}

WeightsManager::WeightsVectorType const& WeightsManager::weights(TimeType epoch)
{
    // First we retrieve new delays if there are any.
    BOOST_LOG_TRIVIAL(debug) << "Requesting weights for epoch = " << epoch;
    DelayManager::DelayVectorType const& delays = _delay_manager.delays();
    DelayManager::DelayType const* delays_ptr = thrust::raw_pointer_cast(delays.data());
    WeightsType* weights_ptr = thrust::raw_pointer_cast(_weights.data());
    FreqType const* frequencies_ptr = thrust::raw_pointer_cast(_channel_frequencies.data());
    dim3 grid(_config.cb_nbeams(),
        _channel_frequencies.size(), 1);
    dim3 block(32, 32, 1);
    BOOST_LOG_TRIVIAL(debug) << "Launching weights generation kernel";
    kernels::generate_weights_k<<< grid, block, 0, _stream >>>(delays_ptr,
        weights_ptr, frequencies_ptr,
        _config.cb_nantennas(),
        _config.cb_nbeams(),
        _channel_frequencies.size(),
        epoch, 0.0, 1);
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_stream));
    BOOST_LOG_TRIVIAL(debug) << "Weights successfully generated";
    return _weights;
}

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

