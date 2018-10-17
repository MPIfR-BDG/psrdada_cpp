#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <cuda.h>
#include <cuComplex.h>
#include <vector>
#include <random>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <math.h>
#include <cmath>
#include <chrono>


/**
 *
 * Required inputs:
 * -- Nantennas total
 * -- Nchans
 * -- Coherent beam:
 *   -- antenna span (start, end)
 *   -- tscrunch
 *   -- fscrunch
 * -- Incoherent beam:
 *   -- antenna span (start, end)
 *   -- tscrunch
 *   -- fscrunch
 *
 */


/**
 *
 * How many threads needed?
 *  - Processing
 *  - Input
 *  - Output
 *  - Delays
 *
 */


class BeamformingPipeline
{
public:
    typedef char2 VoltageType;
    typedef char2 WeightsType;
    typedef char BeamType;
    typedef double TimeType;

public:
    BeamformingPipeline(PipelineConfig const& config);
    ~BeamformingPipeline();
    BeamformingPipeline(BeamformingPipeline const&) = delete;


    void async_process(VoltageType const* taftp_voltages, BeamType* btf_coherent_beams, BeamType* tf_incoherent_beam, TimeType current_time);
    void wait();

private:
    void update_weights(TimeType time);

private:
    WeightsType _weights;
    TimeType _current_weights_epoch;
    cudaStream_t _stream;
};


class WeightsManager
{
public:
    typedef char2 WeightsType;
    typedef double TimeType;

public:
    WeightsManager(PipelineConfig const& config);
    ~WeightsManager();
    WeightsManager(WeightsManager const&) == delete;

    WeightsType const* weights(TimeType epoch);

private:
    std::unique_ptr<DelayClient> _delay_client;
    TimeType _update_rate;
    TimeType _last_update;
    thrust::device_vector<WeightsType> _weights;
};


    float2 const * __restrict__ delay_models,
    char2 * __restrict__ weights,
    float const * __restrict__ channel_frequencies,
    int nantennas,
    int nbeams,
    int nchans,
    float tstart,
    float tstep,
    int ntsteps)

/**
 * @brief      Process a block of data on the GPU
 *
 * @param      taftp_voltages  The taftp voltages
 */
void BeamformingPipeline::async_process(VoltageType const* taftp_voltages, BeamType* btf_coherent_beams, BeamType* tf_incoherent_beam, TimeType epoch)
{

    //Assume data is on the GPU already;




    // Generate weights with padding (if not already generated)

    // Perform split transpose to extract data for coherent beamformer (with multiple-of-4 span)

    // Perform coherent beamforming using padded weights

    // Perform split transpose to extract data for incoherent beamformer (with multiple-of-2 span)

    // Perform incoherent beamforming

    // Copy coherent beam data to host

    // Copy incoherent beam data to host

    // Write data to DADA buffer

}

void BeamformingPipeline::update_weights(time_t epoch)
{
    if (epoch < _current_weights_epoch + _weights_span)
    {
        BOOST_LOG_DEBUG << "Weights do not need updated";
        return;
    }
    else
    {
        BOOST_LOG_DEBUG << "Weights are " << _current_weights_epoch - epoch - _weights_span << " seconds out of date";

    }





}
